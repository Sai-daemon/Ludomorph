"""
OCR Module — Phase 2.3

Tesseract wrapper with configurable preprocessing, LRU image-hash
cache (TTL 2 s), per-character confidence scoring, and automatic
retry with alternate preprocessing.

Async execution via a dedicated ``ThreadPoolExecutor(max_workers=1)``
and ``OMP_THREAD_LIMIT=1`` — as mandated by architecture.md §5.5 and
§7.1.

Usage::

    from src.ocr_module import OCRModule, OCRConfig
    from src.region_profile import RegionConfig

    ocr = OCRModule(OCRConfig(tessdata_path="/usr/share/tessdata"))
    result = await ocr.recognize_region(frame, region_config)
    print(result.text, result.confidence)
"""

from __future__ import annotations

import hashlib
import os
import time as time_module
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mandatory: prevent OpenMP thread explosion inside Tesseract (§5.1, §5.5)
# ---------------------------------------------------------------------------

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# ---------------------------------------------------------------------------
# Dedicated Tesseract thread pool (§5.5, §7.1)
# ---------------------------------------------------------------------------

_tesseract_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tesseract")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OCRConfig:
    """Global OCR configuration (loaded from config.json)."""

    tessdata_path: str = "/usr/share/tesseract-ocr/5/tessdata"
    """Path to the tessdata directory containing .traineddata files."""

    default_lang: str = "eng"
    """Default Tesseract language code."""

    default_psm: int = 6
    """Default Page Segmentation Mode (§5.4)."""

    default_oem: int = 1
    """Default OCR Engine Mode (§5.4). 1 = LSTM only."""

    confidence_threshold: float = 0.6
    """If mean confidence drops below this, retry is attempted (§5.7)."""

    max_retries: int = 2
    """Maximum number of retries with alternate preprocessing (§5.7)."""

    use_cache: bool = True
    """Globally enable/disable the image-hash cache (§5.6)."""

    cache_ttl_seconds: float = 2.0
    """TTL for cached OCR results (§5.6)."""

    cache_max_per_region: int = 100
    """Max cache entries per region name (LRU eviction, §5.6)."""

    timeout_ms: float = 250.0
    """Per-region OCR timeout including retries (§7.3)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Result of a single OCR call on one region (§5.8).

    Attributes:
        text: Recognised text (may be empty on failure).
        confidence: Mean per‑character confidence 0.0‑1.0.
        processing_time_ms: Wall‑clock time for this call (preprocess + Tesseract).
        region_name: Name of the region (from ``RegionConfig.name``).
        success: ``True`` if confidence ≥ threshold, ``False`` otherwise.
    """

    text: str = ""
    confidence: float = 0.0
    processing_time_ms: int = 0
    region_name: str = ""
    success: bool = False

    @classmethod
    def empty(cls, region_name: str) -> "OCRResult":
        """Factory for a failed / empty result (used as fallback)."""
        return cls(region_name=region_name, success=False)


# ---------------------------------------------------------------------------
# Preprocessing helpers (stateless, pure functions)
# ---------------------------------------------------------------------------


def _grayscale(pil_image: "Image.Image") -> "Image.Image":
    """Convert to grayscale (always the first step, §5.3 item 1)."""
    from PIL import Image as _Image

    return pil_image.convert("L")


def _upscale(pil_image: "Image.Image", factor: int = 2) -> "Image.Image":
    """Nearest‑neighbour upscaling (§5.3 item 2)."""
    import numpy as _np

    arr = _np.array(pil_image)
    h, w = arr.shape[:2]
    new_w, new_h = w * factor, h * factor

    if len(arr.shape) == 2:
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        from PIL import Image as _Image

        return _Image.fromarray(resized)
    else:
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        from PIL import Image as _Image

        return _Image.fromarray(resized)


def _denoise(pil_image: "Image.Image", kernel_size: int = 3) -> "Image.Image":
    """Median blur for noise reduction (§5.3 item 3)."""
    import numpy as _np

    arr = _np.array(pil_image)
    # Ensure odd kernel
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    denoised = cv2.medianBlur(arr, k)
    from PIL import Image as _Image

    return _Image.fromarray(denoised)


def _threshold_otsu(pil_image: "Image.Image") -> "Image.Image":
    """Otsu binarisation (§5.3 item 4, default threshold method)."""
    import numpy as _np

    arr = _np.array(pil_image)
    if len(arr.shape) > 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    from PIL import Image as _Image

    return _Image.fromarray(thresh)


def _threshold_adaptive(pil_image: "Image.Image", block_size: int = 11, c: int = 2) -> "Image.Image":
    """Adaptive Gaussian thresholding (§5.3 item 4 alternative)."""
    import numpy as _np

    arr = _np.array(pil_image)
    if len(arr.shape) > 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # block_size must be odd
    bs = block_size if block_size % 2 == 1 else block_size + 1
    thresh = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, c)
    from PIL import Image as _Image

    return _Image.fromarray(thresh)


def _morphology(pil_image: "Image.Image", operation: str = "erode", kernel_size: int = 3) -> "Image.Image":
    """Erosion or dilation (§5.3 item 5)."""
    import numpy as _np

    arr = _np.array(pil_image)
    if len(arr.shape) > 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "erode":
        processed = cv2.erode(arr, kernel, iterations=1)
    elif operation == "dilate":
        processed = cv2.dilate(arr, kernel, iterations=1)
    else:
        return pil_image
    from PIL import Image as _Image

    return _Image.fromarray(processed)


def _deskew(pil_image: "Image.Image", max_angle: float = 5.0) -> "Image.Image":
    """Deskew text image by up to ±max_angle degrees (§5.3 item 6)."""
    import math as _math
    import numpy as _np

    arr = _np.array(pil_image)
    if len(arr.shape) > 2:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr.copy()

    # Threshold to isolate text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all non-zero points
    coords = _np.column_stack(_np.where(binary > 0))
    if len(coords) < 10:
        return pil_image  # Not enough data to estimate skew

    # Minimum area rectangle
    rect = cv2.minAreaRect(coords.astype(_np.float32))
    angle = rect[2]

    # Normalise angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Clamp
    if abs(angle) > max_angle:
        return pil_image

    # Rotate
    h, w = arr.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        arr,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    from PIL import Image as _Image

    return _Image.fromarray(rotated)


# ---------------------------------------------------------------------------
# LRU + TTL Cache
# ---------------------------------------------------------------------------


class _LRUTTLCache:
    """A per‑region image‑hash cache with TTL and LRU eviction (§5.6).

    Cache key = ``(region_name, image_hash, preprocess_hash)``.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 2.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, OCRResult]] = OrderedDict()

    def get(self, key: str) -> OCRResult | None:
        """Return cached result if not expired, else None."""
        now = time_module.monotonic()
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, result = entry
        if now - ts > self._ttl_seconds:
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return result

    def set(self, key: str, result: OCRResult) -> None:
        """Store a new result, evicting LRU if over capacity."""
        self._store[key] = (time_module.monotonic(), result)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# OCRModule
# ---------------------------------------------------------------------------


class OCRModule:
    """Async Tesseract OCR with preprocessing, caching, and retry (§5).

    Each instance wraps a single Tesseract language/configuration and
    holds per‑region caches.  All Tesseract calls are dispatched to
    the global ``_tesseract_executor`` thread pool (max_workers=1) so
    at most one heavy OCR run is active at any time (§5.5, §7.1).

    Typical usage::

        config = OCRConfig(tessdata_path="...")
        ocr = OCRModule(config)
        result = await ocr.recognize_region(frame, region_config)
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config

        # Per‑region LRU caches
        self._caches: dict[str, _LRUTTLCache] = {}

        # Validate Tesseract on init
        self._validate_tesseract()

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def recognize_region(
        self,
        image: "np.ndarray | Image.Image",
        region_config: "RegionConfig",
        *,
        override_bounds: tuple[int, int, int, int] | None = None,
        override_name: str | None = None,
    ) -> OCRResult:
        """Run OCR on the region defined by *region_config*.

        Args:
            image: Full‑frame image (NumPy BGR/grayscale or PIL Image).
                The region is cropped internally using
                ``region_config.bounds`` unless *override_bounds* is given.
            region_config: A ``RegionConfig`` with bounds, preprocess
                steps, and optional OCR overrides.
            override_bounds: Optional ``(x1, y1, x2, y2)`` that replaces
                ``region_config.bounds`` for this call.  Used by the
                dynamic‑region locator to supply runtime‑resolved bounds.
            override_name: Optional name suffix appended to
                ``region_config.name`` for logging/cache isolation.

        Returns:
            ``OCRResult`` — always succeeds (failure → ``success=False``).
        """
        import asyncio

        region_name = region_config.name
        if override_name:
            region_name = f"{region_name}#{override_name}"
        t_start = time_module.monotonic()

        try:
            # 1. Crop
            pil_img = self._to_pil(image)
            crop_bounds = override_bounds if override_bounds is not None else region_config.bounds
            cropped = self._crop_region(pil_img, crop_bounds)

            # 2. Preprocess
            preprocessed, preprocess_hash = self._preprocess_image(cropped, region_config.preprocess)

            # 3. Hash for cache key
            img_hash = self._image_hash(preprocessed)

            # 4. Check cache
            if self.config.use_cache:
                cache = self._get_cache(region_name)
                cache_key = f"{img_hash}:{preprocess_hash}"
                cached = cache.get(cache_key)
                if cached is not None:
                    elapsed_ms = int((time_module.monotonic() - t_start) * 1000)
                    logger.debug(
                        f"OCR cache hit for '{region_name}' "
                        f"(text={cached.text[:40]!r}, conf={cached.confidence:.2f})"
                    )
                    return OCRResult(
                        text=cached.text,
                        confidence=cached.confidence,
                        processing_time_ms=elapsed_ms,
                        region_name=region_name,
                        success=cached.success,
                    )

            # 5. Build Tesseract config string
            config_str = self._build_config_str(region_config)

            # 6. Run Tesseract (async via thread pool)
            try:
                text, confidence = await asyncio.wait_for(
                    self._run_tesseract_with_confidence(preprocessed, config_str),
                    timeout=self.config.timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                elapsed_ms = int((time_module.monotonic() - t_start) * 1000)
                logger.warning(
                    f"OCR timed out for '{region_name}' after {elapsed_ms}ms "
                    f"(limit {self.config.timeout_ms}ms)"
                )
                return OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time_ms=elapsed_ms,
                    region_name=region_name,
                    success=False,
                )

            # 7. Retry with alternate preprocessing if confidence is low
            #    Skip retries when confidence is literally 0.00 — no text was
            #    found at all (blank region, no glyphs).  Retrying different
            #    preprocessing won't help and just wastes time.
            retry_count = 0
            while confidence < self.config.confidence_threshold and retry_count < self.config.max_retries:
                if confidence == 0.0:
                    logger.debug(
                        f"OCR confidence is 0.00 for '{region_name}' — "
                        f"no text detected, skipping retries."
                    )
                    break
                retry_count += 1
                logger.debug(
                    f"OCR low confidence ({confidence:.2f} < {self.config.confidence_threshold}) "
                    f"for '{region_name}'; retry {retry_count}/{self.config.max_retries}"
                )

                # Alternate preprocessing: add sharpening / different threshold
                alt_steps = self._alternate_preprocess_steps(region_config.preprocess, retry_count)
                alt_preprocessed, alt_hash = self._preprocess_image(cropped, alt_steps)

                try:
                    text, confidence = await asyncio.wait_for(
                        self._run_tesseract_with_confidence(alt_preprocessed, config_str),
                        timeout=self.config.timeout_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"OCR retry {retry_count} timed out for '{region_name}'")
                    break

            # 8. Build result
            elapsed_ms = int((time_module.monotonic() - t_start) * 1000)
            success = confidence >= self.config.confidence_threshold

            result = OCRResult(
                text=text.strip() if text else "",
                confidence=round(confidence, 4),
                processing_time_ms=elapsed_ms,
                region_name=region_name,
                success=success,
            )

            # 9. Cache result
            if self.config.use_cache:
                cache = self._get_cache(region_name)
                cache_key = f"{img_hash}:{preprocess_hash}"
                cache.set(cache_key, result)

            if not success:
                logger.debug(
                    f"OCR low confidence for '{region_name}': "
                    f"text={text[:40]!r}, conf={confidence:.2f}"
                )

            return result

        except Exception as exc:
            elapsed_ms = int((time_module.monotonic() - t_start) * 1000)
            logger.warning(f"OCR failed for '{region_name}': {exc}")
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time_ms=elapsed_ms,
                region_name=region_name,
                success=False,
            )

    # ------------------------------------------------------------------
    # Internal: preprocessing pipeline
    # ------------------------------------------------------------------

    def _preprocess_image(
        self,
        pil_image: "Image.Image",
        steps: list[str],
    ) -> tuple["Image.Image", str]:
        """Apply preprocessing steps in order and return (image, hash of steps).

        The step strings are parsed from ``RegionConfig.preprocess``,
        e.g. ``["grayscale", "upscale(2x)", "denoise", "threshold:otsu"]``.
        """
        processed = pil_image

        for step in steps:
            parsed = self._parse_step(step)
            if parsed is None:
                continue
            name, kwargs = parsed
            try:
                processed = self._apply_preprocess_step(processed, name, kwargs)
            except Exception as exc:
                logger.debug(f"Preprocess step '{step}' failed for OCR: {exc}")

        step_hash = hashlib.sha256("|".join(steps).encode()).hexdigest()[:16]
        return processed, step_hash

    # ------------------------------------------------------------------
    # Internal: Tesseract execution (blocking, runs in thread pool)
    # ------------------------------------------------------------------

    async def _run_tesseract_with_confidence(
        self,
        pil_image: "Image.Image",
        config_str: str,
    ) -> tuple[str, float]:
        """Run Tesseract in the dedicated thread pool.

        Returns (text, mean_confidence).
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _tesseract_executor,
            self._tesseract_blocking,
            pil_image,
            config_str,
        )

    def _tesseract_blocking(self, pil_image: "Image.Image", config_str: str) -> tuple[str, float]:
        """Synchronous Tesseract call — runs in the thread pool.

        Returns (raw_text, mean_confidence 0.0‑1.0).
        """
        import pytesseract

        if self.config.tessdata_path:
            os.environ.setdefault("TESSDATA_PREFIX", self.config.tessdata_path)

        # Get per‑character confidence data
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=config_str)

        # Collect confidence values for all recognised words (conf != -1)
        confidences: list[float] = []
        for i, conf_val in enumerate(data["conf"]):
            word_text = (data["text"][i] or "").strip()
            if conf_val != -1 and word_text:
                confidences.append(float(conf_val))

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Get full text
        text = pytesseract.image_to_string(pil_image, config=config_str)

        return text, mean_conf / 100.0  # Tesseract conf is 0‑100

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(image: "np.ndarray | Image.Image") -> "Image.Image":
        """Convert NumPy array (or PIL Image) to a PIL Image."""
        from PIL import Image as _Image

        if isinstance(image, _Image.Image):
            return image

        if isinstance(image, np.ndarray):
            # Handle grayscale vs colour
            if len(image.shape) == 2:
                return _Image.fromarray(image, mode="L")
            if image.shape[2] == 3:
                return _Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if image.shape[2] == 4:
                return _Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            return _Image.fromarray(image[:, :, :3])

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _crop_region(pil_image: "Image.Image", bounds: tuple[int, int, int, int]) -> "Image.Image":
        """Crop to (x1, y1, x2, y2) with safe clamping to image edges."""
        x1, y1, x2, y2 = bounds
        w, h = pil_image.size
        # Clamp to image dimensions (PIL uses half-open intervals, so x2≤w is valid)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if x1 >= x2 or y1 >= y2:
            logger.debug(f"Invalid crop bounds after clamping: ({x1},{y1},{x2},{y2})")
            return pil_image

        return pil_image.crop((x1, y1, x2, y2))

    @staticmethod
    def _image_hash(pil_image: "Image.Image") -> str:
        """SHA‑256 hash of image raw bytes (for cache key)."""
        return hashlib.sha256(pil_image.tobytes()).hexdigest()

    def _get_cache(self, region_name: str) -> _LRUTTLCache:
        """Get or create the per‑region LRU cache."""
        if region_name not in self._caches:
            self._caches[region_name] = _LRUTTLCache(
                max_size=self.config.cache_max_per_region,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
        return self._caches[region_name]

    def _build_config_str(self, region_config: "RegionConfig") -> str:
        """Build Tesseract ``--psm N --oem M -c tessedit_...`` config string.

        Uses per‑region overrides from ``region_config.ocr_config``,
        falling back to global ``OCRConfig`` defaults.
        """
        ocr_cfg = region_config.ocr_config

        lang = self.config.default_lang
        psm = ocr_cfg.get("psm", self.config.default_psm)
        oem = ocr_cfg.get("oem", self.config.default_oem)

        parts = [f"-l {lang}", f"--psm {psm}", f"--oem {oem}"]

        # Character whitelist
        whitelist = ocr_cfg.get("whitelist")
        if whitelist:
            parts.append(f'-c tessedit_char_whitelist="{whitelist}"')

        # Custom words
        custom_words = ocr_cfg.get("custom_words")
        if custom_words:
            parts.append(f'-c user_words_file="{custom_words}"')

        return " ".join(parts)

    @staticmethod
    def _parse_step(step_str: str) -> tuple[str, dict[str, Any]] | None:
        """Parse a preprocessing step string like ``"upscale(2x)"`` or ``"threshold:otsu"``.

        Returns ``(method_name, kwargs_dict)`` or ``None`` if unrecognised.
        """
        s = step_str.strip().lower()
        if not s:
            return None

        # Handle "method(key=val)" form, e.g. "upscale(2x)"
        if "(" in s and s.endswith(")"):
            method, args_str = s[:-1].split("(", 1)
            method = method.strip()
            kwargs: dict[str, Any] = {}
            if args_str:
                for part in args_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        kwargs[k.strip()] = v.strip()
                    else:
                        # Positional arg — store as "value"
                        kwargs["value"] = part
            return method, kwargs

        # Handle "method:subtype" form, e.g. "threshold:otsu", "threshold:adaptive"
        if ":" in s:
            method, sub_type = s.split(":", 1)
            return method.strip(), {"value": sub_type.strip()}

        # Simple method name (no args)
        return s, {}

    def _apply_preprocess_step(
        self,
        pil_image: "Image.Image",
        method: str,
        kwargs: dict[str, Any],
    ) -> "Image.Image":
        """Dispatch a single preprocessing step by name."""
        if method == "grayscale":
            return _grayscale(pil_image)

        if method == "upscale":
            factor_str = kwargs.get("value", "2x")
            factor = int(factor_str.replace("x", ""))
            return _upscale(pil_image, factor)

        if method == "denoise":
            k = int(kwargs.get("kernel", kwargs.get("value", "3")))
            return _denoise(pil_image, k)

        if method == "threshold":
            sub_type = kwargs.get("value", kwargs.get("type", "otsu"))
            if sub_type == "adaptive":
                bs = int(kwargs.get("block_size", "11"))
                c = int(kwargs.get("c", "2"))
                return _threshold_adaptive(pil_image, bs, c)
            return _threshold_otsu(pil_image)

        if method == "erode":
            k = int(kwargs.get("kernel", kwargs.get("value", "3")))
            return _morphology(pil_image, "erode", k)

        if method == "dilate":
            k = int(kwargs.get("kernel", kwargs.get("value", "3")))
            return _morphology(pil_image, "dilate", k)

        if method == "deskew":
            max_angle = float(kwargs.get("max_angle", kwargs.get("value", "5.0")))
            return _deskew(pil_image, max_angle)

        logger.debug(f"Unknown preprocess method: '{method}' — skipping")
        return pil_image

    @staticmethod
    def _alternate_preprocess_steps(original_steps: list[str], retry_number: int) -> list[str]:
        """Return alternate preprocessing steps for retry N.

        Retry 1: replace threshold with adaptive (or add denoise + sharpen).
        Retry 2: add erode + dilate to clean noise.
        """
        steps = list(original_steps)

        if retry_number == 1:
            # Replace Otsu with Adaptive, or add denoise if missing
            replaced = False
            for i, s in enumerate(steps):
                if s.startswith("threshold"):
                    steps[i] = "threshold:adaptive"
                    replaced = True
                    break
            if not replaced and "denoise" not in steps:
                steps.insert(1, "denoise")

        elif retry_number == 2:
            # Aggressive: denoise + erode + dilate
            if "denoise" not in steps:
                steps.insert(1, "denoise")
            if "erode" not in steps:
                steps.append("erode")
            if "dilate" not in steps:
                steps.append("dilate")

        return steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tesseract(self) -> None:
        """Check that Tesseract is available and configured.

        Logs a warning (not an error) if unavailable — OCR will simply
        fail gracefully at runtime.
        """
        try:
            import pytesseract

            if self.config.tessdata_path:
                os.environ.setdefault("TESSDATA_PREFIX", self.config.tessdata_path)

            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract {version} detected")
        except ImportError:
            logger.warning("pytesseract not installed — OCR will be unavailable")
        except Exception as exc:
            logger.warning(f"Tesseract check failed: {exc}")


# ---------------------------------------------------------------------------
# Module-level cleanup helper
# ---------------------------------------------------------------------------


def shutdown_ocr_executor() -> None:
    """Shut down the global Tesseract thread pool (call on app exit)."""
    _tesseract_executor.shutdown(wait=True)
    logger.debug("Tesseract executor shut down")