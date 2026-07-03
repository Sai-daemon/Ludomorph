/**
 * GameAI Activator — KWin Script for KDE Plasma
 *
 * Exposes a D-Bus interface at org.kde.KWin /WindowActivator that allows
 * the Game AI Agent to focus windows by title or WM_CLASS.
 *
 * Installation:
 *   1. Copy this directory to: ~/.local/share/kwin/scripts/gameai-activator/
 *   2. Enable in System Settings → Window Management → KWin Scripts
 *   3. Log out and back in (or run: kwin_x11 --replace & for X11)
 *
 * Usage from the agent:
 *   qdbus org.kde.KWin /WindowActivator activateWindow "Game Title"
 *
 * D-Bus interface: org.kde.KWin.WindowActivator
 * Methods:
 *   activateWindow(title: string) → boolean
 */

function activateWindow(title) {
    const clients = workspace.clientList();
    for (const client of clients) {
        // Match by window title (case-insensitive substring)
        const caption = (client.caption || "").toLowerCase();
        const resourceClass = (client.resourceClass || "").toLowerCase();
        const search = title.toLowerCase();

        if (caption.includes(search) || resourceClass.includes(search)) {
            workspace.activeClient = client;
            return true;
        }
    }
    return false;
}

// Register D-Bus service
registerShortcut(
    "GameAI Activator",
    "GameAI Activator",
    "",  // no default keybinding
    function () {
        // no-op; activation is via D-Bus only
    }
);

// Expose via D-Bus
callDBus(
    "org.kde.KWin",
    "/WindowActivator",
    "org.kde.KWin.WindowActivator",
    "activateWindow",
    activateWindow
);