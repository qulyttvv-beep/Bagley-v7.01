"""
🖥️ Bagley Desktop UI
Ultra-modern interface with Chat + Training tabs
"""

from bagley.ui.app import BagleyApp
from bagley.ui.components import ChatWindow, MediaViewer, Settings, Sidebar

# v2 with training tab
try:
    from bagley.ui.app_v2 import BagleyAppV2, ChatTab, TrainingTab, run_app
except ImportError:
    BagleyAppV2 = None
    ChatTab = None
    TrainingTab = None
    run_app = None

__all__ = [
    # v1
    "BagleyApp",
    "ChatWindow",
    "MediaViewer",
    "Settings",
    "Sidebar",
    # v2
    "BagleyAppV2",
    "ChatTab",
    "TrainingTab",
    "run_app",
]
