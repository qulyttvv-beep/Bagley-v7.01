"""
🖥️ Bagley Desktop UI
Ultra-modern interface with Chat + Training + API tabs
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

# Hardware monitoring
try:
    from bagley.ui.hardware_monitor import (
        HardwareMonitor, GPUStats, CPUStats, SystemStats,
        get_hardware_monitor, get_smart_logger
    )
except ImportError:
    HardwareMonitor = None
    get_hardware_monitor = None

# API server
try:
    from bagley.ui.api_server import (
        BagleyAPIServer, APIConfig, HostMode,
        create_api_server
    )
except ImportError:
    BagleyAPIServer = None
    create_api_server = None

# API tab
try:
    from bagley.ui.api_tab import APITabWidget, HardwareMonitorWidget
except ImportError:
    APITabWidget = None
    HardwareMonitorWidget = None

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
    # Hardware
    "HardwareMonitor",
    "get_hardware_monitor",
    "get_smart_logger",
    # API
    "BagleyAPIServer",
    "APIConfig",
    "HostMode",
    "create_api_server",
    "APITabWidget",
    "HardwareMonitorWidget",
]
