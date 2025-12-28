"""
🌐 API Tab Component
====================
UI for hosting Bagley API - local or worldwide
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextEdit, QLineEdit, QProgressBar, QFileDialog, QListWidget,
        QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QMessageBox, QTabWidget, QSplitter,
        QFormLayout, QPlainTextEdit, QListWidgetItem
    )
    from PySide6.QtCore import Qt, QTimer, QThread, Signal
    from PySide6.QtGui import QFont, QColor
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QTextEdit, QLineEdit, QProgressBar, QFileDialog, QListWidget,
            QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
            QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
            QHeaderView, QMessageBox, QTabWidget, QSplitter,
            QFormLayout, QPlainTextEdit, QListWidgetItem
        )
        from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal
        from PyQt6.QtGui import QFont, QColor
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False

# Import API server
try:
    from bagley.ui.api_server import (
        BagleyAPIServer, APIConfig, HostMode, APIKeyManager
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Import hardware monitor
try:
    from bagley.ui.hardware_monitor import get_hardware_monitor, SystemStats
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False


# Colors
COLORS = {
    "bg_dark": "#0a0a0f",
    "bg_medium": "#12121a",
    "bg_light": "#1a1a2e",
    "accent": "#6366f1",
    "accent_hover": "#818cf8",
    "success": "#22c55e",
    "warning": "#eab308",
    "error": "#ef4444",
    "text": "#ffffff",
    "text_dim": "#a1a1aa",
    "border": "#27272a",
}


if QT_AVAILABLE:

    class HardwareMonitorWidget(QWidget):
        """
        🖥️ Always-on hardware monitoring display
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
            self._start_monitoring()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Compact horizontal layout
            monitor_frame = QFrame()
            monitor_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['bg_medium']};
                    border-radius: 8px;
                    padding: 8px;
                }}
            """)
            
            h_layout = QHBoxLayout(monitor_frame)
            h_layout.setSpacing(16)
            
            # GPU Section
            gpu_widget = QWidget()
            gpu_layout = QVBoxLayout(gpu_widget)
            gpu_layout.setContentsMargins(0, 0, 0, 0)
            gpu_layout.setSpacing(2)
            
            self.gpu_label = QLabel("🎮 GPU")
            self.gpu_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
            gpu_layout.addWidget(self.gpu_label)
            
            self.gpu_temp_label = QLabel("-- °C")
            self.gpu_temp_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 14px; font-weight: bold;")
            gpu_layout.addWidget(self.gpu_temp_label)
            
            self.gpu_util_bar = QProgressBar()
            self.gpu_util_bar.setFixedHeight(6)
            self.gpu_util_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    background-color: {COLORS['bg_dark']};
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {COLORS['accent']};
                    border-radius: 3px;
                }}
            """)
            gpu_layout.addWidget(self.gpu_util_bar)
            
            h_layout.addWidget(gpu_widget)
            
            # Divider
            divider = QFrame()
            divider.setFixedWidth(1)
            divider.setStyleSheet(f"background-color: {COLORS['border']};")
            h_layout.addWidget(divider)
            
            # CPU Section
            cpu_widget = QWidget()
            cpu_layout = QVBoxLayout(cpu_widget)
            cpu_layout.setContentsMargins(0, 0, 0, 0)
            cpu_layout.setSpacing(2)
            
            self.cpu_label = QLabel("💻 CPU")
            self.cpu_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
            cpu_layout.addWidget(self.cpu_label)
            
            self.cpu_temp_label = QLabel("-- °C")
            self.cpu_temp_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 14px; font-weight: bold;")
            cpu_layout.addWidget(self.cpu_temp_label)
            
            self.cpu_util_bar = QProgressBar()
            self.cpu_util_bar.setFixedHeight(6)
            self.cpu_util_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    background-color: {COLORS['bg_dark']};
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {COLORS['success']};
                    border-radius: 3px;
                }}
            """)
            cpu_layout.addWidget(self.cpu_util_bar)
            
            h_layout.addWidget(cpu_widget)
            
            # Divider
            divider2 = QFrame()
            divider2.setFixedWidth(1)
            divider2.setStyleSheet(f"background-color: {COLORS['border']};")
            h_layout.addWidget(divider2)
            
            # RAM Section
            ram_widget = QWidget()
            ram_layout = QVBoxLayout(ram_widget)
            ram_layout.setContentsMargins(0, 0, 0, 0)
            ram_layout.setSpacing(2)
            
            self.ram_label = QLabel("🧠 RAM")
            self.ram_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
            ram_layout.addWidget(self.ram_label)
            
            self.ram_usage_label = QLabel("--/-- GB")
            self.ram_usage_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 14px; font-weight: bold;")
            ram_layout.addWidget(self.ram_usage_label)
            
            self.ram_bar = QProgressBar()
            self.ram_bar.setFixedHeight(6)
            self.ram_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    background-color: {COLORS['bg_dark']};
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {COLORS['warning']};
                    border-radius: 3px;
                }}
            """)
            ram_layout.addWidget(self.ram_bar)
            
            h_layout.addWidget(ram_widget)
            
            layout.addWidget(monitor_frame)
        
        def _start_monitoring(self):
            """Start hardware monitoring"""
            if MONITOR_AVAILABLE:
                monitor = get_hardware_monitor()
                monitor.add_callback(self._update_stats)
            else:
                # Fallback timer
                self.timer = QTimer(self)
                self.timer.timeout.connect(self._fallback_update)
                self.timer.start(2000)
        
        def _update_stats(self, stats: 'SystemStats'):
            """Update display with new stats"""
            # GPU
            if stats.gpus:
                gpu = stats.gpus[0]
                temp = gpu.temperature
                self.gpu_temp_label.setText(f"{temp:.0f} °C")
                self.gpu_util_bar.setValue(int(gpu.utilization))
                
                # Color based on temp
                if temp > 80:
                    self.gpu_temp_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 14px; font-weight: bold;")
                elif temp > 70:
                    self.gpu_temp_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 14px; font-weight: bold;")
                else:
                    self.gpu_temp_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;")
            
            # CPU
            temp = stats.cpu.temperature
            self.cpu_temp_label.setText(f"{temp:.0f} °C" if temp > 0 else "N/A")
            self.cpu_util_bar.setValue(int(stats.cpu.utilization))
            
            if temp > 0:
                if temp > 85:
                    self.cpu_temp_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 14px; font-weight: bold;")
                elif temp > 70:
                    self.cpu_temp_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 14px; font-weight: bold;")
                else:
                    self.cpu_temp_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;")
            
            # RAM
            self.ram_usage_label.setText(f"{stats.ram.used:.1f}/{stats.ram.total:.0f} GB")
            self.ram_bar.setValue(int(stats.ram.percent))
        
        def _fallback_update(self):
            """Fallback update without full monitor"""
            try:
                import psutil
                
                # CPU
                cpu_percent = psutil.cpu_percent()
                self.cpu_util_bar.setValue(int(cpu_percent))
                
                # RAM
                mem = psutil.virtual_memory()
                self.ram_usage_label.setText(f"{mem.used/(1024**3):.1f}/{mem.total/(1024**3):.0f} GB")
                self.ram_bar.setValue(int(mem.percent))
                
            except:
                pass


    class APITabWidget(QWidget):
        """
        🌐 API Hosting Tab
        Configure and run API server
        """
        
        server_started = Signal()
        server_stopped = Signal()
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.server: Optional[BagleyAPIServer] = None
            self.key_manager = APIKeyManager() if API_AVAILABLE else None
            
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            layout.setSpacing(16)
            layout.setContentsMargins(16, 16, 16, 16)
            
            # Header with hardware monitor
            header = QHBoxLayout()
            
            title = QLabel("🌐 API Server")
            title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            header.addWidget(title)
            
            header.addStretch()
            
            # Status indicator
            self.status_indicator = QLabel("● Stopped")
            self.status_indicator.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 14px;")
            header.addWidget(self.status_indicator)
            
            layout.addLayout(header)
            
            # Hardware monitor (always visible)
            self.hw_monitor = HardwareMonitorWidget()
            layout.addWidget(self.hw_monitor)
            
            # Main content
            content = QHBoxLayout()
            
            # Left: Configuration
            left_panel = QFrame()
            left_panel.setFixedWidth(400)
            left_panel.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['bg_medium']};
                    border-radius: 12px;
                }}
            """)
            
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(16, 16, 16, 16)
            left_layout.setSpacing(12)
            
            # Hosting mode
            mode_group = QGroupBox("🌍 Hosting Mode")
            mode_layout = QVBoxLayout(mode_group)
            
            self.mode_combo = QComboBox()
            self.mode_combo.addItems([
                "🏠 Local Only (127.0.0.1)",
                "🏢 LAN (Your Network)",
                "🌐 Worldwide (Internet)"
            ])
            self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            mode_layout.addWidget(self.mode_combo)
            
            self.mode_warning = QLabel("")
            self.mode_warning.setWordWrap(True)
            self.mode_warning.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
            mode_layout.addWidget(self.mode_warning)
            
            left_layout.addWidget(mode_group)
            
            # Port settings
            port_group = QGroupBox("⚙️ Settings")
            port_layout = QFormLayout(port_group)
            
            self.port_spin = QSpinBox()
            self.port_spin.setRange(1024, 65535)
            self.port_spin.setValue(8000)
            port_layout.addRow("Port:", self.port_spin)
            
            self.auth_check = QCheckBox("Require API Key")
            self.auth_check.setChecked(True)
            port_layout.addRow(self.auth_check)
            
            self.rate_limit_spin = QSpinBox()
            self.rate_limit_spin.setRange(1, 1000)
            self.rate_limit_spin.setValue(60)
            port_layout.addRow("Rate Limit/min:", self.rate_limit_spin)
            
            self.cors_check = QCheckBox("Enable CORS")
            self.cors_check.setChecked(True)
            port_layout.addRow(self.cors_check)
            
            left_layout.addWidget(port_group)
            
            # SSL settings
            ssl_group = QGroupBox("🔒 SSL (HTTPS)")
            ssl_layout = QVBoxLayout(ssl_group)
            
            self.ssl_check = QCheckBox("Enable SSL")
            self.ssl_check.setChecked(False)
            self.ssl_check.toggled.connect(self._on_ssl_toggled)
            ssl_layout.addWidget(self.ssl_check)
            
            self.cert_path = QLineEdit()
            self.cert_path.setPlaceholderText("Certificate file path...")
            self.cert_path.setEnabled(False)
            ssl_layout.addWidget(self.cert_path)
            
            self.key_path = QLineEdit()
            self.key_path.setPlaceholderText("Private key file path...")
            self.key_path.setEnabled(False)
            ssl_layout.addWidget(self.key_path)
            
            left_layout.addWidget(ssl_group)
            
            # Start/Stop buttons
            btn_layout = QHBoxLayout()
            
            self.start_btn = QPushButton("🚀 Start Server")
            self.start_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: #16a34a;
                }}
            """)
            self.start_btn.clicked.connect(self._start_server)
            btn_layout.addWidget(self.start_btn)
            
            self.stop_btn = QPushButton("⏹️ Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['error']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #dc2626;
                }}
                QPushButton:disabled {{
                    background-color: {COLORS['bg_dark']};
                    color: {COLORS['text_dim']};
                }}
            """)
            self.stop_btn.clicked.connect(self._stop_server)
            btn_layout.addWidget(self.stop_btn)
            
            left_layout.addLayout(btn_layout)
            
            # URL display
            self.url_label = QLabel("")
            self.url_label.setStyleSheet(f"""
                color: {COLORS['accent']};
                font-family: 'Consolas', monospace;
                padding: 8px;
                background-color: {COLORS['bg_dark']};
                border-radius: 6px;
            """)
            self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            left_layout.addWidget(self.url_label)
            
            left_layout.addStretch()
            
            content.addWidget(left_panel)
            
            # Right: API Keys and Logs
            right_panel = QTabWidget()
            right_panel.setStyleSheet(f"""
                QTabWidget::pane {{
                    background-color: {COLORS['bg_medium']};
                    border-radius: 12px;
                    border: 1px solid {COLORS['border']};
                }}
                QTabBar::tab {{
                    background-color: {COLORS['bg_dark']};
                    color: {COLORS['text_dim']};
                    padding: 8px 16px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }}
                QTabBar::tab:selected {{
                    background-color: {COLORS['bg_medium']};
                    color: {COLORS['text']};
                }}
            """)
            
            # API Keys tab
            keys_tab = QWidget()
            keys_layout = QVBoxLayout(keys_tab)
            
            keys_header = QHBoxLayout()
            keys_header.addWidget(QLabel("🔑 API Keys"))
            
            self.generate_key_btn = QPushButton("+ Generate Key")
            self.generate_key_btn.clicked.connect(self._generate_key)
            keys_header.addWidget(self.generate_key_btn)
            
            keys_layout.addLayout(keys_header)
            
            self.keys_table = QTableWidget(0, 4)
            self.keys_table.setHorizontalHeaderLabels(["Name", "Created", "Requests", "Status"])
            self.keys_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.keys_table.setStyleSheet(f"""
                QTableWidget {{
                    background-color: {COLORS['bg_dark']};
                    border: none;
                    gridline-color: {COLORS['border']};
                }}
                QHeaderView::section {{
                    background-color: {COLORS['bg_medium']};
                    color: {COLORS['text']};
                    padding: 8px;
                    border: none;
                }}
            """)
            keys_layout.addWidget(self.keys_table)
            
            right_panel.addTab(keys_tab, "🔑 API Keys")
            
            # Logs tab
            logs_tab = QWidget()
            logs_layout = QVBoxLayout(logs_tab)
            
            self.logs_output = QPlainTextEdit()
            self.logs_output.setReadOnly(True)
            self.logs_output.setStyleSheet(f"""
                QPlainTextEdit {{
                    background-color: {COLORS['bg_dark']};
                    color: {COLORS['text']};
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    border: none;
                    border-radius: 8px;
                }}
            """)
            logs_layout.addWidget(self.logs_output)
            
            right_panel.addTab(logs_tab, "📝 Logs")
            
            # Endpoints tab
            endpoints_tab = QWidget()
            endpoints_layout = QVBoxLayout(endpoints_tab)
            
            endpoints_info = QLabel("""
<h3>📡 Available Endpoints</h3>
<table style="width:100%">
<tr><td><b>POST /chat</b></td><td>Chat/conversation</td></tr>
<tr><td><b>POST /image</b></td><td>Image generation</td></tr>
<tr><td><b>POST /video</b></td><td>Video generation</td></tr>
<tr><td><b>POST /tts</b></td><td>Text-to-speech</td></tr>
<tr><td><b>POST /3d</b></td><td>3D model generation</td></tr>
<tr><td><b>GET /health</b></td><td>Server health check</td></tr>
<tr><td><b>GET /docs</b></td><td>API documentation</td></tr>
</table>

<h3>📋 Example Request</h3>
<pre>
curl -X POST http://localhost:8000/chat \\
  -H "Authorization: Bearer bg_your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
</pre>
            """)
            endpoints_info.setStyleSheet(f"color: {COLORS['text']}; padding: 16px;")
            endpoints_info.setTextFormat(Qt.TextFormat.RichText)
            endpoints_layout.addWidget(endpoints_info)
            
            right_panel.addTab(endpoints_tab, "📡 Endpoints")
            
            content.addWidget(right_panel, stretch=1)
            
            layout.addLayout(content, stretch=1)
            
            # Load existing keys
            self._refresh_keys()
        
        def _on_mode_changed(self, index: int):
            """Handle hosting mode change"""
            if index == 2:  # Worldwide
                self.mode_warning.setText(
                    "⚠️ WARNING: This will expose your API to the internet!\n"
                    "Make sure to enable authentication and consider using SSL."
                )
            elif index == 1:  # LAN
                self.mode_warning.setText(
                    "ℹ️ API will be accessible from devices on your local network."
                )
            else:
                self.mode_warning.setText("")
        
        def _on_ssl_toggled(self, checked: bool):
            """Handle SSL toggle"""
            self.cert_path.setEnabled(checked)
            self.key_path.setEnabled(checked)
        
        def _start_server(self):
            """Start the API server"""
            if not API_AVAILABLE:
                QMessageBox.warning(
                    self, "Not Available",
                    "API server not available. Install: pip install fastapi uvicorn"
                )
                return
            
            # Create config
            modes = [HostMode.LOCAL, HostMode.LAN, HostMode.WORLDWIDE]
            config = APIConfig(
                host_mode=modes[self.mode_combo.currentIndex()],
                port=self.port_spin.value(),
                require_auth=self.auth_check.isChecked(),
                rate_limit_per_minute=self.rate_limit_spin.value(),
                cors_enabled=self.cors_check.isChecked(),
                ssl_enabled=self.ssl_check.isChecked(),
                ssl_cert_path=self.cert_path.text(),
                ssl_key_path=self.key_path.text()
            )
            
            # Create and start server
            self.server = BagleyAPIServer(config)
            self.server.key_manager = self.key_manager
            
            # Start in background
            self.server.start_background()
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_indicator.setText("● Running")
            self.status_indicator.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px;")
            self.url_label.setText(f"🔗 {self.server.get_url()}")
            
            self._log(f"Server started at {self.server.get_url()}")
            self.server_started.emit()
        
        def _stop_server(self):
            """Stop the API server"""
            if self.server:
                self.server.stop()
                self.server = None
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_indicator.setText("● Stopped")
            self.status_indicator.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 14px;")
            self.url_label.setText("")
            
            self._log("Server stopped")
            self.server_stopped.emit()
        
        def _generate_key(self):
            """Generate a new API key"""
            if not self.key_manager:
                return
            
            # Simple dialog for key name
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(
                self, "Generate API Key",
                "Enter a name for this key:"
            )
            
            if ok and name:
                key = self.key_manager.generate_key(name)
                
                # Show the key (only time it's visible)
                QMessageBox.information(
                    self, "API Key Generated",
                    f"Your new API key:\n\n{key}\n\n"
                    "⚠️ Copy this now! It won't be shown again."
                )
                
                self._refresh_keys()
                self._log(f"Generated new API key: {name}")
        
        def _refresh_keys(self):
            """Refresh API keys table"""
            if not self.key_manager:
                return
            
            keys = self.key_manager.list_keys()
            self.keys_table.setRowCount(len(keys))
            
            for i, key_data in enumerate(keys):
                self.keys_table.setItem(i, 0, QTableWidgetItem(key_data.get('name', 'Unknown')))
                self.keys_table.setItem(i, 1, QTableWidgetItem(key_data.get('created', '')[:10]))
                self.keys_table.setItem(i, 2, QTableWidgetItem(str(key_data.get('requests_total', 0))))
                
                status = "Active" if key_data.get('active', True) else "Revoked"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(
                    QColor(COLORS['success'] if status == "Active" else COLORS['error'])
                )
                self.keys_table.setItem(i, 3, status_item)
        
        def _log(self, message: str):
            """Add message to log"""
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs_output.appendPlainText(f"[{timestamp}] {message}")


    # ==================== Exports ====================
    
    __all__ = [
        'HardwareMonitorWidget',
        'APITabWidget',
    ]

else:
    __all__ = []
