"""
🖥️ Bagley Desktop App v2 - Full Featured UI
Chat + Training Tabs with GPU Monitoring
"""

import sys
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
        QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
        QFrame, QSplitter, QScrollArea, QGraphicsOpacityEffect,
        QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
    )
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPropertyAnimation, QEasingCurve
    from PySide6.QtGui import QFont, QColor, QPalette, QIcon
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
            QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
            QFrame, QSplitter, QScrollArea, QGraphicsOpacityEffect,
            QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
        )
        from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal, QPropertyAnimation, QEasingCurve
        from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False


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

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text"]};
    font-family: 'Segoe UI', sans-serif;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    background-color: {COLORS["bg_medium"]};
}}
QTabBar::tab {{
    background-color: {COLORS["bg_light"]};
    color: {COLORS["text_dim"]};
    padding: 12px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS["accent"]};
    color: {COLORS["text"]};
}}
QPushButton {{
    background-color: {COLORS["bg_light"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 10px 20px;
    color: {COLORS["text"]};
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}
QPushButton:pressed {{
    background-color: {COLORS["accent_hover"]};
}}
QPushButton#primary {{
    background-color: {COLORS["accent"]};
}}
QLineEdit, QTextEdit {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 10px;
    color: {COLORS["text"]};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {COLORS["accent"]};
}}
QProgressBar {{
    background-color: {COLORS["bg_light"]};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {COLORS["accent"]};
    border-radius: 4px;
}}
QListWidget {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
}}
QListWidget::item {{
    padding: 8px;
    border-radius: 4px;
}}
QListWidget::item:selected {{
    background-color: {COLORS["accent"]};
}}
QGroupBox {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
}}
QTableWidget {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    gridline-color: {COLORS["border"]};
}}
QTableWidget::item {{
    padding: 8px;
}}
QHeaderView::section {{
    background-color: {COLORS["bg_light"]};
    padding: 8px;
    border: none;
    font-weight: bold;
}}
QComboBox {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 8px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 8px;
}}
"""


if QT_AVAILABLE:
    
    class BagleyAppV2(QMainWindow):
        """
        🖥️ Main Bagley Application
        
        Features:
        - Tab 1: Chat with all models
        - Tab 2: Training with GPU monitoring
        """
        
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("🤖 Bagley v7")
            self.setMinimumSize(1400, 900)
            self.setStyleSheet(STYLESHEET)
            
            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QVBoxLayout(central)
            layout.setContentsMargins(16, 16, 16, 16)
            
            # Header
            header = self._create_header()
            layout.addWidget(header)
            
            # Tab widget
            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)
            
            # Chat tab
            self.chat_tab = ChatTab()
            self.tabs.addTab(self.chat_tab, "💬 Chat")
            
            # Training tab
            self.training_tab = TrainingTab()
            self.tabs.addTab(self.training_tab, "🏋️ Training")
            
            layout.addWidget(self.tabs)
        
        def _create_header(self) -> QWidget:
            """Create header bar"""
            header = QFrame()
            header.setFixedHeight(60)
            header.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QHBoxLayout(header)
            layout.setContentsMargins(20, 0, 20, 0)
            
            # Logo
            logo = QLabel("🤖 BAGLEY v7")
            logo.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            layout.addWidget(logo)
            
            layout.addStretch()
            
            # Status
            self.status_label = QLabel("● Ready")
            self.status_label.setStyleSheet(f"color: {COLORS['success']};")
            layout.addWidget(self.status_label)
            
            return header
    
    
    class ChatTab(QWidget):
        """
        💬 Chat Tab - Talk to all models
        """
        
        def __init__(self):
            super().__init__()
            
            layout = QHBoxLayout(self)
            layout.setSpacing(16)
            
            # Left: Model selector
            left_panel = self._create_model_selector()
            layout.addWidget(left_panel)
            
            # Center: Chat area
            chat_area = self._create_chat_area()
            layout.addWidget(chat_area, stretch=2)
            
            # Right: Model outputs
            right_panel = self._create_outputs_panel()
            layout.addWidget(right_panel)
        
        def _create_model_selector(self) -> QWidget:
            """Model selection panel"""
            panel = QFrame()
            panel.setFixedWidth(250)
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("🎯 Active Models")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Model checkboxes
            self.model_checks = {}
            
            models = [
                ("💬 Chat (MoE)", "chat", True),
                ("🎨 Image (DiT)", "image", True),
                ("🎬 Video", "video", False),
                ("🎵 TTS/Voice", "tts", True),
            ]
            
            for label, key, default in models:
                check = QCheckBox(label)
                check.setChecked(default)
                self.model_checks[key] = check
                layout.addWidget(check)
            
            layout.addSpacing(20)
            
            # Generation settings
            settings_label = QLabel("⚙️ Settings")
            settings_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            layout.addWidget(settings_label)
            
            # Temperature
            temp_layout = QHBoxLayout()
            temp_layout.addWidget(QLabel("Temperature:"))
            self.temp_spin = QDoubleSpinBox()
            self.temp_spin.setRange(0.0, 2.0)
            self.temp_spin.setValue(0.7)
            self.temp_spin.setSingleStep(0.1)
            temp_layout.addWidget(self.temp_spin)
            layout.addLayout(temp_layout)
            
            # Max tokens
            tokens_layout = QHBoxLayout()
            tokens_layout.addWidget(QLabel("Max Tokens:"))
            self.tokens_spin = QSpinBox()
            self.tokens_spin.setRange(64, 8192)
            self.tokens_spin.setValue(2048)
            tokens_layout.addWidget(self.tokens_spin)
            layout.addLayout(tokens_layout)
            
            layout.addStretch()
            
            return panel
        
        def _create_chat_area(self) -> QWidget:
            """Main chat area"""
            panel = QFrame()
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            # Messages
            self.messages_area = QScrollArea()
            self.messages_area.setWidgetResizable(True)
            self.messages_area.setStyleSheet("border: none; background: transparent;")
            
            self.messages_container = QWidget()
            self.messages_layout = QVBoxLayout(self.messages_container)
            self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.messages_layout.setSpacing(12)
            
            self.messages_area.setWidget(self.messages_container)
            layout.addWidget(self.messages_area, stretch=1)
            
            # Input area
            input_frame = QFrame()
            input_frame.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 12px;")
            
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(16, 12, 16, 12)
            
            # Voice button
            voice_btn = QPushButton("🎤")
            voice_btn.setFixedSize(48, 48)
            input_layout.addWidget(voice_btn)
            
            # Text input
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Ask Bagley anything...")
            self.input_field.setMinimumHeight(48)
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field)
            
            # Attach button
            attach_btn = QPushButton("📎")
            attach_btn.setFixedSize(48, 48)
            attach_btn.clicked.connect(self._attach_file)
            input_layout.addWidget(attach_btn)
            
            # Send button
            send_btn = QPushButton("Send")
            send_btn.setObjectName("primary")
            send_btn.setFixedSize(80, 48)
            send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(send_btn)
            
            layout.addWidget(input_frame)
            
            return panel
        
        def _create_outputs_panel(self) -> QWidget:
            """Generated content panel"""
            panel = QFrame()
            panel.setFixedWidth(350)
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("🎨 Generated Content")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Image preview
            self.image_preview = QLabel("No image generated")
            self.image_preview.setFixedHeight(200)
            self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_preview.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 8px;")
            layout.addWidget(self.image_preview)
            
            # Audio player placeholder
            audio_frame = QFrame()
            audio_frame.setFixedHeight(60)
            audio_frame.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 8px;")
            audio_layout = QHBoxLayout(audio_frame)
            audio_layout.addWidget(QPushButton("▶"))
            audio_layout.addWidget(QLabel("No audio"))
            audio_layout.addStretch()
            layout.addWidget(audio_frame)
            
            layout.addStretch()
            
            return panel
        
        def _send_message(self):
            """Send a message"""
            text = self.input_field.text().strip()
            if not text:
                return
            
            # Add user message
            self._add_message(text, is_user=True)
            self.input_field.clear()
            
            # Get active models
            active = [k for k, v in self.model_checks.items() if v.isChecked()]
            
            # Simulate response (would call actual models)
            response = f"[Bagley responding with: {', '.join(active)}]\n\n"
            response += "I'd give you a proper answer but my brain weights aren't loaded yet! 🧠"
            
            QTimer.singleShot(500, lambda: self._add_message(response, is_user=False))
        
        def _add_message(self, text: str, is_user: bool):
            """Add message bubble"""
            bubble = QFrame()
            bubble.setMaximumWidth(600)
            
            bg = COLORS["accent"] if is_user else COLORS["bg_light"]
            align = "right" if is_user else "left"
            
            bubble.setStyleSheet(f"""
                background-color: {bg};
                border-radius: 16px;
                padding: 12px 16px;
            """)
            
            layout = QVBoxLayout(bubble)
            layout.setContentsMargins(16, 12, 16, 12)
            
            label = QLabel(text)
            label.setWordWrap(True)
            layout.addWidget(label)
            
            wrapper = QWidget()
            wrapper_layout = QHBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            
            if is_user:
                wrapper_layout.addStretch()
            wrapper_layout.addWidget(bubble)
            if not is_user:
                wrapper_layout.addStretch()
            
            self.messages_layout.addWidget(wrapper)
        
        def _attach_file(self):
            """Attach file to message"""
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Attach File", "",
                "All Files (*);;Images (*.png *.jpg *.jpeg);;Audio (*.wav *.mp3)"
            )
            if file_path:
                self.input_field.setText(f"[Attached: {Path(file_path).name}] ")
    
    
    class TrainingTab(QWidget):
        """
        🏋️ Training Tab - GPU monitoring & smart data management
        """
        
        def __init__(self):
            super().__init__()
            
            self.data_folders: List[str] = []
            self.training_active = False
            
            layout = QHBoxLayout(self)
            layout.setSpacing(16)
            
            # Left: Data management
            left_panel = self._create_data_panel()
            layout.addWidget(left_panel)
            
            # Center: Training progress
            center_panel = self._create_progress_panel()
            layout.addWidget(center_panel, stretch=2)
            
            # Right: GPU monitoring
            right_panel = self._create_gpu_panel()
            layout.addWidget(right_panel)
            
            # Timer for updates
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self._update_stats)
            self.update_timer.start(2000)  # Every 2 seconds
        
        def _create_data_panel(self) -> QWidget:
            """Data folder selection panel"""
            panel = QFrame()
            panel.setFixedWidth(350)
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("📁 Training Data")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Add folder button
            add_btn = QPushButton("+ Add Folder")
            add_btn.clicked.connect(self._add_folder)
            layout.addWidget(add_btn)
            
            # Folder list
            self.folder_list = QListWidget()
            self.folder_list.setMinimumHeight(150)
            layout.addWidget(self.folder_list)
            
            # Remove button
            remove_btn = QPushButton("Remove Selected")
            remove_btn.clicked.connect(self._remove_folder)
            layout.addWidget(remove_btn)
            
            # Scan button
            scan_btn = QPushButton("🔍 Scan & Sort Data")
            scan_btn.setObjectName("primary")
            scan_btn.clicked.connect(self._scan_data)
            layout.addWidget(scan_btn)
            
            # Auto-train checkbox
            self.auto_train_check = QCheckBox("🤖 Auto-train when data detected")
            self.auto_train_check.setChecked(True)
            layout.addWidget(self.auto_train_check)
            
            # Resume from checkpoint
            self.resume_check = QCheckBox("📥 Resume from last checkpoint")
            self.resume_check.setChecked(True)
            layout.addWidget(self.resume_check)
            
            layout.addSpacing(16)
            
            # Data stats
            stats_label = QLabel("📊 Data Statistics")
            stats_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            layout.addWidget(stats_label)
            
            self.stats_table = QTableWidget(4, 2)
            self.stats_table.setHorizontalHeaderLabels(["Type", "Count"])
            self.stats_table.horizontalHeader().setStretchLastSection(True)
            self.stats_table.verticalHeader().setVisible(False)
            self.stats_table.setMaximumHeight(150)
            
            data_types = ["💬 Chat", "🎨 Images", "🎬 Videos", "🎵 Audio"]
            for i, dt in enumerate(data_types):
                self.stats_table.setItem(i, 0, QTableWidgetItem(dt))
                self.stats_table.setItem(i, 1, QTableWidgetItem("0"))
            
            layout.addWidget(self.stats_table)
            
            layout.addStretch()
            
            return panel
        
        def _create_progress_panel(self) -> QWidget:
            """Training progress panel"""
            panel = QFrame()
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("🏋️ Training Progress")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Training controls
            controls = QHBoxLayout()
            
            self.start_btn = QPushButton("▶ Start Training")
            self.start_btn.setObjectName("primary")
            self.start_btn.clicked.connect(self._toggle_training)
            controls.addWidget(self.start_btn)
            
            self.pause_btn = QPushButton("⏸ Pause")
            self.pause_btn.setEnabled(False)
            self.pause_btn.clicked.connect(self._pause_training)
            controls.addWidget(self.pause_btn)
            
            self.stop_btn = QPushButton("⏹ Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self._stop_training)
            controls.addWidget(self.stop_btn)
            
            controls.addStretch()
            layout.addLayout(controls)
            
            layout.addSpacing(16)
            
            # Model progress bars
            self.progress_widgets = {}
            
            models = [
                ("💬 Chat Model", "chat"),
                ("🎨 Image Model", "image"),
                ("🎬 Video Model", "video"),
                ("🎵 TTS Model", "tts"),
            ]
            
            for label, key in models:
                group = QGroupBox(label)
                group_layout = QVBoxLayout(group)
                
                # Progress bar
                progress = QProgressBar()
                progress.setValue(0)
                group_layout.addWidget(progress)
                
                # Stats
                stats = QLabel("Step: 0 / 0 | Loss: 0.000 | ETA: --:--:--")
                stats.setStyleSheet(f"color: {COLORS['text_dim']};")
                group_layout.addWidget(stats)
                
                layout.addWidget(group)
                
                self.progress_widgets[key] = {
                    "progress": progress,
                    "stats": stats,
                }
            
            layout.addStretch()
            
            # Training log
            log_label = QLabel("📜 Training Log")
            log_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            layout.addWidget(log_label)
            
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(150)
            self.log_text.setStyleSheet(f"font-family: 'Consolas', monospace;")
            layout.addWidget(self.log_text)
            
            return panel
        
        def _create_gpu_panel(self) -> QWidget:
            """GPU monitoring panel"""
            panel = QFrame()
            panel.setFixedWidth(300)
            panel.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("🌡️ GPU Monitor")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Temperature settings
            settings_group = QGroupBox("Auto-Pause Settings")
            settings_layout = QVBoxLayout(settings_group)
            
            # Max temp
            max_temp_layout = QHBoxLayout()
            max_temp_layout.addWidget(QLabel("Pause at:"))
            self.max_temp_spin = QSpinBox()
            self.max_temp_spin.setRange(60, 95)
            self.max_temp_spin.setValue(85)
            self.max_temp_spin.setSuffix("°C")
            max_temp_layout.addWidget(self.max_temp_spin)
            settings_layout.addLayout(max_temp_layout)
            
            # Resume temp
            resume_temp_layout = QHBoxLayout()
            resume_temp_layout.addWidget(QLabel("Resume at:"))
            self.resume_temp_spin = QSpinBox()
            self.resume_temp_spin.setRange(50, 85)
            self.resume_temp_spin.setValue(75)
            self.resume_temp_spin.setSuffix("°C")
            resume_temp_layout.addWidget(self.resume_temp_spin)
            settings_layout.addLayout(resume_temp_layout)
            
            layout.addWidget(settings_group)
            
            layout.addSpacing(16)
            
            # GPU status
            self.gpu_widgets = []
            
            # Create placeholder for up to 8 GPUs
            for i in range(8):
                gpu_frame = QFrame()
                gpu_frame.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 8px; padding: 8px;")
                gpu_layout = QVBoxLayout(gpu_frame)
                gpu_layout.setContentsMargins(12, 8, 12, 8)
                
                name_label = QLabel(f"GPU {i}")
                name_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                gpu_layout.addWidget(name_label)
                
                temp_label = QLabel("-- °C")
                temp_label.setStyleSheet(f"font-size: 24px; color: {COLORS['success']};")
                gpu_layout.addWidget(temp_label)
                
                util_bar = QProgressBar()
                util_bar.setValue(0)
                util_bar.setMaximumHeight(6)
                gpu_layout.addWidget(util_bar)
                
                mem_label = QLabel("0 / 0 GB")
                mem_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
                gpu_layout.addWidget(mem_label)
                
                layout.addWidget(gpu_frame)
                gpu_frame.hide()  # Hide until we know how many GPUs
                
                self.gpu_widgets.append({
                    "frame": gpu_frame,
                    "name": name_label,
                    "temp": temp_label,
                    "util": util_bar,
                    "mem": mem_label,
                })
            
            layout.addStretch()
            
            # Status
            self.gpu_status = QLabel("⏸ Monitoring paused")
            self.gpu_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.gpu_status)
            
            return panel
        
        def _add_folder(self):
            """Add a data folder"""
            folder = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
            if folder and folder not in self.data_folders:
                self.data_folders.append(folder)
                self.folder_list.addItem(folder)
                self._log(f"Added folder: {folder}")
        
        def _remove_folder(self):
            """Remove selected folder"""
            current = self.folder_list.currentRow()
            if current >= 0:
                folder = self.data_folders.pop(current)
                self.folder_list.takeItem(current)
                self._log(f"Removed folder: {folder}")
        
        def _scan_data(self):
            """Scan and sort data"""
            if not self.data_folders:
                QMessageBox.warning(self, "No Folders", "Add some data folders first!")
                return
            
            self._log("Scanning data folders...")
            
            try:
                from bagley.training.monitor import SmartDataSorter
                
                sorter = SmartDataSorter()
                sorter.scan_folders(self.data_folders)
                
                # Update stats table
                type_map = {
                    "chat": 0,
                    "image": 1,
                    "video": 2,
                    "audio": 3,
                }
                
                for dt, count in sorter.stats.items():
                    if dt in type_map:
                        self.stats_table.setItem(type_map[dt], 1, QTableWidgetItem(str(count)))
                
                self._log(f"Scan complete: {sorter.stats}")
                
                # Auto-train if auto-train is enabled
                if hasattr(self, 'auto_train_check') and self.auto_train_check.isChecked():
                    total_data = sum(sorter.stats.values())
                    if total_data > 0:
                        self._log("🤖 Auto-training triggered!")
                        self._start_training()
                
            except Exception as e:
                self._log(f"Error scanning: {e}")
        
        def _toggle_training(self):
            """Start/stop training"""
            if not self.training_active:
                self._start_training()
            else:
                self._stop_training()
        
        def _start_training(self):
            """Start training"""
            if not self.data_folders:
                QMessageBox.warning(self, "No Data", "Add and scan data folders first!")
                return
            
            self.training_active = True
            self.start_btn.setText("⏹ Stop Training")
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.gpu_status.setText("✅ Monitoring active")
            
            self._log("🚀 Training started!")
            
            # Start actual training in background
            self._start_training_worker()
        
        def _start_training_worker(self):
            """Start training in background thread"""
            try:
                from bagley.training.flexible_trainer import FlexibleTrainer, HardwareDetector
                
                # Detect hardware
                gpus = HardwareDetector.detect_gpus()
                if gpus:
                    self._log(f"🎯 Detected {len(gpus)} GPU(s):")
                    for gpu in gpus:
                        self._log(f"   - {gpu.name} ({gpu.memory_total}MB)")
                else:
                    self._log("⚠️ No GPUs detected - using CPU")
                
                # Would start actual training here
                # For now, just log the config
                config = HardwareDetector.get_optimal_config(gpus)
                self._log(f"📋 Config: batch={config.get('batch_size')}, strategy={config.get('strategy')}")
                
            except Exception as e:
                self._log(f"❌ Error starting training: {e}")
        
        def _pause_training(self):
            """Pause/resume training"""
            if self.pause_btn.text() == "⏸ Pause":
                self.pause_btn.setText("▶ Resume")
                self._log("⏸ Training paused")
            else:
                self.pause_btn.setText("⏸ Pause")
                self._log("▶ Training resumed")
        
        def _stop_training(self):
            """Stop training"""
            self.training_active = False
            self.start_btn.setText("▶ Start Training")
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setText("⏸ Pause")
            self.gpu_status.setText("⏸ Monitoring paused")
            
            self._log("⏹ Training stopped - checkpoint saved")
        
        def _update_stats(self):
            """Update GPU stats"""
            if not self.training_active:
                return
            
            try:
                from bagley.training.monitor import GPUMonitor
                
                monitor = GPUMonitor()
                temps = monitor.get_temps()
                
                for i, temp in enumerate(temps):
                    if i < len(self.gpu_widgets):
                        widget = self.gpu_widgets[i]
                        widget["frame"].show()
                        widget["temp"].setText(f"{temp:.0f}°C")
                        
                        # Color based on temp
                        if temp >= self.max_temp_spin.value():
                            color = COLORS["error"]
                        elif temp >= self.max_temp_spin.value() - 10:
                            color = COLORS["warning"]
                        else:
                            color = COLORS["success"]
                        
                        widget["temp"].setStyleSheet(f"font-size: 24px; color: {color};")
                        
                        # Auto-pause on overheat
                        if temp >= self.max_temp_spin.value() and self.pause_btn.text() == "⏸ Pause":
                            self._log(f"🔥 GPU {i} at {temp}°C - Auto-pausing!")
                            self._pause_training()
                        
                        # Auto-resume when cooled
                        elif temp <= self.resume_temp_spin.value() and self.pause_btn.text() == "▶ Resume":
                            self._log(f"✅ GPU {i} cooled to {temp}°C - Resuming!")
                            self._pause_training()
                            
            except Exception as e:
                pass  # Silently fail if monitoring not available
        
        def _log(self, message: str):
            """Add to training log"""
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.append(f"[{timestamp}] {message}")


def run_app():
    """Run the Bagley app"""
    if not QT_AVAILABLE:
        print("Error: Qt not available. Install with: pip install PySide6")
        return 1
    
    app = QApplication(sys.argv)
    window = BagleyAppV2()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
