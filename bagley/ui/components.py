"""
🎨 Bagley UI Components
Reusable UI components with animations
"""

from typing import Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


# Try to import Qt framework
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QFrame, QTextEdit, QScrollArea, QGraphicsOpacityEffect,
        QSlider, QComboBox, QCheckBox, QProgressBar
    )
    from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal, QTimer
    from PySide6.QtGui import QFont, QColor, QPainter, QPen
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QFrame, QTextEdit, QScrollArea, QGraphicsOpacityEffect,
            QSlider, QComboBox, QCheckBox, QProgressBar
        )
        from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal as Signal, QTimer
        from PyQt6.QtGui import QFont, QColor, QPainter, QPen
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False


# Colors
COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_tertiary": "#1a1a2e",
    "accent": "#6366f1",
    "accent_hover": "#818cf8",
    "text_primary": "#ffffff",
    "text_secondary": "#a1a1aa",
    "border": "#27272a",
    "success": "#22c55e",
    "error": "#ef4444",
}


if QT_AVAILABLE:
    
    class ChatWindow(QFrame):
        """
        Chat window component with messages display and input
        """
        
        message_sent = Signal(str)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-radius: 16px;
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            # Messages scroll area
            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setStyleSheet("background: transparent; border: none;")
            
            self.messages_widget = QWidget()
            self.messages_layout = QVBoxLayout(self.messages_widget)
            self.messages_layout.setContentsMargins(16, 16, 16, 16)
            self.messages_layout.setSpacing(12)
            self.messages_layout.addStretch()
            
            self.scroll_area.setWidget(self.messages_widget)
            layout.addWidget(self.scroll_area, stretch=1)
            
            # Input area
            input_frame = QFrame()
            input_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_tertiary"]};
                    border-radius: 12px;
                    margin: 16px;
                }}
            """)
            
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(16, 12, 16, 12)
            
            self.input_field = QTextEdit()
            self.input_field.setPlaceholderText("Type a message...")
            self.input_field.setMaximumHeight(100)
            self.input_field.setStyleSheet(f"""
                QTextEdit {{
                    background: transparent;
                    border: none;
                    color: {COLORS["text_primary"]};
                    font-size: 14px;
                }}
            """)
            input_layout.addWidget(self.input_field, stretch=1)
            
            send_btn = QPushButton("Send")
            send_btn.clicked.connect(self._on_send)
            send_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["accent"]};
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    color: white;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["accent_hover"]};
                }}
            """)
            input_layout.addWidget(send_btn)
            
            layout.addWidget(input_frame)
        
        def _on_send(self):
            text = self.input_field.toPlainText().strip()
            if text:
                self.add_message(text, is_user=True)
                self.message_sent.emit(text)
                self.input_field.clear()
        
        def add_message(self, text: str, is_user: bool = False):
            """Add a message to the chat"""
            bubble = ChatBubble(text, is_user)
            self.messages_layout.insertWidget(
                self.messages_layout.count() - 1,
                bubble,
                alignment=Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft
            )
            
            # Scroll to bottom
            QTimer.singleShot(50, self._scroll_to_bottom)
        
        def _scroll_to_bottom(self):
            scrollbar = self.scroll_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    
    class ChatBubble(QFrame):
        """Animated chat message bubble"""
        
        def __init__(self, text: str, is_user: bool):
            super().__init__()
            
            bg = COLORS["accent"] if is_user else COLORS["bg_tertiary"]
            radius = "16px 16px 4px 16px" if is_user else "16px 16px 16px 4px"
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {bg};
                    border-radius: {radius};
                    padding: 12px 16px;
                }}
            """)
            self.setMaximumWidth(500)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 14px;")
            layout.addWidget(label)
            
            # Fade in animation
            self.effect = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(self.effect)
            
            self.anim = QPropertyAnimation(self.effect, b"opacity")
            self.anim.setDuration(200)
            self.anim.setStartValue(0)
            self.anim.setEndValue(1)
            self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self.anim.start()
    
    
    class MediaViewer(QFrame):
        """Media viewer for images, videos, audio"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-radius: 16px;
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            
            # Media display area
            self.display_area = QLabel()
            self.display_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.display_area.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS["bg_tertiary"]};
                    border-radius: 12px;
                    min-height: 300px;
                }}
            """)
            self.display_area.setText("Drop media here or generate new")
            layout.addWidget(self.display_area, stretch=1)
            
            # Controls
            controls = QHBoxLayout()
            
            play_btn = QPushButton("▶")
            play_btn.setFixedSize(40, 40)
            controls.addWidget(play_btn)
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {COLORS["bg_tertiary"]};
                    border-radius: 4px;
                    height: 8px;
                }}
                QProgressBar::chunk {{
                    background-color: {COLORS["accent"]};
                    border-radius: 4px;
                }}
            """)
            controls.addWidget(progress, stretch=1)
            
            layout.addLayout(controls)
    
    
    class Settings(QFrame):
        """Settings panel with categories"""
        
        setting_changed = Signal(str, object)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-radius: 16px;
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            
            # Title
            title = QLabel("Settings")
            title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Scroll area for settings
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background: transparent; border: none;")
            
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(8)
            
            # Add settings categories
            self._add_setting_group(content_layout, "AI Model", [
                ("Temperature", "slider", 0.7, (0, 1)),
                ("Max Tokens", "slider", 2048, (256, 8192)),
                ("System Prompt", "text", "You are Bagley..."),
            ])
            
            self._add_setting_group(content_layout, "Voice", [
                ("Enable TTS", "checkbox", True),
                ("Voice", "dropdown", "Default", ["Default", "Male", "Female"]),
                ("Speed", "slider", 1.0, (0.5, 2.0)),
            ])
            
            self._add_setting_group(content_layout, "Appearance", [
                ("Theme", "dropdown", "Dark", ["Dark", "Light", "System"]),
                ("Animations", "checkbox", True),
            ])
            
            content_layout.addStretch()
            scroll.setWidget(content)
            layout.addWidget(scroll)
        
        def _add_setting_group(self, parent_layout, title: str, settings: list):
            """Add a settings group"""
            group = QFrame()
            group.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_tertiary"]};
                    border-radius: 12px;
                    padding: 16px;
                }}
            """)
            
            layout = QVBoxLayout(group)
            layout.setContentsMargins(16, 12, 16, 12)
            layout.setSpacing(12)
            
            # Group title
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
            layout.addWidget(title_label)
            
            # Settings
            for setting in settings:
                self._add_setting_item(layout, *setting)
            
            parent_layout.addWidget(group)
        
        def _add_setting_item(self, parent_layout, name: str, type_: str, default, options=None):
            """Add a single setting item"""
            item_layout = QHBoxLayout()
            
            label = QLabel(name)
            label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            item_layout.addWidget(label)
            
            if type_ == "slider":
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(int(options[0] * 100))
                slider.setMaximum(int(options[1] * 100))
                slider.setValue(int(default * 100))
                slider.setStyleSheet(f"""
                    QSlider::groove:horizontal {{
                        background: {COLORS["bg_primary"]};
                        height: 6px;
                        border-radius: 3px;
                    }}
                    QSlider::handle:horizontal {{
                        background: {COLORS["accent"]};
                        width: 16px;
                        height: 16px;
                        margin: -5px 0;
                        border-radius: 8px;
                    }}
                """)
                item_layout.addWidget(slider)
                
            elif type_ == "dropdown":
                combo = QComboBox()
                combo.addItems(options)
                combo.setCurrentText(str(default))
                combo.setStyleSheet(f"""
                    QComboBox {{
                        background: {COLORS["bg_primary"]};
                        border: 1px solid {COLORS["border"]};
                        border-radius: 6px;
                        padding: 6px 12px;
                        min-width: 100px;
                    }}
                """)
                item_layout.addWidget(combo)
                
            elif type_ == "checkbox":
                check = QCheckBox()
                check.setChecked(default)
                item_layout.addWidget(check)
            
            parent_layout.addLayout(item_layout)
    
    
    class Sidebar(QFrame):
        """Navigation sidebar"""
        
        nav_changed = Signal(int)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setFixedWidth(64)
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-right: 1px solid {COLORS["border"]};
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 16, 8, 16)
            layout.setSpacing(8)
            
            # Logo
            logo = QLabel("🤖")
            logo.setFont(QFont("Segoe UI Emoji", 20))
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo)
            
            layout.addSpacing(16)
            
            # Nav items
            icons = ["💬", "🎨", "⚙️"]
            for i, icon in enumerate(icons):
                btn = QPushButton(icon)
                btn.setFixedSize(48, 48)
                btn.setFont(QFont("Segoe UI Emoji", 16))
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        border: none;
                        border-radius: 12px;
                    }}
                    QPushButton:hover {{
                        background: {COLORS["bg_tertiary"]};
                    }}
                """)
                btn.clicked.connect(lambda checked, idx=i: self.nav_changed.emit(idx))
                layout.addWidget(btn)
            
            layout.addStretch()
    
    
    class VoiceVisualizer(QWidget):
        """Audio waveform visualizer"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setMinimumHeight(60)
            self.levels = [0.0] * 32
            
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_levels)
        
        def start(self):
            self.timer.start(50)
        
        def stop(self):
            self.timer.stop()
            self.levels = [0.0] * 32
            self.update()
        
        def _update_levels(self):
            import random
            self.levels = [random.uniform(0.2, 1.0) for _ in range(32)]
            self.update()
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            pen = QPen(QColor(COLORS["accent"]))
            pen.setWidth(3)
            painter.setPen(pen)
            
            width = self.width()
            height = self.height()
            bar_width = width / len(self.levels)
            
            for i, level in enumerate(self.levels):
                x = i * bar_width + bar_width / 4
                bar_height = level * (height - 10)
                y = (height - bar_height) / 2
                
                painter.drawLine(int(x), int(y), int(x), int(y + bar_height))

else:
    # Placeholder classes when Qt is not available
    class ChatWindow:
        pass
    
    class ChatBubble:
        pass
    
    class MediaViewer:
        pass
    
    class Settings:
        pass
    
    class Sidebar:
        pass
    
    class VoiceVisualizer:
        pass


__all__ = [
    "ChatWindow",
    "ChatBubble", 
    "MediaViewer",
    "Settings",
    "Sidebar",
    "VoiceVisualizer",
]
