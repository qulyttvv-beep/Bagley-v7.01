"""
🎨 Bagley Desktop App - Main Application
Ultra-modern PyQt6/PySide6 interface with smooth animations
"""

import sys
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Try to import Qt framework
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QStackedWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
        QSplitter, QFrame, QGraphicsOpacityEffect, QScrollArea
    )
    from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QThread
    from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
    QT_AVAILABLE = True
    QT_VERSION = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QStackedWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
            QSplitter, QFrame, QGraphicsOpacityEffect, QScrollArea
        )
        from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal as Signal, QThread
        from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
        QT_AVAILABLE = True
        QT_VERSION = "PyQt6"
    except ImportError:
        QT_AVAILABLE = False
        QT_VERSION = None
        logger.warning("No Qt framework found. Install PySide6 or PyQt6 for GUI.")


# Color scheme - Dark futuristic theme
COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_tertiary": "#1a1a2e",
    "accent": "#6366f1",  # Indigo
    "accent_hover": "#818cf8",
    "accent_glow": "rgba(99, 102, 241, 0.3)",
    "text_primary": "#ffffff",
    "text_secondary": "#a1a1aa",
    "border": "#27272a",
    "success": "#22c55e",
    "error": "#ef4444",
    "warning": "#eab308",
}


STYLESHEET = """
QMainWindow {
    background-color: #0a0a0f;
}

QWidget {
    color: #ffffff;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
}

QPushButton {
    background-color: #1a1a2e;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 16px;
    color: #ffffff;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #6366f1;
    border-color: #6366f1;
}

QPushButton:pressed {
    background-color: #4f46e5;
}

QLineEdit {
    background-color: #12121a;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 12px 16px;
    color: #ffffff;
    font-size: 14px;
}

QLineEdit:focus {
    border-color: #6366f1;
    background-color: #1a1a2e;
}

QTextEdit {
    background-color: #12121a;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 8px;
    color: #ffffff;
}

QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollBar:vertical {
    background-color: #12121a;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #27272a;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6366f1;
}

QLabel {
    color: #ffffff;
}

QFrame {
    border: none;
}
"""


class BagleyApp:
    """
    🎨 Main Bagley Desktop Application
    
    Features:
    - Smooth animated transitions
    - Glassmorphism effects
    - Dark futuristic theme
    - Voice visualization
    - Multi-modal panels
    """
    
    def __init__(self):
        if not QT_AVAILABLE:
            raise ImportError("Qt framework not available. Install PySide6 or PyQt6.")
        
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        
        # Set dark palette
        self._setup_palette()
        
        # Create main window
        self.window = MainWindow()
        
        logger.info(f"Initialized BagleyApp with {QT_VERSION}")
    
    def _setup_palette(self):
        """Setup dark color palette"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg_primary"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text_primary"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["bg_secondary"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["bg_tertiary"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text_primary"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["bg_tertiary"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text_primary"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent"]))
        self.app.setPalette(palette)
    
    def run(self) -> int:
        """Run the application"""
        self.window.show()
        return self.app.exec()


if QT_AVAILABLE:
    class MainWindow(QMainWindow):
        """Main application window"""
        
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("🤖 Bagley AI")
            self.setMinimumSize(1200, 800)
            self.setStyleSheet(STYLESHEET)
            
            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            
            # Main layout
            main_layout = QHBoxLayout(central)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # Sidebar
            self.sidebar = Sidebar()
            main_layout.addWidget(self.sidebar)
            
            # Content area
            content_layout = QVBoxLayout()
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            
            # Header
            header = Header()
            content_layout.addWidget(header)
            
            # Main content stack
            self.content_stack = QStackedWidget()
            
            # Chat view
            self.chat_view = ChatView()
            self.content_stack.addWidget(self.chat_view)
            
            # Media view
            self.media_view = MediaView()
            self.content_stack.addWidget(self.media_view)
            
            # Settings view
            self.settings_view = SettingsView()
            self.content_stack.addWidget(self.settings_view)
            
            content_layout.addWidget(self.content_stack)
            
            content_container = QWidget()
            content_container.setLayout(content_layout)
            main_layout.addWidget(content_container, stretch=1)
            
            # Connect sidebar navigation
            self.sidebar.nav_changed.connect(self._on_nav_changed)
        
        def _on_nav_changed(self, index: int):
            """Handle navigation change with animation"""
            self.content_stack.setCurrentIndex(index)
    
    
    class Sidebar(QFrame):
        """Collapsible sidebar navigation"""
        
        nav_changed = Signal(int)
        
        def __init__(self):
            super().__init__()
            
            self.setFixedWidth(72)
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-right: 1px solid {COLORS["border"]};
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 16, 12, 16)
            layout.setSpacing(8)
            
            # Logo
            logo = QLabel("🤖")
            logo.setFont(QFont("Segoe UI Emoji", 24))
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo)
            
            layout.addSpacing(24)
            
            # Nav buttons
            self.nav_buttons = []
            
            icons = ["💬", "🎨", "🎬", "🎵", "⚙️"]
            tooltips = ["Chat", "Images", "Videos", "Audio", "Settings"]
            
            for i, (icon, tooltip) in enumerate(zip(icons, tooltips)):
                btn = NavButton(icon, tooltip)
                btn.clicked.connect(lambda checked, idx=i: self._on_click(idx))
                layout.addWidget(btn)
                self.nav_buttons.append(btn)
            
            layout.addStretch()
        
        def _on_click(self, index: int):
            # Map nav to content views (Chat=0, Media=1, Settings=2)
            if index < 4:
                self.nav_changed.emit(1 if index > 0 else 0)  # Chat or Media
            else:
                self.nav_changed.emit(2)  # Settings
    
    
    class NavButton(QPushButton):
        """Animated navigation button"""
        
        def __init__(self, icon: str, tooltip: str):
            super().__init__(icon)
            
            self.setToolTip(tooltip)
            self.setFixedSize(48, 48)
            self.setFont(QFont("Segoe UI Emoji", 16))
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    border-radius: 12px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["bg_tertiary"]};
                }}
                QPushButton:pressed {{
                    background-color: {COLORS["accent"]};
                }}
            """)
    
    
    class Header(QFrame):
        """Top header bar"""
        
        def __init__(self):
            super().__init__()
            
            self.setFixedHeight(64)
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-bottom: 1px solid {COLORS["border"]};
                }}
            """)
            
            layout = QHBoxLayout(self)
            layout.setContentsMargins(24, 0, 24, 0)
            
            # Title
            title = QLabel("Bagley AI")
            title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            layout.addWidget(title)
            
            layout.addStretch()
            
            # Status indicator
            self.status = QLabel("● Online")
            self.status.setStyleSheet(f"color: {COLORS['success']};")
            layout.addWidget(self.status)
    
    
    class ChatView(QWidget):
        """Main chat interface"""
        
        def __init__(self):
            super().__init__()
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            # Messages area
            self.messages_area = QScrollArea()
            self.messages_area.setWidgetResizable(True)
            self.messages_area.setStyleSheet("background-color: transparent;")
            
            self.messages_container = QWidget()
            self.messages_layout = QVBoxLayout(self.messages_container)
            self.messages_layout.setContentsMargins(24, 24, 24, 24)
            self.messages_layout.setSpacing(16)
            self.messages_layout.addStretch()
            
            self.messages_area.setWidget(self.messages_container)
            layout.addWidget(self.messages_area, stretch=1)
            
            # Input area
            input_container = QFrame()
            input_container.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS["bg_secondary"]};
                    border-top: 1px solid {COLORS["border"]};
                }}
            """)
            
            input_layout = QHBoxLayout(input_container)
            input_layout.setContentsMargins(24, 16, 24, 16)
            input_layout.setSpacing(12)
            
            # Voice button
            voice_btn = QPushButton("🎤")
            voice_btn.setFixedSize(48, 48)
            voice_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["bg_tertiary"]};
                    border-radius: 24px;
                    font-size: 18px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["accent"]};
                }}
            """)
            input_layout.addWidget(voice_btn)
            
            # Text input
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Message Bagley...")
            self.input_field.setMinimumHeight(48)
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field, stretch=1)
            
            # Send button
            send_btn = QPushButton("➤")
            send_btn.setFixedSize(48, 48)
            send_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["accent"]};
                    border-radius: 24px;
                    font-size: 18px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["accent_hover"]};
                }}
            """)
            send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(send_btn)
            
            layout.addWidget(input_container)
        
        def _send_message(self):
            """Send a message"""
            text = self.input_field.text().strip()
            if text:
                self._add_message(text, is_user=True)
                self.input_field.clear()
                
                # Simulate response (would connect to orchestrator)
                QTimer.singleShot(500, lambda: self._add_message(
                    "I'm Bagley! Still getting my brain connected... 🧠",
                    is_user=False
                ))
        
        def _add_message(self, text: str, is_user: bool):
            """Add a message bubble"""
            bubble = MessageBubble(text, is_user)
            
            # Insert before the stretch
            self.messages_layout.insertWidget(
                self.messages_layout.count() - 1,
                bubble
            )
            
            # Scroll to bottom
            QTimer.singleShot(50, lambda: self.messages_area.verticalScrollBar().setValue(
                self.messages_area.verticalScrollBar().maximum()
            ))
    
    
    class MessageBubble(QFrame):
        """Animated message bubble"""
        
        def __init__(self, text: str, is_user: bool):
            super().__init__()
            
            bg_color = COLORS["accent"] if is_user else COLORS["bg_tertiary"]
            align = "right" if is_user else "left"
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {bg_color};
                    border-radius: 16px;
                    padding: 12px 16px;
                    max-width: 70%;
                }}
            """)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 12, 16, 12)
            
            label = QLabel(text)
            label.setWordWrap(True)
            label.setFont(QFont("Segoe UI", 13))
            layout.addWidget(label)
            
            # Fade in animation
            self.opacity_effect = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(self.opacity_effect)
            
            self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
            self.animation.setDuration(200)
            self.animation.setStartValue(0.0)
            self.animation.setEndValue(1.0)
            self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self.animation.start()
    
    
    class MediaView(QWidget):
        """Media gallery view"""
        
        def __init__(self):
            super().__init__()
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 24, 24, 24)
            
            title = QLabel("🎨 Media Gallery")
            title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            layout.addWidget(title)
            
            info = QLabel("Generated images, videos, and audio will appear here.")
            info.setStyleSheet(f"color: {COLORS['text_secondary']};")
            layout.addWidget(info)
            
            layout.addStretch()
    
    
    class SettingsView(QWidget):
        """Settings panel"""
        
        def __init__(self):
            super().__init__()
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 24, 24, 24)
            
            title = QLabel("⚙️ Settings")
            title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Settings categories
            categories = [
                ("🤖 AI Model", "Configure chat model parameters"),
                ("🎨 Image Generation", "FLUX-style image model settings"),
                ("🎬 Video Generation", "Video model configuration"),
                ("🎵 Voice & TTS", "Text-to-speech and voice settings"),
                ("🔊 Audio", "System audio settings"),
                ("🎨 Appearance", "Theme and display options"),
            ]
            
            for icon_title, desc in categories:
                setting_item = QFrame()
                setting_item.setStyleSheet(f"""
                    QFrame {{
                        background-color: {COLORS["bg_tertiary"]};
                        border-radius: 12px;
                        padding: 16px;
                    }}
                    QFrame:hover {{
                        background-color: {COLORS["bg_secondary"]};
                    }}
                """)
                
                item_layout = QHBoxLayout(setting_item)
                
                text_layout = QVBoxLayout()
                title_label = QLabel(icon_title)
                title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
                text_layout.addWidget(title_label)
                
                desc_label = QLabel(desc)
                desc_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
                text_layout.addWidget(desc_label)
                
                item_layout.addLayout(text_layout)
                item_layout.addStretch()
                item_layout.addWidget(QLabel("→"))
                
                layout.addWidget(setting_item)
            
            layout.addStretch()

else:
    # Fallback classes when Qt is not available
    class MainWindow:
        pass
    
    class Sidebar:
        pass
    
    class NavButton:
        pass
    
    class Header:
        pass
    
    class ChatView:
        pass
    
    class MessageBubble:
        pass
    
    class MediaView:
        pass
    
    class SettingsView:
        pass


def main():
    """Entry point for desktop app"""
    if not QT_AVAILABLE:
        print("Error: Qt framework not available.")
        print("Install with: pip install PySide6")
        return 1
    
    app = BagleyApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
