"""
🖥️ Bagley Desktop App v3 - Modern UI
=====================================
Features:
- 💬 Chat Tab - Talk to all models
- 🏋️ Training Tab - Train with GPU monitoring
- 🌐 API Tab - Host worldwide API
- 📊 Always-on system monitoring
"""

import sys
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
        QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
        QFrame, QSplitter, QScrollArea, QGraphicsOpacityEffect,
        QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
        QStatusBar, QToolBar, QMenu, QSystemTrayIcon, QStyle,
        QGraphicsDropShadowEffect, QSizePolicy, QFormLayout,
        QPlainTextEdit, QToolButton
    )
    from PySide6.QtCore import (
        Qt, QTimer, QThread, Signal, QPropertyAnimation, 
        QEasingCurve, QSize, QPoint, QParallelAnimationGroup
    )
    from PySide6.QtGui import (
        QFont, QColor, QPalette, QIcon, QLinearGradient,
        QPainter, QBrush, QPen, QAction
    )
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
            QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
            QFrame, QSplitter, QScrollArea, QGraphicsOpacityEffect,
            QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
            QStatusBar, QToolBar, QMenu, QSystemTrayIcon, QStyle,
            QGraphicsDropShadowEffect, QSizePolicy, QFormLayout,
            QPlainTextEdit, QToolButton
        )
        from PyQt6.QtCore import (
            Qt, QTimer, QThread, pyqtSignal as Signal, QPropertyAnimation,
            QEasingCurve, QSize, QPoint, QParallelAnimationGroup
        )
        from PyQt6.QtGui import (
            QFont, QColor, QPalette, QIcon, QLinearGradient,
            QPainter, QBrush, QPen, QAction
        )
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        logger.error("Qt not available")


# ==================== Modern Color Scheme ====================

COLORS = {
    # Backgrounds
    "bg_dark": "#0d0d12",
    "bg_medium": "#14141c",
    "bg_light": "#1c1c28",
    "bg_card": "#202030",
    "bg_hover": "#252535",
    
    # Accent colors
    "accent": "#7c3aed",
    "accent_hover": "#8b5cf6",
    "accent_glow": "#a78bfa",
    "accent_secondary": "#06b6d4",
    
    # Status colors
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "info": "#3b82f6",
    
    # Text
    "text": "#f4f4f5",
    "text_secondary": "#a1a1aa",
    "text_dim": "#71717a",
    
    # Borders
    "border": "#27272a",
    "border_light": "#3f3f46",
    
    # Gradients
    "gradient_start": "#7c3aed",
    "gradient_end": "#06b6d4",
}

MODERN_STYLESHEET = f"""
/* ===== Base ===== */
QMainWindow, QWidget {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text"]};
    font-family: 'Segoe UI Variable', 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}

/* ===== Tab Widget ===== */
QTabWidget::pane {{
    border: none;
    background-color: transparent;
    margin-top: -1px;
}}

QTabBar {{
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: {COLORS["bg_medium"]};
    color: {COLORS["text_secondary"]};
    padding: 14px 28px;
    margin-right: 4px;
    border: none;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 {COLORS["accent"]}, stop:1 {COLORS["accent_secondary"]});
    color: {COLORS["text"]};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS["bg_hover"]};
    color: {COLORS["text"]};
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 12px 24px;
    color: {COLORS["text"]};
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS["bg_hover"]};
    border-color: {COLORS["accent"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["bg_light"]};
}}

QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent"]}, stop:1 {COLORS["accent_secondary"]});
    border: none;
    color: white;
    font-weight: 600;
}}

QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent_hover"]}, stop:1 {COLORS["accent_secondary"]});
}}

QPushButton#success {{
    background-color: {COLORS["success"]};
    border: none;
    color: white;
}}

QPushButton#danger {{
    background-color: {COLORS["error"]};
    border: none;
    color: white;
}}

/* ===== Input Fields ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 12px 16px;
    color: {COLORS["text"]};
    selection-background-color: {COLORS["accent"]};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS["accent"]};
    background-color: {COLORS["bg_light"]};
}}

QLineEdit::placeholder {{
    color: {COLORS["text_dim"]};
}}

/* ===== Progress Bars ===== */
QProgressBar {{
    background-color: {COLORS["bg_medium"]};
    border-radius: 6px;
    height: 12px;
    text-align: center;
    font-size: 10px;
    color: {COLORS["text_dim"]};
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent"]}, stop:1 {COLORS["accent_secondary"]});
    border-radius: 6px;
}}

/* ===== Scroll Bars ===== */
QScrollBar:vertical {{
    background-color: {COLORS["bg_dark"]};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS["border_light"]};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS["accent"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {COLORS["bg_dark"]};
    height: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS["border_light"]};
    border-radius: 4px;
    min-width: 30px;
}}

/* ===== Group Boxes ===== */
QGroupBox {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    margin-top: 16px;
    padding: 20px 16px 16px 16px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    top: 4px;
    padding: 0 8px;
    color: {COLORS["text"]};
}}

/* ===== Combo Box ===== */
QComboBox {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 10px 16px;
    color: {COLORS["text"]};
    min-width: 100px;
}}

QComboBox:hover {{
    border-color: {COLORS["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    selection-background-color: {COLORS["accent"]};
}}

/* ===== Spin Boxes ===== */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS["bg_medium"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 10px 16px;
    color: {COLORS["text"]};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS["accent"]};
}}

/* ===== Check Boxes ===== */
QCheckBox {{
    spacing: 10px;
    color: {COLORS["text"]};
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 2px solid {COLORS["border_light"]};
    background-color: {COLORS["bg_medium"]};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS["accent"]};
}}

/* ===== Tables ===== */
QTableWidget {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    gridline-color: {COLORS["border"]};
}}

QTableWidget::item {{
    padding: 10px;
    border-bottom: 1px solid {COLORS["border"]};
}}

QTableWidget::item:selected {{
    background-color: {COLORS["accent"]};
}}

QHeaderView::section {{
    background-color: {COLORS["bg_light"]};
    padding: 12px;
    border: none;
    font-weight: 600;
    color: {COLORS["text_secondary"]};
}}

/* ===== List Widget ===== */
QListWidget {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 8px;
}}

QListWidget::item {{
    padding: 12px;
    border-radius: 8px;
    margin: 2px;
}}

QListWidget::item:selected {{
    background-color: {COLORS["accent"]};
}}

QListWidget::item:hover:!selected {{
    background-color: {COLORS["bg_hover"]};
}}

/* ===== Status Bar ===== */
QStatusBar {{
    background-color: {COLORS["bg_medium"]};
    border-top: 1px solid {COLORS["border"]};
    padding: 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* ===== Tool Tips ===== */
QToolTip {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 8px 12px;
    color: {COLORS["text"]};
}}

/* ===== Menu ===== */
QMenu {{
    background-color: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 8px;
}}

QMenu::item {{
    padding: 10px 24px;
    border-radius: 6px;
}}

QMenu::item:selected {{
    background-color: {COLORS["accent"]};
}}
"""


# ==================== Card Widget ====================

if QT_AVAILABLE:
    
    class GlowCard(QFrame):
        """Card with glow effect on hover"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet(f"""
                GlowCard {{
                    background-color: {COLORS["bg_card"]};
                    border: 1px solid {COLORS["border"]};
                    border-radius: 16px;
                }}
                GlowCard:hover {{
                    border-color: {COLORS["accent"]};
                }}
            """)
            
            # Add shadow
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setXOffset(0)
            shadow.setYOffset(4)
            shadow.setColor(QColor(0, 0, 0, 60))
            self.setGraphicsEffect(shadow)


    class StatCard(GlowCard):
        """Card displaying a statistic"""
        
        def __init__(self, icon: str, title: str, value: str = "--", parent=None):
            super().__init__(parent)
            self.setFixedHeight(100)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 16, 20, 16)
            
            # Header
            header = QHBoxLayout()
            
            icon_label = QLabel(icon)
            icon_label.setFont(QFont("Segoe UI Emoji", 20))
            header.addWidget(icon_label)
            
            title_label = QLabel(title)
            title_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            header.addWidget(title_label)
            header.addStretch()
            
            layout.addLayout(header)
            
            # Value
            self.value_label = QLabel(value)
            self.value_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
            self.value_label.setStyleSheet(f"color: {COLORS['text']};")
            layout.addWidget(self.value_label)
        
        def set_value(self, value: str, color: str = None):
            self.value_label.setText(value)
            if color:
                self.value_label.setStyleSheet(f"color: {color};")


    # ==================== Modern System Monitor Widget ====================
    
    class SystemMonitorWidget(QWidget):
        """
        📊 Always-visible system monitor
        Shows GPU/CPU temps in real-time
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumWidth(320)
            
            layout = QVBoxLayout(self)
            layout.setSpacing(12)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Title
            title_layout = QHBoxLayout()
            title = QLabel("📊 System Monitor")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            title_layout.addWidget(title)
            
            # Refresh indicator
            self.refresh_indicator = QLabel("●")
            self.refresh_indicator.setStyleSheet(f"color: {COLORS['success']};")
            title_layout.addWidget(self.refresh_indicator)
            title_layout.addStretch()
            
            layout.addLayout(title_layout)
            
            # GPU cards
            self.gpu_cards = []
            self.gpu_container = QVBoxLayout()
            layout.addLayout(self.gpu_container)
            
            # CPU card
            self.cpu_card = self._create_cpu_card()
            layout.addWidget(self.cpu_card)
            
            # RAM card
            self.ram_card = self._create_ram_card()
            layout.addWidget(self.ram_card)
            
            layout.addStretch()
            
            # Initialize monitor
            self._init_monitor()
        
        def _create_gpu_card(self, index: int, name: str = "GPU") -> GlowCard:
            """Create a GPU monitoring card"""
            card = GlowCard()
            card.setMinimumHeight(120)
            
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 12, 16, 12)
            
            # Header
            header = QHBoxLayout()
            header.addWidget(QLabel(f"🎮 GPU {index}"))
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            header.addWidget(name_label)
            header.addStretch()
            layout.addLayout(header)
            
            # Temperature
            temp_layout = QHBoxLayout()
            temp_label = QLabel("🌡️")
            temp_layout.addWidget(temp_label)
            
            temp_value = QLabel("--°C")
            temp_value.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            temp_value.setStyleSheet(f"color: {COLORS['success']};")
            temp_layout.addWidget(temp_value)
            temp_layout.addStretch()
            
            # Utilization
            util_label = QLabel("0%")
            util_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            temp_layout.addWidget(util_label)
            
            layout.addLayout(temp_layout)
            
            # Memory bar
            mem_bar = QProgressBar()
            mem_bar.setValue(0)
            mem_bar.setTextVisible(False)
            mem_bar.setMaximumHeight(8)
            layout.addWidget(mem_bar)
            
            mem_label = QLabel("0 / 0 GB")
            mem_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            layout.addWidget(mem_label)
            
            # Store references
            card.temp_value = temp_value
            card.util_label = util_label
            card.mem_bar = mem_bar
            card.mem_label = mem_label
            card.name_label = name_label
            
            return card
        
        def _create_cpu_card(self) -> GlowCard:
            """Create CPU monitoring card"""
            card = GlowCard()
            
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 12, 16, 12)
            
            # Header
            header = QHBoxLayout()
            header.addWidget(QLabel("🖥️ CPU"))
            
            self.cpu_name = QLabel("...")
            self.cpu_name.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            header.addWidget(self.cpu_name)
            header.addStretch()
            layout.addLayout(header)
            
            # Stats row
            stats = QHBoxLayout()
            
            # Temperature
            temp_col = QVBoxLayout()
            temp_col.addWidget(QLabel("Temperature"))
            self.cpu_temp = QLabel("--°C")
            self.cpu_temp.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            self.cpu_temp.setStyleSheet(f"color: {COLORS['success']};")
            temp_col.addWidget(self.cpu_temp)
            stats.addLayout(temp_col)
            
            stats.addStretch()
            
            # Utilization
            util_col = QVBoxLayout()
            util_col.addWidget(QLabel("Usage"))
            self.cpu_util = QLabel("--%")
            self.cpu_util.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            util_col.addWidget(self.cpu_util)
            stats.addLayout(util_col)
            
            stats.addStretch()
            
            # Frequency
            freq_col = QVBoxLayout()
            freq_col.addWidget(QLabel("Frequency"))
            self.cpu_freq = QLabel("-- GHz")
            self.cpu_freq.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            freq_col.addWidget(self.cpu_freq)
            stats.addLayout(freq_col)
            
            layout.addLayout(stats)
            
            return card
        
        def _create_ram_card(self) -> GlowCard:
            """Create RAM monitoring card"""
            card = GlowCard()
            
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 12, 16, 12)
            
            header = QHBoxLayout()
            header.addWidget(QLabel("💾 Memory"))
            header.addStretch()
            
            self.ram_label = QLabel("0 / 0 GB")
            self.ram_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            header.addWidget(self.ram_label)
            
            layout.addLayout(header)
            
            self.ram_bar = QProgressBar()
            self.ram_bar.setValue(0)
            layout.addWidget(self.ram_bar)
            
            return card
        
        def _init_monitor(self):
            """Initialize hardware monitor"""
            try:
                from bagley.core.hardware_monitor import BackgroundMonitor
                
                self.monitor = BackgroundMonitor(
                    update_interval=2.0,
                    on_update=self._on_stats_update
                )
                self.monitor.start()
                
            except ImportError:
                logger.warning("Hardware monitor not available")
                self.monitor = None
        
        def _on_stats_update(self, stats):
            """Called when new stats are available"""
            # Blink indicator
            self.refresh_indicator.setStyleSheet(f"color: {COLORS['accent']};")
            QTimer.singleShot(200, lambda: self.refresh_indicator.setStyleSheet(f"color: {COLORS['success']};"))
            
            # Update GPU cards
            for i, gpu in enumerate(stats.gpus):
                if i >= len(self.gpu_cards):
                    # Create new card
                    card = self._create_gpu_card(i, gpu.name)
                    self.gpu_cards.append(card)
                    self.gpu_container.addWidget(card)
                
                card = self.gpu_cards[i]
                
                # Temperature color
                temp = gpu.temperature
                if temp >= 85:
                    color = COLORS['error']
                elif temp >= 75:
                    color = COLORS['warning']
                else:
                    color = COLORS['success']
                
                card.temp_value.setText(f"{temp:.0f}°C")
                card.temp_value.setStyleSheet(f"color: {color}; font-size: 24px;")
                card.util_label.setText(f"{gpu.utilization:.0f}%")
                
                # Memory
                if gpu.memory_total > 0:
                    mem_pct = (gpu.memory_used / gpu.memory_total) * 100
                    card.mem_bar.setValue(int(mem_pct))
                    card.mem_label.setText(f"{gpu.memory_used:.1f} / {gpu.memory_total:.1f} GB")
                
                card.name_label.setText(gpu.name[:30])
            
            # Update CPU
            cpu = stats.cpu
            self.cpu_name.setText(cpu.name[:40])
            
            # CPU temp color
            temp = cpu.temperature
            if temp > 0:
                if temp >= 90:
                    color = COLORS['error']
                elif temp >= 80:
                    color = COLORS['warning']
                else:
                    color = COLORS['success']
                self.cpu_temp.setText(f"{temp:.0f}°C")
                self.cpu_temp.setStyleSheet(f"color: {color}; font-size: 18px;")
            else:
                self.cpu_temp.setText("N/A")
            
            self.cpu_util.setText(f"{cpu.utilization:.0f}%")
            self.cpu_freq.setText(f"{cpu.frequency:.2f} GHz")
            
            # Update RAM
            if stats.ram_total_gb > 0:
                pct = (stats.ram_used_gb / stats.ram_total_gb) * 100
                self.ram_bar.setValue(int(pct))
                self.ram_label.setText(f"{stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB ({pct:.0f}%)")
        
        def stop(self):
            """Stop monitoring"""
            if self.monitor:
                self.monitor.stop()


    # ==================== API Tab ====================
    
    class APITab(QWidget):
        """
        🌐 API Hosting Tab
        Host Bagley API locally or worldwide
        """
        
        def __init__(self):
            super().__init__()
            self.server = None
            
            layout = QHBoxLayout(self)
            layout.setSpacing(20)
            
            # Left: Server controls
            left = self._create_controls_panel()
            layout.addWidget(left)
            
            # Center: Logs
            center = self._create_logs_panel()
            layout.addWidget(center, stretch=2)
            
            # Right: Stats & Endpoints
            right = self._create_info_panel()
            layout.addWidget(right)
        
        def _create_controls_panel(self) -> QWidget:
            """Server control panel"""
            panel = GlowCard()
            panel.setFixedWidth(350)
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)
            
            title = QLabel("🌐 API Server")
            title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # Status indicator
            status_frame = QFrame()
            status_frame.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 10px; padding: 12px;")
            status_layout = QHBoxLayout(status_frame)
            
            self.status_dot = QLabel("●")
            self.status_dot.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 16px;")
            status_layout.addWidget(self.status_dot)
            
            self.status_text = QLabel("Stopped")
            self.status_text.setFont(QFont("Segoe UI", 14))
            status_layout.addWidget(self.status_text)
            status_layout.addStretch()
            
            layout.addWidget(status_frame)
            
            # Server settings
            settings_group = QGroupBox("⚙️ Server Settings")
            settings_layout = QFormLayout(settings_group)
            
            # Port
            self.port_spin = QSpinBox()
            self.port_spin.setRange(1000, 65535)
            self.port_spin.setValue(8000)
            settings_layout.addRow("Port:", self.port_spin)
            
            # Host
            self.host_combo = QComboBox()
            self.host_combo.addItems(["0.0.0.0 (All)", "127.0.0.1 (Local)", "Custom"])
            settings_layout.addRow("Host:", self.host_combo)
            
            layout.addWidget(settings_group)
            
            # Worldwide access
            tunnel_group = QGroupBox("🌍 Worldwide Access")
            tunnel_layout = QVBoxLayout(tunnel_group)
            
            self.tunnel_combo = QComboBox()
            self.tunnel_combo.addItems([
                "None (Local Only)",
                "Ngrok (Recommended)",
                "Cloudflare Tunnel",
                "localhost.run (Free)",
                "Serveo (Free)"
            ])
            tunnel_layout.addWidget(self.tunnel_combo)
            
            # Ngrok token
            self.ngrok_token = QLineEdit()
            self.ngrok_token.setPlaceholderText("Ngrok auth token (optional)")
            self.ngrok_token.setEchoMode(QLineEdit.EchoMode.Password)
            tunnel_layout.addWidget(self.ngrok_token)
            
            layout.addWidget(tunnel_group)
            
            # Authentication
            auth_group = QGroupBox("🔐 Authentication")
            auth_layout = QVBoxLayout(auth_group)
            
            self.auth_check = QCheckBox("Enable API Key Authentication")
            self.auth_check.setChecked(True)
            auth_layout.addWidget(self.auth_check)
            
            self.api_key_field = QLineEdit()
            self.api_key_field.setPlaceholderText("API Key (auto-generated)")
            self.api_key_field.setReadOnly(True)
            auth_layout.addWidget(self.api_key_field)
            
            copy_btn = QPushButton("📋 Copy Key")
            copy_btn.clicked.connect(self._copy_api_key)
            auth_layout.addWidget(copy_btn)
            
            layout.addWidget(auth_group)
            
            layout.addStretch()
            
            # Start/Stop buttons
            btn_layout = QHBoxLayout()
            
            self.start_btn = QPushButton("▶ Start Server")
            self.start_btn.setObjectName("success")
            self.start_btn.clicked.connect(self._toggle_server)
            btn_layout.addWidget(self.start_btn)
            
            self.tunnel_btn = QPushButton("🌍 Start Tunnel")
            self.tunnel_btn.setEnabled(False)
            self.tunnel_btn.clicked.connect(self._start_tunnel)
            btn_layout.addWidget(self.tunnel_btn)
            
            layout.addLayout(btn_layout)
            
            return panel
        
        def _create_logs_panel(self) -> QWidget:
            """API logs panel"""
            panel = GlowCard()
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(20, 20, 20, 20)
            
            header = QHBoxLayout()
            title = QLabel("📜 API Logs")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            header.addWidget(title)
            header.addStretch()
            
            clear_btn = QPushButton("🗑️ Clear")
            clear_btn.clicked.connect(lambda: self.log_output.clear())
            header.addWidget(clear_btn)
            
            layout.addLayout(header)
            
            self.log_output = QPlainTextEdit()
            self.log_output.setReadOnly(True)
            self.log_output.setStyleSheet(f"""
                QPlainTextEdit {{
                    font-family: 'Cascadia Code', 'Consolas', monospace;
                    font-size: 12px;
                    background-color: {COLORS['bg_dark']};
                    border-radius: 8px;
                }}
            """)
            layout.addWidget(self.log_output)
            
            return panel
        
        def _create_info_panel(self) -> QWidget:
            """API info and endpoints panel"""
            panel = GlowCard()
            panel.setFixedWidth(320)
            
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)
            
            title = QLabel("📡 Endpoints")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            layout.addWidget(title)
            
            # URLs
            url_group = QGroupBox("🔗 Access URLs")
            url_layout = QVBoxLayout(url_group)
            
            url_layout.addWidget(QLabel("Local:"))
            self.local_url = QLineEdit()
            self.local_url.setReadOnly(True)
            self.local_url.setText("http://localhost:8000")
            url_layout.addWidget(self.local_url)
            
            url_layout.addWidget(QLabel("Public:"))
            self.public_url = QLineEdit()
            self.public_url.setReadOnly(True)
            self.public_url.setPlaceholderText("Start tunnel for public URL")
            url_layout.addWidget(self.public_url)
            
            layout.addWidget(url_group)
            
            # Endpoints list
            endpoints_group = QGroupBox("Available Endpoints")
            endpoints_layout = QVBoxLayout(endpoints_group)
            
            endpoints = [
                ("POST /chat", "Send chat messages"),
                ("POST /image", "Generate images"),
                ("POST /video", "Generate videos"),
                ("POST /tts", "Text to speech"),
                ("POST /3d", "Generate 3D models"),
                ("GET /health", "Health check"),
                ("GET /metrics", "API metrics"),
            ]
            
            for endpoint, desc in endpoints:
                row = QHBoxLayout()
                ep_label = QLabel(endpoint)
                ep_label.setStyleSheet(f"color: {COLORS['accent']}; font-family: monospace;")
                row.addWidget(ep_label)
                row.addStretch()
                desc_label = QLabel(desc)
                desc_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
                row.addWidget(desc_label)
                endpoints_layout.addLayout(row)
            
            layout.addWidget(endpoints_group)
            
            # Stats
            stats_group = QGroupBox("📊 Statistics")
            stats_layout = QFormLayout(stats_group)
            
            self.requests_label = QLabel("0")
            stats_layout.addRow("Total Requests:", self.requests_label)
            
            self.success_label = QLabel("0")
            stats_layout.addRow("Successful:", self.success_label)
            
            self.latency_label = QLabel("-- ms")
            stats_layout.addRow("Avg Latency:", self.latency_label)
            
            layout.addWidget(stats_group)
            
            layout.addStretch()
            
            return panel
        
        def _toggle_server(self):
            """Start or stop the server"""
            if self.server is None or not self.server.is_running:
                self._start_server()
            else:
                self._stop_server()
        
        def _start_server(self):
            """Start the API server"""
            try:
                from bagley.api.server import BagleyAPIServer, APIConfig, TunnelProvider
                
                # Get tunnel provider
                tunnel_map = {
                    0: TunnelProvider.NONE,
                    1: TunnelProvider.NGROK,
                    2: TunnelProvider.CLOUDFLARE,
                    3: TunnelProvider.LOCALHOST_RUN,
                    4: TunnelProvider.SERVEO,
                }
                
                config = APIConfig(
                    port=self.port_spin.value(),
                    enable_auth=self.auth_check.isChecked(),
                    tunnel_provider=tunnel_map.get(self.tunnel_combo.currentIndex(), TunnelProvider.NONE),
                    ngrok_token=self.ngrok_token.text()
                )
                
                self.server = BagleyAPIServer(config)
                self.server.start(background=True)
                
                # Update UI
                self.status_dot.setStyleSheet(f"color: {COLORS['success']}; font-size: 16px;")
                self.status_text.setText("Running")
                self.start_btn.setText("⏹ Stop Server")
                self.start_btn.setObjectName("danger")
                self.tunnel_btn.setEnabled(True)
                
                self.local_url.setText(f"http://localhost:{config.port}")
                
                if config.enable_auth:
                    self.api_key_field.setText(config.api_key)
                
                self._log(f"✅ Server started on port {config.port}")
                
                # Start stats update timer
                self.stats_timer = QTimer()
                self.stats_timer.timeout.connect(self._update_stats)
                self.stats_timer.start(5000)
                
            except Exception as e:
                self._log(f"❌ Failed to start server: {e}")
                QMessageBox.critical(self, "Error", f"Failed to start server:\n{e}")
        
        def _stop_server(self):
            """Stop the API server"""
            if self.server:
                self.server.stop()
                self.server = None
            
            self.status_dot.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 16px;")
            self.status_text.setText("Stopped")
            self.start_btn.setText("▶ Start Server")
            self.start_btn.setObjectName("success")
            self.tunnel_btn.setEnabled(False)
            
            if hasattr(self, 'stats_timer'):
                self.stats_timer.stop()
            
            self._log("⏹ Server stopped")
        
        def _start_tunnel(self):
            """Start worldwide tunnel"""
            if self.server:
                self._log("🌍 Starting tunnel...")
                url = self.server.start_tunnel()
                if url:
                    self.public_url.setText(url)
                    self._log(f"✅ Tunnel active: {url}")
                else:
                    self._log("❌ Failed to start tunnel")
        
        def _copy_api_key(self):
            """Copy API key to clipboard"""
            key = self.api_key_field.text()
            if key:
                QApplication.clipboard().setText(key)
                self._log("📋 API key copied to clipboard")
        
        def _update_stats(self):
            """Update API statistics"""
            if self.server:
                metrics = self.server.logger.get_metrics()
                self.requests_label.setText(str(metrics.get('total_requests', 0)))
                self.success_label.setText(str(metrics.get('successful', 0)))
                self.latency_label.setText(f"{metrics.get('avg_latency_ms', 0):.1f} ms")
        
        def _log(self, message: str):
            """Add log message"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_output.appendPlainText(f"[{timestamp}] {message}")


    # ==================== Main Application ====================
    
    class BagleyAppV3(QMainWindow):
        """
        🤖 Bagley v7 - Modern Desktop App
        """
        
        def __init__(self):
            super().__init__()
            
            self.setWindowTitle("🤖 Bagley v7.01")
            self.setMinimumSize(1600, 1000)
            self.setStyleSheet(MODERN_STYLESHEET)
            
            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            
            main_layout = QHBoxLayout(central)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # Left: System monitor (always visible)
            self.system_monitor = SystemMonitorWidget()
            self.system_monitor.setFixedWidth(340)
            self.system_monitor.setStyleSheet(f"""
                background-color: {COLORS['bg_medium']};
                border-right: 1px solid {COLORS['border']};
            """)
            
            # Add padding
            monitor_container = QWidget()
            monitor_layout = QVBoxLayout(monitor_container)
            monitor_layout.setContentsMargins(16, 16, 16, 16)
            monitor_layout.addWidget(self.system_monitor)
            monitor_container.setStyleSheet(f"background-color: {COLORS['bg_medium']};")
            
            main_layout.addWidget(monitor_container)
            
            # Right: Main content area
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(20, 16, 20, 16)
            content_layout.setSpacing(16)
            
            # Header
            header = self._create_header()
            content_layout.addWidget(header)
            
            # Tabs
            self.tabs = QTabWidget()
            
            # Chat tab (placeholder)
            chat_placeholder = QLabel("💬 Chat Tab - Coming from app_v2.py")
            chat_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(chat_placeholder, "💬 Chat")
            
            # Training tab (placeholder)
            training_placeholder = QLabel("🏋️ Training Tab - Coming from app_v2.py")
            training_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(training_placeholder, "🏋️ Training")
            
            # API tab
            self.api_tab = APITab()
            self.tabs.addTab(self.api_tab, "🌐 API")
            
            # 3D Generation tab (placeholder)
            gen_3d_placeholder = QLabel("🎨 3D Generation - Coming from training_components.py")
            gen_3d_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(gen_3d_placeholder, "🎨 3D Gen")
            
            content_layout.addWidget(self.tabs)
            
            main_layout.addWidget(content, stretch=1)
            
            # Status bar
            self.statusBar().showMessage("Ready")
        
        def _create_header(self) -> QWidget:
            """Create modern header"""
            header = QFrame()
            header.setFixedHeight(70)
            header.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {COLORS['bg_card']}, stop:1 {COLORS['bg_medium']});
                    border-radius: 16px;
                }}
            """)
            
            layout = QHBoxLayout(header)
            layout.setContentsMargins(24, 0, 24, 0)
            
            # Logo
            logo = QLabel("🤖 BAGLEY")
            logo.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
            logo.setStyleSheet(f"""
                color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_secondary']});
            """)
            layout.addWidget(logo)
            
            version = QLabel("v7.01")
            version.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px; margin-left: 8px;")
            layout.addWidget(version)
            
            layout.addStretch()
            
            # Quick actions
            settings_btn = QPushButton("⚙️")
            settings_btn.setFixedSize(44, 44)
            settings_btn.setToolTip("Settings")
            layout.addWidget(settings_btn)
            
            return header
        
        def closeEvent(self, event):
            """Cleanup on close"""
            self.system_monitor.stop()
            if hasattr(self, 'api_tab') and self.api_tab.server:
                self.api_tab.server.stop()
            event.accept()


def run_app_v3():
    """Run the modern Bagley app"""
    if not QT_AVAILABLE:
        print("Error: Qt not available. Install with: pip install PySide6")
        return 1
    
    app = QApplication(sys.argv)
    window = BagleyAppV3()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app_v3())
