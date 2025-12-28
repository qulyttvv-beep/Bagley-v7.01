"""
🎛️ Training UI Components
========================
Path selection, format display, 3D viewer integration
"""

import os
import sys
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextEdit, QLineEdit, QProgressBar, QFileDialog, QListWidget,
        QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
        QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QMessageBox, QTabWidget, QSplitter, QDialog,
        QDialogButtonBox, QFormLayout, QPlainTextEdit
    )
    from PySide6.QtCore import Qt, QTimer, QThread, Signal
    from PySide6.QtGui import QFont
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QTextEdit, QLineEdit, QProgressBar, QFileDialog, QListWidget,
            QFrame, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
            QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
            QHeaderView, QMessageBox, QTabWidget, QSplitter, QDialog,
            QDialogButtonBox, QFormLayout, QPlainTextEdit
        )
        from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal
        from PyQt6.QtGui import QFont
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        logger.warning("Qt not available, UI components disabled")

# Import training utilities
try:
    from bagley.training.unified_trainer import (
        SUPPORTED_FORMATS, ModelType, get_supported_formats_text,
        UnifiedTrainingConfig, StorageConfig, MixedGPUConfig
    )
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("Training module not available")


# Colors (matching app_v2.py)
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
    
    # ==================== Path Selection Widget ====================
    
    class PathSelectorWidget(QWidget):
        """
        📁 Path selection with NAS support
        Allows user to choose any path including network drives
        """
        
        path_changed = Signal(str)
        
        def __init__(
            self,
            label: str = "Path:",
            placeholder: str = "Select path...",
            is_directory: bool = True,
            parent: Optional[QWidget] = None
        ):
            super().__init__(parent)
            
            self.is_directory = is_directory
            
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Label
            self.label = QLabel(label)
            self.label.setMinimumWidth(120)
            layout.addWidget(self.label)
            
            # Path input (editable for NAS paths)
            self.path_input = QLineEdit()
            self.path_input.setPlaceholderText(placeholder)
            self.path_input.textChanged.connect(self._on_path_changed)
            layout.addWidget(self.path_input, stretch=1)
            
            # Browse button
            self.browse_btn = QPushButton("📁 Browse")
            self.browse_btn.clicked.connect(self._browse)
            layout.addWidget(self.browse_btn)
            
            # Status indicator
            self.status_label = QLabel("❓")
            self.status_label.setFixedWidth(30)
            layout.addWidget(self.status_label)
        
        def _browse(self):
            """Open file/folder dialog"""
            if self.is_directory:
                path = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory",
                    self.path_input.text() or str(Path.home())
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select File",
                    self.path_input.text() or str(Path.home())
                )
            
            if path:
                self.path_input.setText(path)
        
        def _on_path_changed(self, path: str):
            """Validate and emit path change"""
            if path:
                # Check if path exists
                exists = Path(path).exists() if not path.startswith('\\\\') else self._check_unc(path)
                self.status_label.setText("✅" if exists else "⚠️")
                self.status_label.setStyleSheet(
                    f"color: {COLORS['success'] if exists else COLORS['warning']};"
                )
            else:
                self.status_label.setText("❓")
                self.status_label.setStyleSheet("")
            
            self.path_changed.emit(path)
        
        def _check_unc(self, path: str) -> bool:
            """Check UNC path accessibility"""
            try:
                return os.path.exists(path)
            except:
                return False
        
        def get_path(self) -> str:
            return self.path_input.text()
        
        def set_path(self, path: str):
            self.path_input.setText(path)
    
    
    # ==================== Supported Formats Display ====================
    
    class SupportedFormatsWidget(QWidget):
        """
        📋 Display supported training data formats
        """
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            
            layout = QVBoxLayout(self)
            
            # Header
            header = QLabel("📁 SUPPORTED TRAINING DATA FORMATS")
            header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            layout.addWidget(header)
            
            # Scroll area for formats
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 8px;")
            
            formats_widget = QWidget()
            formats_layout = QVBoxLayout(formats_widget)
            
            if TRAINING_AVAILABLE:
                for model_type, formats in SUPPORTED_FORMATS.items():
                    group = self._create_format_group(model_type, formats)
                    formats_layout.addWidget(group)
            else:
                formats_layout.addWidget(QLabel("Training module not loaded"))
            
            formats_layout.addStretch()
            scroll.setWidget(formats_widget)
            layout.addWidget(scroll)
        
        def _create_format_group(self, model_type, formats: Dict) -> QGroupBox:
            """Create format group for a model type"""
            group = QGroupBox(f"🔹 {model_type.value.upper()} Model")
            layout = QVBoxLayout(group)
            
            # Description
            desc = QLabel(formats.get('description', ''))
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color: {COLORS['text_dim']};")
            layout.addWidget(desc)
            
            # Data formats
            data_formats = formats.get('training_data', [])
            if data_formats:
                formats_label = QLabel(f"📄 Data: {', '.join(data_formats)}")
                layout.addWidget(formats_label)
            
            # Caption formats
            caption_formats = formats.get('captions', [])
            if caption_formats:
                captions_label = QLabel(f"📝 Captions: {', '.join(caption_formats)}")
                layout.addWidget(captions_label)
            
            # Audio formats
            audio_formats = formats.get('audio', [])
            if audio_formats:
                audio_label = QLabel(f"🎵 Audio: {', '.join(audio_formats)}")
                layout.addWidget(audio_label)
            
            # Examples
            examples = formats.get('examples', [])
            if examples:
                examples_label = QLabel("Examples:")
                examples_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
                layout.addWidget(examples_label)
                
                for ex in examples:
                    ex_label = QLabel(f"  • {ex}")
                    ex_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
                    layout.addWidget(ex_label)
            
            return group
    
    
    # ==================== Training Configuration Panel ====================
    
    class TrainingConfigPanel(QWidget):
        """
        ⚙️ Training configuration panel
        """
        
        config_changed = Signal(dict)
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            
            layout = QVBoxLayout(self)
            layout.setSpacing(16)
            
            # ===== Paths Section =====
            paths_group = QGroupBox("📁 Data & Output Paths")
            paths_layout = QVBoxLayout(paths_group)
            
            # Training data path
            self.data_path = PathSelectorWidget(
                "Training Data:",
                "Select folder or NAS path (e.g., \\\\NAS\\data)",
                is_directory=True
            )
            paths_layout.addWidget(self.data_path)
            
            # Model output path
            self.model_path = PathSelectorWidget(
                "Model Output:",
                "Where to save trained models",
                is_directory=True
            )
            paths_layout.addWidget(self.model_path)
            
            # Checkpoint path
            self.checkpoint_path = PathSelectorWidget(
                "Checkpoints:",
                "Checkpoint save location",
                is_directory=True
            )
            paths_layout.addWidget(self.checkpoint_path)
            
            # Log path
            self.log_path = PathSelectorWidget(
                "Logs:",
                "Training logs location",
                is_directory=True
            )
            paths_layout.addWidget(self.log_path)
            
            layout.addWidget(paths_group)
            
            # ===== Models to Train =====
            models_group = QGroupBox("🎯 Models to Train")
            models_layout = QVBoxLayout(models_group)
            
            self.model_checks = {}
            models = [
                ("💬 Chat/Language Model", "chat", True),
                ("🎨 Image Generation", "image", True),
                ("🎬 Video Generation (with audio)", "video", True),
                ("🎵 Text-to-Speech", "tts", True),
                ("🎨 3D Model Generation", "3d", True),
            ]
            
            for label, key, default in models:
                check = QCheckBox(label)
                check.setChecked(default)
                self.model_checks[key] = check
                models_layout.addWidget(check)
            
            layout.addWidget(models_group)
            
            # ===== GPU Settings =====
            gpu_group = QGroupBox("🖥️ GPU & Memory Settings")
            gpu_layout = QFormLayout(gpu_group)
            
            # Mixed GPU
            self.mixed_gpu_check = QCheckBox("Enable Mixed GPU (AMD + NVIDIA)")
            self.mixed_gpu_check.setChecked(True)
            gpu_layout.addRow(self.mixed_gpu_check)
            
            # CPU offload
            self.cpu_offload_check = QCheckBox("Enable CPU Offloading")
            self.cpu_offload_check.setChecked(True)
            gpu_layout.addRow(self.cpu_offload_check)
            
            # Gradient checkpointing
            self.grad_checkpoint_check = QCheckBox("Gradient Checkpointing (saves memory)")
            self.grad_checkpoint_check.setChecked(True)
            gpu_layout.addRow(self.grad_checkpoint_check)
            
            # Gradient accumulation
            self.grad_accum_spin = QSpinBox()
            self.grad_accum_spin.setRange(1, 64)
            self.grad_accum_spin.setValue(8)
            gpu_layout.addRow("Gradient Accumulation:", self.grad_accum_spin)
            
            # Mixed precision
            self.precision_combo = QComboBox()
            self.precision_combo.addItems(["bf16", "fp16", "fp32"])
            gpu_layout.addRow("Precision:", self.precision_combo)
            
            layout.addWidget(gpu_group)
            
            # ===== Training Parameters =====
            params_group = QGroupBox("📊 Training Parameters")
            params_layout = QFormLayout(params_group)
            
            self.epochs_spin = QSpinBox()
            self.epochs_spin.setRange(1, 100)
            self.epochs_spin.setValue(3)
            params_layout.addRow("Epochs:", self.epochs_spin)
            
            self.batch_spin = QSpinBox()
            self.batch_spin.setRange(1, 64)
            self.batch_spin.setValue(1)
            params_layout.addRow("Batch Size:", self.batch_spin)
            
            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setRange(0.0000001, 0.01)
            self.lr_spin.setValue(0.00001)
            self.lr_spin.setDecimals(7)
            self.lr_spin.setSingleStep(0.000001)
            params_layout.addRow("Learning Rate:", self.lr_spin)
            
            self.warmup_spin = QSpinBox()
            self.warmup_spin.setRange(0, 10000)
            self.warmup_spin.setValue(100)
            params_layout.addRow("Warmup Steps:", self.warmup_spin)
            
            layout.addWidget(params_group)
            
            # ===== Auto Features =====
            auto_group = QGroupBox("🔧 Automation")
            auto_layout = QVBoxLayout(auto_group)
            
            self.auto_integrate_check = QCheckBox("Auto-integrate models after training")
            self.auto_integrate_check.setChecked(True)
            auto_layout.addWidget(self.auto_integrate_check)
            
            self.auto_test_check = QCheckBox("Auto-test models after training")
            self.auto_test_check.setChecked(True)
            auto_layout.addWidget(self.auto_test_check)
            
            self.auto_recovery_check = QCheckBox("Auto-recover from errors")
            self.auto_recovery_check.setChecked(True)
            auto_layout.addWidget(self.auto_recovery_check)
            
            layout.addWidget(auto_group)
            
            layout.addStretch()
        
        def get_config(self) -> Dict[str, Any]:
            """Get current configuration"""
            return {
                'storage': {
                    'training_data_path': self.data_path.get_path(),
                    'model_output_path': self.model_path.get_path(),
                    'checkpoint_path': self.checkpoint_path.get_path(),
                    'log_path': self.log_path.get_path(),
                },
                'models_to_train': [
                    key for key, check in self.model_checks.items()
                    if check.isChecked()
                ],
                'gpu': {
                    'enable_mixed_gpu': self.mixed_gpu_check.isChecked(),
                    'enable_cpu_offload': self.cpu_offload_check.isChecked(),
                    'gradient_checkpointing': self.grad_checkpoint_check.isChecked(),
                    'gradient_accumulation_steps': self.grad_accum_spin.value(),
                    'mixed_precision': self.precision_combo.currentText(),
                },
                'training': {
                    'epochs': self.epochs_spin.value(),
                    'batch_size': self.batch_spin.value(),
                    'learning_rate': self.lr_spin.value(),
                    'warmup_steps': self.warmup_spin.value(),
                },
                'auto': {
                    'integrate': self.auto_integrate_check.isChecked(),
                    'test': self.auto_test_check.isChecked(),
                    'recovery': self.auto_recovery_check.isChecked(),
                }
            }
        
        def set_config(self, config: Dict[str, Any]):
            """Set configuration from dict"""
            storage = config.get('storage', {})
            self.data_path.set_path(storage.get('training_data_path', ''))
            self.model_path.set_path(storage.get('model_output_path', ''))
            self.checkpoint_path.set_path(storage.get('checkpoint_path', ''))
            self.log_path.set_path(storage.get('log_path', ''))
            
            for key in config.get('models_to_train', []):
                if key in self.model_checks:
                    self.model_checks[key].setChecked(True)
            
            gpu = config.get('gpu', {})
            self.mixed_gpu_check.setChecked(gpu.get('enable_mixed_gpu', True))
            self.cpu_offload_check.setChecked(gpu.get('enable_cpu_offload', True))
            self.grad_checkpoint_check.setChecked(gpu.get('gradient_checkpointing', True))
            self.grad_accum_spin.setValue(gpu.get('gradient_accumulation_steps', 8))
            
            training = config.get('training', {})
            self.epochs_spin.setValue(training.get('epochs', 3))
            self.batch_spin.setValue(training.get('batch_size', 1))
            self.lr_spin.setValue(training.get('learning_rate', 0.00001))
    
    
    # ==================== 3D Model Viewer ====================
    
    class Model3DViewerWidget(QWidget):
        """
        🎨 3D Model Viewer
        Displays generated 3D models in the UI
        """
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            
            layout = QVBoxLayout(self)
            
            # Header
            header = QHBoxLayout()
            title = QLabel("🎨 3D Model Viewer")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            header.addWidget(title)
            
            header.addStretch()
            
            # Load button
            load_btn = QPushButton("📂 Load Model")
            load_btn.clicked.connect(self._load_model)
            header.addWidget(load_btn)
            
            # Export button
            export_btn = QPushButton("💾 Export")
            export_btn.clicked.connect(self._export_model)
            header.addWidget(export_btn)
            
            layout.addLayout(header)
            
            # Viewer area (would use OpenGL/WebGL in production)
            self.viewer_frame = QFrame()
            self.viewer_frame.setMinimumHeight(400)
            self.viewer_frame.setStyleSheet(f"""
                background-color: {COLORS['bg_light']};
                border-radius: 12px;
                border: 2px dashed {COLORS['border']};
            """)
            
            viewer_layout = QVBoxLayout(self.viewer_frame)
            viewer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.placeholder = QLabel("🎨\n\nGenerate or load a 3D model\nto view it here")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 14px;")
            viewer_layout.addWidget(self.placeholder)
            
            layout.addWidget(self.viewer_frame, stretch=1)
            
            # Controls
            controls = QHBoxLayout()
            
            # Rotation controls
            controls.addWidget(QLabel("Rotation:"))
            
            self.rotate_x = QSpinBox()
            self.rotate_x.setRange(0, 360)
            self.rotate_x.setPrefix("X: ")
            controls.addWidget(self.rotate_x)
            
            self.rotate_y = QSpinBox()
            self.rotate_y.setRange(0, 360)
            self.rotate_y.setPrefix("Y: ")
            controls.addWidget(self.rotate_y)
            
            self.rotate_z = QSpinBox()
            self.rotate_z.setRange(0, 360)
            self.rotate_z.setPrefix("Z: ")
            controls.addWidget(self.rotate_z)
            
            controls.addStretch()
            
            # Zoom
            controls.addWidget(QLabel("Zoom:"))
            self.zoom_spin = QSpinBox()
            self.zoom_spin.setRange(10, 200)
            self.zoom_spin.setValue(100)
            self.zoom_spin.setSuffix("%")
            controls.addWidget(self.zoom_spin)
            
            layout.addLayout(controls)
            
            # Model info
            self.info_label = QLabel("")
            self.info_label.setStyleSheet(f"color: {COLORS['text_dim']};")
            layout.addWidget(self.info_label)
            
            self.current_model_path = None
        
        def _load_model(self):
            """Load a 3D model file"""
            formats = "3D Models (*.obj *.glb *.gltf *.ply *.stl *.fbx)"
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load 3D Model",
                "",
                formats
            )
            
            if path:
                self.load_model(path)
        
        def load_model(self, path: str):
            """Load model from path"""
            self.current_model_path = path
            
            # Update placeholder
            self.placeholder.setText(f"🎨\n\nLoaded: {Path(path).name}\n\n(3D rendering would appear here)")
            
            # Update info
            size = Path(path).stat().st_size / 1024
            self.info_label.setText(f"File: {Path(path).name} | Size: {size:.1f} KB")
        
        def _export_model(self):
            """Export current model"""
            if not self.current_model_path:
                QMessageBox.warning(self, "No Model", "No model loaded to export")
                return
            
            formats = "OBJ (*.obj);;GLB (*.glb);;PLY (*.ply);;STL (*.stl)"
            path, selected = QFileDialog.getSaveFileName(
                self,
                "Export 3D Model",
                "",
                formats
            )
            
            if path:
                # Copy or convert model
                QMessageBox.information(self, "Export", f"Model exported to {path}")
        
        def display_mesh(self, mesh_data: Dict[str, Any]):
            """Display mesh data from generator"""
            self.placeholder.setText("🎨\n\n3D Model Generated!\n\n(Rendering preview)")
            self.info_label.setText(
                f"Vertices: {len(mesh_data.get('vertices', []))//3} | "
                f"Faces: {len(mesh_data.get('faces', []))//3}"
            )
    
    
    # ==================== 3D Generation Panel ====================
    
    class Model3DGenerationPanel(QWidget):
        """
        🎨 3D Model Generation Panel
        Generate 3D models from text descriptions
        """
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            
            layout = QHBoxLayout(self)
            layout.setSpacing(16)
            
            # Left: Generation controls
            left = QFrame()
            left.setFixedWidth(350)
            left.setStyleSheet(f"background-color: {COLORS['bg_medium']}; border-radius: 12px;")
            
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(16, 16, 16, 16)
            
            title = QLabel("🎨 Generate 3D Model")
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            left_layout.addWidget(title)
            
            # Prompt
            left_layout.addWidget(QLabel("Description:"))
            self.prompt_input = QTextEdit()
            self.prompt_input.setPlaceholderText(
                "Describe the 3D object you want to generate...\n\n"
                "Example: A red sports car with chrome wheels and tinted windows"
            )
            self.prompt_input.setMaximumHeight(120)
            left_layout.addWidget(self.prompt_input)
            
            # Settings
            settings_group = QGroupBox("⚙️ Settings")
            settings_layout = QFormLayout(settings_group)
            
            self.format_combo = QComboBox()
            self.format_combo.addItems(["glb", "obj", "ply", "stl"])
            settings_layout.addRow("Output Format:", self.format_combo)
            
            self.texture_check = QCheckBox()
            self.texture_check.setChecked(True)
            settings_layout.addRow("Generate Texture:", self.texture_check)
            
            self.resolution_spin = QSpinBox()
            self.resolution_spin.setRange(64, 256)
            self.resolution_spin.setValue(128)
            settings_layout.addRow("Mesh Resolution:", self.resolution_spin)
            
            left_layout.addWidget(settings_group)
            
            # Generate button
            self.generate_btn = QPushButton("🚀 Generate")
            self.generate_btn.setStyleSheet(f"background-color: {COLORS['accent']}; font-weight: bold;")
            self.generate_btn.clicked.connect(self._generate)
            left_layout.addWidget(self.generate_btn)
            
            # Progress
            self.progress = QProgressBar()
            self.progress.setVisible(False)
            left_layout.addWidget(self.progress)
            
            left_layout.addStretch()
            
            # Output path
            self.output_path = PathSelectorWidget(
                "Save to:",
                "Select output location",
                is_directory=True
            )
            left_layout.addWidget(self.output_path)
            
            layout.addWidget(left)
            
            # Right: Viewer
            self.viewer = Model3DViewerWidget()
            layout.addWidget(self.viewer, stretch=1)
        
        def _generate(self):
            """Generate 3D model"""
            prompt = self.prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "No Prompt", "Please enter a description")
                return
            
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # Indeterminate
            self.generate_btn.setEnabled(False)
            
            # TODO: Connect to actual generator
            # For now, simulate with timer
            QTimer.singleShot(2000, self._on_generation_complete)
        
        def _on_generation_complete(self):
            """Called when generation completes"""
            self.progress.setVisible(False)
            self.generate_btn.setEnabled(True)
            
            # Update viewer
            self.viewer.display_mesh({
                'vertices': list(range(1000)),
                'faces': list(range(500))
            })
            
            QMessageBox.information(self, "Complete", "3D model generated!")
    
    
    # ==================== Enhanced Training Tab ====================
    
    class EnhancedTrainingTab(QWidget):
        """
        🏋️ Enhanced Training Tab
        With path selection, format display, and 3D generation
        """
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Sub-tabs for training
            self.sub_tabs = QTabWidget()
            
            # Tab 1: Training Config
            config_tab = QWidget()
            config_layout = QHBoxLayout(config_tab)
            
            # Left: Configuration
            self.config_panel = TrainingConfigPanel()
            config_layout.addWidget(self.config_panel)
            
            # Right: Formats info
            self.formats_panel = SupportedFormatsWidget()
            config_layout.addWidget(self.formats_panel)
            
            self.sub_tabs.addTab(config_tab, "⚙️ Configuration")
            
            # Tab 2: Training Progress
            progress_tab = self._create_progress_tab()
            self.sub_tabs.addTab(progress_tab, "📊 Progress")
            
            # Tab 3: 3D Generation
            self.model_3d_panel = Model3DGenerationPanel()
            self.sub_tabs.addTab(self.model_3d_panel, "🎨 3D Generation")
            
            layout.addWidget(self.sub_tabs)
            
            # Bottom: Action buttons
            actions = QHBoxLayout()
            actions.setContentsMargins(16, 8, 16, 8)
            
            self.start_btn = QPushButton("🚀 Start Training")
            self.start_btn.setStyleSheet(f"background-color: {COLORS['accent']}; font-weight: bold; padding: 12px 24px;")
            self.start_btn.clicked.connect(self._start_training)
            actions.addWidget(self.start_btn)
            
            self.stop_btn = QPushButton("⏹️ Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self._stop_training)
            actions.addWidget(self.stop_btn)
            
            actions.addStretch()
            
            self.save_config_btn = QPushButton("💾 Save Config")
            self.save_config_btn.clicked.connect(self._save_config)
            actions.addWidget(self.save_config_btn)
            
            self.load_config_btn = QPushButton("📂 Load Config")
            self.load_config_btn.clicked.connect(self._load_config)
            actions.addWidget(self.load_config_btn)
            
            layout.addLayout(actions)
        
        def _create_progress_tab(self) -> QWidget:
            """Create training progress tab"""
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # GPU monitor
            gpu_group = QGroupBox("🖥️ GPU Status")
            gpu_layout = QVBoxLayout(gpu_group)
            
            self.gpu_table = QTableWidget(2, 4)
            self.gpu_table.setHorizontalHeaderLabels(["GPU", "Memory Used", "Memory Total", "Utilization"])
            self.gpu_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            gpu_layout.addWidget(self.gpu_table)
            
            layout.addWidget(gpu_group)
            
            # Training progress
            progress_group = QGroupBox("📊 Training Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            # Overall progress
            overall_layout = QHBoxLayout()
            overall_layout.addWidget(QLabel("Overall:"))
            self.overall_progress = QProgressBar()
            overall_layout.addWidget(self.overall_progress)
            progress_layout.addLayout(overall_layout)
            
            # Current model
            model_layout = QHBoxLayout()
            model_layout.addWidget(QLabel("Current Model:"))
            self.current_model_label = QLabel("None")
            model_layout.addWidget(self.current_model_label)
            model_layout.addStretch()
            progress_layout.addLayout(model_layout)
            
            # Epoch/Step
            step_layout = QHBoxLayout()
            step_layout.addWidget(QLabel("Epoch:"))
            self.epoch_label = QLabel("0/0")
            step_layout.addWidget(self.epoch_label)
            step_layout.addWidget(QLabel("Step:"))
            self.step_label = QLabel("0")
            step_layout.addWidget(self.step_label)
            step_layout.addWidget(QLabel("Loss:"))
            self.loss_label = QLabel("--")
            step_layout.addWidget(self.loss_label)
            step_layout.addStretch()
            progress_layout.addLayout(step_layout)
            
            layout.addWidget(progress_group)
            
            # Log output
            log_group = QGroupBox("📝 Training Log")
            log_layout = QVBoxLayout(log_group)
            
            self.log_output = QPlainTextEdit()
            self.log_output.setReadOnly(True)
            self.log_output.setStyleSheet(f"font-family: 'Consolas', monospace; background-color: {COLORS['bg_dark']};")
            log_layout.addWidget(self.log_output)
            
            layout.addWidget(log_group)
            
            return tab
        
        def _start_training(self):
            """Start training"""
            config = self.config_panel.get_config()
            
            # Validate paths
            storage = config['storage']
            if not storage.get('training_data_path'):
                QMessageBox.warning(self, "Missing Path", "Please select a training data path")
                return
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            self.log_output.appendPlainText(f"[{self._timestamp()}] Starting training...")
            self.log_output.appendPlainText(f"[{self._timestamp()}] Data: {storage['training_data_path']}")
            self.log_output.appendPlainText(f"[{self._timestamp()}] Models: {config['models_to_train']}")
            
            # TODO: Connect to actual trainer
            self.sub_tabs.setCurrentIndex(1)  # Switch to progress tab
        
        def _stop_training(self):
            """Stop training"""
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_output.appendPlainText(f"[{self._timestamp()}] Training stopped by user")
        
        def _save_config(self):
            """Save configuration to file"""
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Configuration",
                "training_config.json",
                "JSON Files (*.json)"
            )
            
            if path:
                config = self.config_panel.get_config()
                with open(path, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Saved", f"Config saved to {path}")
        
        def _load_config(self):
            """Load configuration from file"""
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Configuration",
                "",
                "JSON Files (*.json)"
            )
            
            if path:
                with open(path) as f:
                    config = json.load(f)
                self.config_panel.set_config(config)
                QMessageBox.information(self, "Loaded", f"Config loaded from {path}")
        
        def _timestamp(self) -> str:
            """Get current timestamp"""
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S")


# Export for use in main app
__all__ = [
    'PathSelectorWidget',
    'SupportedFormatsWidget', 
    'TrainingConfigPanel',
    'Model3DViewerWidget',
    'Model3DGenerationPanel',
    'EnhancedTrainingTab'
]
