"""
🖥️ Hardware Monitor
===================
Always-on GPU and CPU temperature monitoring
Works with NVIDIA, AMD, and Intel
"""

import os
import sys
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU statistics"""
    index: int
    name: str
    temperature: float  # Celsius
    utilization: float  # Percentage
    memory_used: float  # GB
    memory_total: float  # GB
    fan_speed: Optional[float] = None  # Percentage
    power_draw: Optional[float] = None  # Watts
    clock_speed: Optional[float] = None  # MHz


@dataclass
class CPUStats:
    """CPU statistics"""
    name: str
    temperature: float  # Celsius (average across cores)
    utilization: float  # Percentage
    core_temps: List[float] = field(default_factory=list)
    frequency: Optional[float] = None  # MHz


@dataclass
class RAMStats:
    """RAM statistics"""
    total: float  # GB
    used: float  # GB
    available: float  # GB
    percent: float


@dataclass
class SystemStats:
    """Complete system statistics"""
    timestamp: datetime
    gpus: List[GPUStats]
    cpu: CPUStats
    ram: RAMStats


class HardwareMonitor:
    """
    🖥️ Always-on hardware monitoring
    Supports NVIDIA (nvml), AMD (rocm-smi), and generic fallbacks
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SystemStats], None]] = []
        self._last_stats: Optional[SystemStats] = None
        
        # Initialize hardware interfaces
        self._nvml_available = self._init_nvml()
        self._psutil_available = self._init_psutil()
        self._wmi_available = self._init_wmi()
    
    def _init_nvml(self) -> bool:
        """Initialize NVIDIA Management Library"""
        try:
            import pynvml
            pynvml.nvmlInit()
            logger.info("NVML initialized successfully")
            return True
        except Exception as e:
            logger.debug(f"NVML not available: {e}")
            return False
    
    def _init_psutil(self) -> bool:
        """Initialize psutil for CPU/RAM"""
        try:
            import psutil
            logger.info("psutil available")
            return True
        except ImportError:
            logger.warning("psutil not installed - pip install psutil")
            return False
    
    def _init_wmi(self) -> bool:
        """Initialize WMI for Windows temps"""
        if sys.platform != 'win32':
            return False
        try:
            import wmi
            logger.info("WMI available for Windows monitoring")
            return True
        except ImportError:
            logger.debug("WMI not available")
            return False
    
    def start(self):
        """Start monitoring"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Hardware monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Hardware monitoring stopped")
    
    def add_callback(self, callback: Callable[[SystemStats], None]):
        """Add callback for stats updates"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SystemStats], None]):
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_stats(self) -> Optional[SystemStats]:
        """Get current stats (non-blocking)"""
        if self._last_stats:
            return self._last_stats
        return self._collect_stats()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                stats = self._collect_stats()
                self._last_stats = stats
                
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.update_interval)
    
    def _collect_stats(self) -> SystemStats:
        """Collect all hardware stats"""
        return SystemStats(
            timestamp=datetime.now(),
            gpus=self._get_gpu_stats(),
            cpu=self._get_cpu_stats(),
            ram=self._get_ram_stats()
        )
    
    def _get_gpu_stats(self) -> List[GPUStats]:
        """Get GPU statistics"""
        gpus = []
        
        # Try NVIDIA first
        if self._nvml_available:
            gpus.extend(self._get_nvidia_stats())
        
        # Try AMD
        amd_gpus = self._get_amd_stats()
        gpus.extend(amd_gpus)
        
        # Fallback to torch if available
        if not gpus:
            gpus = self._get_torch_gpu_stats()
        
        return gpus
    
    def _get_nvidia_stats(self) -> List[GPUStats]:
        """Get NVIDIA GPU stats via NVML"""
        gpus = []
        try:
            import pynvml
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Optional metrics
                fan = None
                power = None
                clock = None
                
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                except:
                    pass
                
                try:
                    clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except:
                    pass
                
                gpus.append(GPUStats(
                    index=i,
                    name=name,
                    temperature=float(temp),
                    utilization=float(util.gpu),
                    memory_used=mem.used / (1024**3),
                    memory_total=mem.total / (1024**3),
                    fan_speed=float(fan) if fan else None,
                    power_draw=float(power) if power else None,
                    clock_speed=float(clock) if clock else None
                ))
                
        except Exception as e:
            logger.debug(f"NVIDIA stats error: {e}")
        
        return gpus
    
    def _get_amd_stats(self) -> List[GPUStats]:
        """Get AMD GPU stats via rocm-smi"""
        gpus = []
        try:
            import subprocess
            
            # Try rocm-smi
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--showuse', '--showmeminfo', 'vram', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                for gpu_id, gpu_data in data.items():
                    if gpu_id.startswith('card'):
                        idx = int(gpu_id.replace('card', ''))
                        
                        gpus.append(GPUStats(
                            index=idx,
                            name=f"AMD GPU {idx}",
                            temperature=float(gpu_data.get('Temperature (Sensor edge) (C)', 0)),
                            utilization=float(gpu_data.get('GPU use (%)', 0)),
                            memory_used=float(gpu_data.get('VRAM Total Used Memory (B)', 0)) / (1024**3),
                            memory_total=float(gpu_data.get('VRAM Total Memory (B)', 0)) / (1024**3),
                        ))
                        
        except FileNotFoundError:
            pass  # rocm-smi not installed
        except Exception as e:
            logger.debug(f"AMD stats error: {e}")
        
        return gpus
    
    def _get_torch_gpu_stats(self) -> List[GPUStats]:
        """Fallback: get basic GPU stats via torch"""
        gpus = []
        try:
            import torch
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem_allocated = torch.cuda.memory_allocated(i)
                    mem_total = props.total_memory
                    
                    gpus.append(GPUStats(
                        index=i,
                        name=props.name,
                        temperature=0.0,  # Not available via torch
                        utilization=0.0,  # Not available via torch
                        memory_used=mem_allocated / (1024**3),
                        memory_total=mem_total / (1024**3),
                    ))
                    
        except Exception as e:
            logger.debug(f"Torch GPU stats error: {e}")
        
        return gpus
    
    def _get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics"""
        name = "Unknown CPU"
        temp = 0.0
        util = 0.0
        core_temps = []
        freq = None
        
        if self._psutil_available:
            import psutil
            
            # CPU utilization
            util = psutil.cpu_percent(interval=None)
            
            # CPU frequency
            try:
                freq_info = psutil.cpu_freq()
                if freq_info:
                    freq = freq_info.current
            except:
                pass
            
            # CPU temperature
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try common sensor names
                    for sensor_name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                        if sensor_name in temps:
                            core_temps = [t.current for t in temps[sensor_name]]
                            temp = sum(core_temps) / len(core_temps) if core_temps else 0.0
                            break
            except:
                pass
        
        # Try WMI on Windows for CPU name and temp
        if self._wmi_available and temp == 0.0:
            try:
                import wmi
                w = wmi.WMI()
                
                # CPU name
                for proc in w.Win32_Processor():
                    name = proc.Name
                    break
                
                # Temperature via WMI (may require admin)
                try:
                    w_thermal = wmi.WMI(namespace="root\\wmi")
                    for temp_sensor in w_thermal.MSAcpi_ThermalZoneTemperature():
                        # Convert from tenths of Kelvin to Celsius
                        temp = (temp_sensor.CurrentTemperature / 10.0) - 273.15
                        break
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"WMI CPU error: {e}")
        
        # Try reading from sysfs on Linux
        if temp == 0.0 and sys.platform == 'linux':
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
            except:
                pass
        
        # Get CPU name via platform if not found
        if name == "Unknown CPU":
            try:
                import platform
                name = platform.processor() or "Unknown CPU"
            except:
                pass
        
        return CPUStats(
            name=name,
            temperature=temp,
            utilization=util,
            core_temps=core_temps,
            frequency=freq
        )
    
    def _get_ram_stats(self) -> RAMStats:
        """Get RAM statistics"""
        if self._psutil_available:
            import psutil
            mem = psutil.virtual_memory()
            return RAMStats(
                total=mem.total / (1024**3),
                used=mem.used / (1024**3),
                available=mem.available / (1024**3),
                percent=mem.percent
            )
        
        return RAMStats(total=0, used=0, available=0, percent=0)


# ==================== Smart Logging ====================

class SmartLogger:
    """
    📝 Smart logging system with rotation and filtering
    """
    
    def __init__(self, log_dir: str = "logs", max_files: int = 10, max_size_mb: int = 10):
        self.log_dir = log_dir
        self.max_files = max_files
        self.max_size_mb = max_size_mb
        self.current_log = None
        self.log_file = None
        
        os.makedirs(log_dir, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup rotating file handler"""
        from pathlib import Path
        
        # Find next log number
        existing = list(Path(self.log_dir).glob("bagley_*.log"))
        if existing:
            numbers = []
            for f in existing:
                try:
                    num = int(f.stem.split('_')[1])
                    numbers.append(num)
                except:
                    pass
            next_num = max(numbers, default=0) + 1
        else:
            next_num = 1
        
        # Create new log file
        self.current_log = f"bagley_{next_num:04d}.log"
        self.log_file = os.path.join(self.log_dir, self.current_log)
        
        # Setup handler
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        
        # Cleanup old logs
        self._cleanup_old_logs()
        
        logger.info(f"Logging to {self.log_file}")
    
    def _cleanup_old_logs(self):
        """Remove old log files"""
        from pathlib import Path
        
        logs = sorted(Path(self.log_dir).glob("bagley_*.log"))
        while len(logs) > self.max_files:
            oldest = logs.pop(0)
            try:
                oldest.unlink()
                logger.debug(f"Deleted old log: {oldest}")
            except:
                pass
    
    def log(self, level: str, message: str, **kwargs):
        """Log a message with extra context"""
        extra = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
        full_msg = f"{message} | {extra}" if extra else message
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(full_msg)
    
    def get_recent_logs(self, lines: int = 100) -> List[str]:
        """Get recent log lines"""
        if not self.log_file or not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except:
            return []


# ==================== Global Instance ====================

_monitor: Optional[HardwareMonitor] = None
_smart_logger: Optional[SmartLogger] = None


def get_hardware_monitor() -> HardwareMonitor:
    """Get or create hardware monitor singleton"""
    global _monitor
    if _monitor is None:
        _monitor = HardwareMonitor()
        _monitor.start()
    return _monitor


def get_smart_logger() -> SmartLogger:
    """Get or create smart logger singleton"""
    global _smart_logger
    if _smart_logger is None:
        _smart_logger = SmartLogger()
    return _smart_logger


# ==================== Exports ====================

__all__ = [
    'GPUStats',
    'CPUStats', 
    'RAMStats',
    'SystemStats',
    'HardwareMonitor',
    'SmartLogger',
    'get_hardware_monitor',
    'get_smart_logger',
]
