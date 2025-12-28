"""
🌡️ Hardware Monitor
===================
Always-on GPU and CPU temperature monitoring
Works with NVIDIA, AMD, Intel, and fallback methods
"""

import os
import sys
import time
import logging
import threading
import subprocess
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== Data Classes ====================

@dataclass
class GPUStats:
    """GPU statistics"""
    index: int
    name: str
    temperature: float  # Celsius
    utilization: float  # Percentage
    memory_used: float  # GB
    memory_total: float  # GB
    fan_speed: float  # Percentage
    power_draw: float  # Watts
    clock_speed: int  # MHz
    vendor: str  # nvidia, amd, intel


@dataclass
class CPUStats:
    """CPU statistics"""
    name: str
    temperature: float  # Celsius
    utilization: float  # Percentage
    frequency: float  # GHz
    core_count: int
    thread_count: int
    per_core_temps: List[float] = field(default_factory=list)
    per_core_utils: List[float] = field(default_factory=list)


@dataclass
class SystemStats:
    """Overall system statistics"""
    timestamp: datetime
    gpus: List[GPUStats]
    cpu: CPUStats
    ram_used_gb: float
    ram_total_gb: float
    disk_used_gb: float
    disk_total_gb: float


# ==================== GPU Monitor ====================

class GPUMonitor:
    """
    🎮 GPU Temperature and Stats Monitor
    Supports NVIDIA (NVML), AMD (ROCm), and fallbacks
    """
    
    def __init__(self):
        self.nvidia_available = False
        self.amd_available = False
        self.nvml = None
        
        self._init_nvidia()
        self._init_amd()
    
    def _init_nvidia(self):
        """Initialize NVIDIA monitoring via NVML"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvidia_available = True
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"✅ NVIDIA NVML initialized - {device_count} GPU(s)")
        except ImportError:
            logger.debug("pynvml not installed")
        except Exception as e:
            logger.debug(f"NVML init failed: {e}")
    
    def _init_amd(self):
        """Initialize AMD monitoring"""
        try:
            # Try ROCm SMI
            result = subprocess.run(
                ['rocm-smi', '--showtemp'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.amd_available = True
                logger.info("✅ AMD ROCm-SMI available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.debug(f"ROCm init failed: {e}")
    
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get stats for all GPUs"""
        stats = []
        
        # NVIDIA GPUs
        if self.nvidia_available:
            stats.extend(self._get_nvidia_stats())
        
        # AMD GPUs
        if self.amd_available:
            stats.extend(self._get_amd_stats())
        
        # Fallback: try nvidia-smi CLI
        if not stats:
            stats.extend(self._get_nvidia_smi_fallback())
        
        return stats
    
    def _get_nvidia_stats(self) -> List[GPUStats]:
        """Get NVIDIA GPU stats via NVML"""
        stats = []
        
        try:
            device_count = self.nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Name
                name = self.nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Temperature
                try:
                    temp = self.nvml.nvmlDeviceGetTemperature(
                        handle, 
                        self.nvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temp = 0.0
                
                # Utilization
                try:
                    util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0.0
                
                # Memory
                try:
                    mem = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem.used / (1024**3)
                    mem_total = mem.total / (1024**3)
                except:
                    mem_used = 0.0
                    mem_total = 0.0
                
                # Fan speed
                try:
                    fan = self.nvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan = 0.0
                
                # Power
                try:
                    power = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                except:
                    power = 0.0
                
                # Clock
                try:
                    clock = self.nvml.nvmlDeviceGetClockInfo(
                        handle,
                        self.nvml.NVML_CLOCK_GRAPHICS
                    )
                except:
                    clock = 0
                
                stats.append(GPUStats(
                    index=i,
                    name=name,
                    temperature=temp,
                    utilization=gpu_util,
                    memory_used=mem_used,
                    memory_total=mem_total,
                    fan_speed=fan,
                    power_draw=power,
                    clock_speed=clock,
                    vendor='nvidia'
                ))
                
        except Exception as e:
            logger.error(f"NVML error: {e}")
        
        return stats
    
    def _get_amd_stats(self) -> List[GPUStats]:
        """Get AMD GPU stats via ROCm-SMI"""
        stats = []
        
        try:
            # Get temperature
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--json'],
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
                        
                        # Get more info
                        mem_result = subprocess.run(
                            ['rocm-smi', '--showmeminfo', 'vram', '--json'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        mem_used = 0.0
                        mem_total = 0.0
                        if mem_result.returncode == 0:
                            mem_data = json.loads(mem_result.stdout)
                            if gpu_id in mem_data:
                                mem_info = mem_data[gpu_id]
                                mem_total = float(mem_info.get('VRAM Total Memory (B)', 0)) / (1024**3)
                                mem_used = float(mem_info.get('VRAM Total Used Memory (B)', 0)) / (1024**3)
                        
                        stats.append(GPUStats(
                            index=idx,
                            name=gpu_data.get('Card series', f'AMD GPU {idx}'),
                            temperature=float(gpu_data.get('Temperature (Sensor edge) (C)', 0)),
                            utilization=0.0,  # Would need additional query
                            memory_used=mem_used,
                            memory_total=mem_total,
                            fan_speed=0.0,
                            power_draw=0.0,
                            clock_speed=0,
                            vendor='amd'
                        ))
                        
        except Exception as e:
            logger.debug(f"ROCm-SMI error: {e}")
        
        return stats
    
    def _get_nvidia_smi_fallback(self) -> List[GPUStats]:
        """Fallback: use nvidia-smi CLI"""
        stats = []
        
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,fan.speed,power.draw,clocks.gr',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 9:
                            stats.append(GPUStats(
                                index=int(parts[0]),
                                name=parts[1],
                                temperature=float(parts[2]) if parts[2] not in ['[N/A]', ''] else 0,
                                utilization=float(parts[3]) if parts[3] not in ['[N/A]', ''] else 0,
                                memory_used=float(parts[4]) / 1024 if parts[4] not in ['[N/A]', ''] else 0,
                                memory_total=float(parts[5]) / 1024 if parts[5] not in ['[N/A]', ''] else 0,
                                fan_speed=float(parts[6]) if parts[6] not in ['[N/A]', ''] else 0,
                                power_draw=float(parts[7]) if parts[7] not in ['[N/A]', ''] else 0,
                                clock_speed=int(float(parts[8])) if parts[8] not in ['[N/A]', ''] else 0,
                                vendor='nvidia'
                            ))
                            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            logger.debug(f"nvidia-smi fallback error: {e}")
        
        return stats
    
    def get_temperatures(self) -> List[float]:
        """Get just temperatures for all GPUs"""
        stats = self.get_gpu_stats()
        return [s.temperature for s in stats]
    
    def shutdown(self):
        """Cleanup NVML"""
        if self.nvidia_available and self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except:
                pass


# ==================== CPU Monitor ====================

class CPUMonitor:
    """
    🖥️ CPU Temperature and Stats Monitor
    Works with Windows, Linux, and macOS
    """
    
    def __init__(self):
        self.psutil_available = False
        self.wmi_available = False
        
        self._init_psutil()
        self._init_wmi()
    
    def _init_psutil(self):
        """Initialize psutil"""
        try:
            import psutil
            self.psutil = psutil
            self.psutil_available = True
            logger.info("✅ psutil available for CPU monitoring")
        except ImportError:
            logger.debug("psutil not installed")
    
    def _init_wmi(self):
        """Initialize WMI for Windows temperature"""
        if sys.platform == 'win32':
            try:
                import wmi
                self.wmi = wmi.WMI(namespace="root\\wmi")
                self.wmi_available = True
                logger.info("✅ WMI available for CPU temperature")
            except ImportError:
                logger.debug("WMI not installed")
            except Exception as e:
                logger.debug(f"WMI init failed: {e}")
    
    def get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics"""
        # Name
        name = self._get_cpu_name()
        
        # Temperature
        temp, per_core = self._get_cpu_temperature()
        
        # Utilization
        util = 0.0
        per_core_utils = []
        if self.psutil_available:
            util = self.psutil.cpu_percent(interval=0.1)
            per_core_utils = self.psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Frequency
        freq = 0.0
        if self.psutil_available:
            freq_info = self.psutil.cpu_freq()
            if freq_info:
                freq = freq_info.current / 1000  # MHz to GHz
        
        # Core/thread count
        cores = os.cpu_count() or 1
        threads = cores
        if self.psutil_available:
            cores = self.psutil.cpu_count(logical=False) or cores
            threads = self.psutil.cpu_count(logical=True) or threads
        
        return CPUStats(
            name=name,
            temperature=temp,
            utilization=util,
            frequency=freq,
            core_count=cores,
            thread_count=threads,
            per_core_temps=per_core,
            per_core_utils=per_core_utils
        )
    
    def _get_cpu_name(self) -> str:
        """Get CPU name"""
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                )
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return name.strip()
            except:
                pass
        
        elif sys.platform == 'linux':
            try:
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            except:
                pass
        
        return "Unknown CPU"
    
    def _get_cpu_temperature(self) -> Tuple[float, List[float]]:
        """Get CPU temperature"""
        temp = 0.0
        per_core = []
        
        # Method 1: psutil sensors (Linux)
        if self.psutil_available and hasattr(self.psutil, 'sensors_temperatures'):
            try:
                temps = self.psutil.sensors_temperatures()
                if temps:
                    # Try different sensor names
                    for name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                        if name in temps:
                            sensor_temps = temps[name]
                            if sensor_temps:
                                per_core = [s.current for s in sensor_temps]
                                temp = sum(per_core) / len(per_core)
                                return temp, per_core
            except:
                pass
        
        # Method 2: WMI (Windows)
        if self.wmi_available:
            try:
                sensors = self.wmi.MSAcpi_ThermalZoneTemperature()
                if sensors:
                    temps = []
                    for sensor in sensors:
                        # WMI returns in tenths of Kelvin
                        celsius = (sensor.CurrentTemperature / 10.0) - 273.15
                        temps.append(celsius)
                    if temps:
                        per_core = temps
                        temp = sum(temps) / len(temps)
                        return temp, per_core
            except:
                pass
        
        # Method 3: OpenHardwareMonitor / LibreHardwareMonitor (Windows)
        if sys.platform == 'win32':
            try:
                import wmi
                w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                sensors = w.Sensor()
                cpu_temps = [
                    s.Value for s in sensors 
                    if s.SensorType == 'Temperature' and 'CPU' in s.Name
                ]
                if cpu_temps:
                    per_core = cpu_temps
                    temp = sum(cpu_temps) / len(cpu_temps)
                    return temp, per_core
            except:
                pass
        
        # Method 4: lm-sensors CLI (Linux)
        if sys.platform == 'linux':
            try:
                result = subprocess.run(
                    ['sensors', '-u'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    import re
                    temps = re.findall(r'temp\d+_input:\s*([\d.]+)', result.stdout)
                    if temps:
                        per_core = [float(t) for t in temps]
                        temp = sum(per_core) / len(per_core)
                        return temp, per_core
            except:
                pass
        
        # Method 5: Thermal zone (Linux)
        if sys.platform == 'linux':
            try:
                thermal_zones = Path('/sys/class/thermal/')
                if thermal_zones.exists():
                    temps = []
                    for zone in thermal_zones.glob('thermal_zone*'):
                        temp_file = zone / 'temp'
                        if temp_file.exists():
                            with open(temp_file) as f:
                                temps.append(int(f.read().strip()) / 1000)
                    if temps:
                        per_core = temps
                        temp = sum(temps) / len(temps)
                        return temp, per_core
            except:
                pass
        
        return temp, per_core
    
    def get_temperature(self) -> float:
        """Get just CPU temperature"""
        temp, _ = self._get_cpu_temperature()
        return temp


# ==================== System Monitor ====================

class SystemMonitor:
    """
    🖥️ Combined system monitor
    GPU + CPU + RAM + Disk
    """
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.cpu_monitor = CPUMonitor()
        
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
    
    def get_stats(self) -> SystemStats:
        """Get all system stats"""
        # GPU
        gpus = self.gpu_monitor.get_gpu_stats()
        
        # CPU
        cpu = self.cpu_monitor.get_cpu_stats()
        
        # RAM
        ram_used = 0.0
        ram_total = 0.0
        if self.psutil:
            mem = self.psutil.virtual_memory()
            ram_used = mem.used / (1024**3)
            ram_total = mem.total / (1024**3)
        
        # Disk
        disk_used = 0.0
        disk_total = 0.0
        if self.psutil:
            try:
                disk = self.psutil.disk_usage('/')
                disk_used = disk.used / (1024**3)
                disk_total = disk.total / (1024**3)
            except:
                pass
        
        return SystemStats(
            timestamp=datetime.now(),
            gpus=gpus,
            cpu=cpu,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            disk_used_gb=disk_used,
            disk_total_gb=disk_total
        )
    
    def shutdown(self):
        """Cleanup"""
        self.gpu_monitor.shutdown()


# ==================== Background Monitor ====================

class BackgroundMonitor:
    """
    📊 Always-on background monitoring
    Updates stats at regular intervals
    """
    
    def __init__(
        self,
        update_interval: float = 2.0,
        on_update: Optional[Callable[[SystemStats], None]] = None,
        on_overheat: Optional[Callable[[str, float], None]] = None,
        gpu_temp_limit: float = 85.0,
        cpu_temp_limit: float = 90.0
    ):
        self.update_interval = update_interval
        self.on_update = on_update
        self.on_overheat = on_overheat
        self.gpu_temp_limit = gpu_temp_limit
        self.cpu_temp_limit = cpu_temp_limit
        
        self.monitor = SystemMonitor()
        self.is_running = False
        self.thread = None
        self.latest_stats: Optional[SystemStats] = None
        
        # History for graphing
        self.history: List[SystemStats] = []
        self.max_history = 300  # ~10 minutes at 2s interval
    
    def start(self):
        """Start background monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("📊 Background monitoring started")
    
    def stop(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.monitor.shutdown()
        logger.info("📊 Background monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                stats = self.monitor.get_stats()
                self.latest_stats = stats
                
                # Add to history
                self.history.append(stats)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Check for overheating
                self._check_temps(stats)
                
                # Callback
                if self.on_update:
                    self.on_update(stats)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.update_interval)
    
    def _check_temps(self, stats: SystemStats):
        """Check temperatures and trigger callback if overheating"""
        if self.on_overheat:
            # Check GPUs
            for gpu in stats.gpus:
                if gpu.temperature >= self.gpu_temp_limit:
                    self.on_overheat(f"GPU {gpu.index} ({gpu.name})", gpu.temperature)
            
            # Check CPU
            if stats.cpu.temperature >= self.cpu_temp_limit:
                self.on_overheat("CPU", stats.cpu.temperature)
    
    def get_latest(self) -> Optional[SystemStats]:
        """Get latest stats"""
        return self.latest_stats
    
    def get_history(self) -> List[SystemStats]:
        """Get stats history"""
        return self.history


# ==================== Factory ====================

def create_monitor(
    update_interval: float = 2.0,
    auto_start: bool = True
) -> BackgroundMonitor:
    """Create a background monitor"""
    monitor = BackgroundMonitor(update_interval=update_interval)
    if auto_start:
        monitor.start()
    return monitor


# ==================== Exports ====================

__all__ = [
    'GPUStats',
    'CPUStats',
    'SystemStats',
    'GPUMonitor',
    'CPUMonitor',
    'SystemMonitor',
    'BackgroundMonitor',
    'create_monitor',
]
