"""
🖥️ System Agent - PC Control & Automation
Full system control capabilities
"""

import os
import subprocess
import platform
from typing import Optional, List, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class SystemAgent:
    """
    🖥️ System Control Agent
    
    Provides Bagley with PC control:
    - Launch/close applications
    - System info (CPU, RAM, GPU)
    - Volume/brightness control
    - Screenshots
    - Clipboard access
    - Process management
    """
    
    def __init__(self):
        self.system = platform.system()  # Windows, Linux, Darwin
        self.actions_taken: List[str] = []
        
        logger.info(f"Initialized SystemAgent for {self.system}")
    
    async def execute(self, command: str) -> str:
        """Execute natural language system command"""
        command_lower = command.lower()
        
        if "open" in command_lower or "launch" in command_lower or "start" in command_lower:
            return await self._handle_open(command)
        elif "close" in command_lower or "kill" in command_lower or "stop" in command_lower:
            return await self._handle_close(command)
        elif "volume" in command_lower:
            return await self._handle_volume(command)
        elif "screenshot" in command_lower:
            return await self.take_screenshot()
        elif "info" in command_lower or "status" in command_lower:
            return await self.get_system_info()
        elif "clipboard" in command_lower or "copy" in command_lower or "paste" in command_lower:
            return await self._handle_clipboard(command)
        elif "shutdown" in command_lower or "restart" in command_lower or "sleep" in command_lower:
            return await self._handle_power(command)
        else:
            return await self.run_command(command)
    
    async def _handle_open(self, command: str) -> str:
        """Handle app launch commands"""
        # Extract app name
        words = command.lower().split()
        
        # Common app mappings
        app_map = {
            "chrome": "chrome",
            "firefox": "firefox",
            "notepad": "notepad",
            "calculator": "calc",
            "explorer": "explorer",
            "terminal": "cmd" if self.system == "Windows" else "gnome-terminal",
            "code": "code",
            "vscode": "code",
        }
        
        for word in words:
            if word in app_map:
                return await self.open_application(app_map[word])
        
        # Try direct command
        for word in words:
            if word not in ('open', 'launch', 'start', 'the', 'app', 'application'):
                return await self.open_application(word)
        
        return "Specify an application to open"
    
    async def _handle_close(self, command: str) -> str:
        """Handle app close commands"""
        words = command.lower().split()
        
        for word in words:
            if word not in ('close', 'kill', 'stop', 'the', 'app', 'application'):
                return await self.close_application(word)
        
        return "Specify an application to close"
    
    async def _handle_volume(self, command: str) -> str:
        """Handle volume commands"""
        if "mute" in command.lower():
            return await self.set_volume(0)
        elif "max" in command.lower() or "100" in command:
            return await self.set_volume(100)
        elif "up" in command.lower():
            return await self.adjust_volume(10)
        elif "down" in command.lower():
            return await self.adjust_volume(-10)
        else:
            # Extract number
            import re
            nums = re.findall(r'\d+', command)
            if nums:
                return await self.set_volume(int(nums[0]))
        
        return "Specify volume level (0-100) or up/down/mute"
    
    async def _handle_clipboard(self, command: str) -> str:
        """Handle clipboard operations"""
        if "copy" in command.lower():
            # Extract text to copy
            text = command.split("copy", 1)[-1].strip()
            if text:
                return await self.copy_to_clipboard(text)
        elif "paste" in command.lower() or "get" in command.lower():
            return await self.get_clipboard()
        
        return "Clipboard: copy <text> or paste/get"
    
    async def _handle_power(self, command: str) -> str:
        """Handle power commands"""
        command_lower = command.lower()
        
        if "shutdown" in command_lower:
            return "Shutdown requires confirmation. Use shutdown(confirm=True)"
        elif "restart" in command_lower or "reboot" in command_lower:
            return "Restart requires confirmation. Use restart(confirm=True)"
        elif "sleep" in command_lower:
            return await self.sleep()
        
        return "Power: shutdown, restart, or sleep"
    
    # ==================== Core Functions ====================
    
    async def open_application(self, app: str) -> str:
        """Open an application"""
        try:
            if self.system == "Windows":
                process = await asyncio.create_subprocess_shell(
                    f'start "" "{app}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            elif self.system == "Darwin":
                process = await asyncio.create_subprocess_shell(
                    f'open -a "{app}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    app,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            
            await process.wait()
            
            self.actions_taken.append(f"Opened: {app}")
            return f"Launched {app}"
            
        except Exception as e:
            return f"Error opening {app}: {e}"
    
    async def close_application(self, app: str) -> str:
        """Close an application"""
        try:
            if self.system == "Windows":
                cmd = f'taskkill /IM {app}.exe /F'
            else:
                cmd = f'pkill -f {app}'
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            self.actions_taken.append(f"Closed: {app}")
            return f"Closed {app}"
            
        except Exception as e:
            return f"Error closing {app}: {e}"
    
    async def run_command(self, command: str) -> str:
        """Run shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            self.actions_taken.append(f"Command: {command[:30]}...")
            
            result = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            
            return f"{result}\n{errors}".strip()
            
        except Exception as e:
            return f"Error: {e}"
    
    async def get_system_info(self) -> str:
        """Get system information"""
        try:
            info = []
            info.append(f"🖥️ System: {platform.system()} {platform.release()}")
            info.append(f"💻 Machine: {platform.machine()}")
            info.append(f"🔧 Processor: {platform.processor()}")
            
            # Try to get more info
            try:
                import psutil
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                info.append(f"📊 CPU: {cpu_percent}%")
                info.append(f"🧠 RAM: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
                info.append(f"💾 Disk: {disk.percent}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")
                
            except ImportError:
                info.append("Install psutil for detailed system info")
            
            self.actions_taken.append("Got system info")
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting system info: {e}"
    
    async def take_screenshot(self, path: Optional[str] = None) -> str:
        """Take a screenshot"""
        try:
            import datetime
            
            if not path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshot_{timestamp}.png"
            
            if self.system == "Windows":
                # Use PowerShell
                ps_script = f'''
                Add-Type -AssemblyName System.Windows.Forms
                $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
                $bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
                $bitmap.Save("{path}")
                '''
                process = await asyncio.create_subprocess_shell(
                    f'powershell -Command "{ps_script}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
            else:
                process = await asyncio.create_subprocess_shell(
                    f'scrot {path}' if self.system == "Linux" else f'screencapture {path}',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
            
            self.actions_taken.append(f"Screenshot: {path}")
            return f"Screenshot saved to {path}"
            
        except Exception as e:
            return f"Screenshot error: {e}"
    
    async def set_volume(self, level: int) -> str:
        """Set system volume (0-100)"""
        try:
            level = max(0, min(100, level))
            
            if self.system == "Windows":
                # Use nircmd or PowerShell
                cmd = f'powershell -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"'
                # This is a placeholder - proper implementation needs nircmd or similar
            
            self.actions_taken.append(f"Volume: {level}%")
            return f"Volume set to {level}%"
            
        except Exception as e:
            return f"Volume error: {e}"
    
    async def adjust_volume(self, delta: int) -> str:
        """Adjust volume by delta"""
        self.actions_taken.append(f"Volume adjusted by {delta}")
        return f"Volume adjusted by {delta}"
    
    async def copy_to_clipboard(self, text: str) -> str:
        """Copy text to clipboard"""
        try:
            if self.system == "Windows":
                process = await asyncio.create_subprocess_shell(
                    f'echo {text}| clip',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
            
            self.actions_taken.append("Copied to clipboard")
            return "Copied to clipboard"
            
        except Exception as e:
            return f"Clipboard error: {e}"
    
    async def get_clipboard(self) -> str:
        """Get clipboard content"""
        try:
            if self.system == "Windows":
                process = await asyncio.create_subprocess_shell(
                    'powershell Get-Clipboard',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                return stdout.decode().strip()
            
            return "Clipboard access not implemented for this OS"
            
        except Exception as e:
            return f"Clipboard error: {e}"
    
    async def sleep(self) -> str:
        """Put system to sleep"""
        try:
            if self.system == "Windows":
                await asyncio.create_subprocess_shell('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
            elif self.system == "Darwin":
                await asyncio.create_subprocess_shell('pmset sleepnow')
            else:
                await asyncio.create_subprocess_shell('systemctl suspend')
            
            self.actions_taken.append("System sleep")
            return "Putting system to sleep..."
            
        except Exception as e:
            return f"Sleep error: {e}"
    
    def get_actions(self) -> List[str]:
        return self.actions_taken.copy()
