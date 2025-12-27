"""
💻 VS Code Agent - Full Editor Control
Autonomous code editing, writing, and execution
"""

import os
import json
import subprocess
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class VSCodeAgent:
    """
    💻 VS Code Integration Agent
    
    Provides Bagley with full control over VS Code:
    - Open/close files and folders
    - Edit code with precise changes
    - Run code and capture output
    - Navigate and search codebase
    - Manage extensions
    - Debug support
    
    Uses VS Code's CLI and potentially LSP for rich functionality.
    """
    
    def __init__(self, vscode_path: Optional[str] = None):
        self.vscode_path = vscode_path or self._find_vscode()
        self.actions_taken: List[str] = []
        self.current_workspace: Optional[str] = None
        
        logger.info(f"Initialized VSCodeAgent with VS Code at: {self.vscode_path}")
    
    def _find_vscode(self) -> str:
        """Find VS Code installation"""
        # Common paths
        paths = [
            r"C:\Users\{}\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd",
            r"C:\Program Files\Microsoft VS Code\bin\code.cmd",
            "/usr/bin/code",
            "/usr/local/bin/code",
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        ]
        
        # Try to find with current user
        import getpass
        user = getpass.getuser()
        
        for path in paths:
            p = path.format(user) if '{}' in path else path
            if os.path.exists(p):
                return p
        
        # Fallback to just "code" (assumes it's in PATH)
        return "code"
    
    async def execute(self, command: str) -> str:
        """
        Execute a natural language command.
        
        Args:
            command: Natural language instruction (e.g., "open file main.py")
            
        Returns:
            Result description
        """
        command_lower = command.lower()
        
        # Parse and route command
        if "open" in command_lower and ("file" in command_lower or "folder" in command_lower):
            return await self._handle_open(command)
        elif "edit" in command_lower or "change" in command_lower or "modify" in command_lower:
            return await self._handle_edit(command)
        elif "run" in command_lower or "execute" in command_lower:
            return await self._handle_run(command)
        elif "search" in command_lower or "find" in command_lower:
            return await self._handle_search(command)
        elif "create" in command_lower or "new" in command_lower:
            return await self._handle_create(command)
        elif "close" in command_lower:
            return await self._handle_close(command)
        else:
            return f"Didn't understand command: {command}. Try: open, edit, run, search, create, close"
    
    async def _handle_open(self, command: str) -> str:
        """Handle file/folder open commands"""
        # Extract path from command (simple heuristic)
        words = command.split()
        path = None
        
        for i, word in enumerate(words):
            if word.lower() in ('file', 'folder', 'directory'):
                if i + 1 < len(words):
                    path = words[i + 1]
                    break
        
        if not path:
            # Try to find a path-like word
            for word in words:
                if '/' in word or '\\' in word or '.' in word:
                    path = word
                    break
        
        if path:
            return await self.open_file(path)
        else:
            return "Couldn't find a file path in the command"
    
    async def _handle_edit(self, command: str) -> str:
        """Handle edit commands"""
        self.actions_taken.append(f"Edit command: {command[:50]}...")
        return "Edit functionality requires specific file and changes. Use: edit_file(path, changes)"
    
    async def _handle_run(self, command: str) -> str:
        """Handle run commands"""
        words = command.split()
        
        # Find file to run
        for word in words:
            if word.endswith('.py'):
                return await self.run_python(word)
            elif word.endswith('.js'):
                return await self.run_node(word)
        
        return "Specify a file to run (e.g., 'run main.py')"
    
    async def _handle_search(self, command: str) -> str:
        """Handle search commands"""
        # Extract search term
        if 'for' in command.lower():
            idx = command.lower().index('for')
            term = command[idx + 3:].strip()
        else:
            term = ' '.join(command.split()[1:])
        
        return await self.search_workspace(term)
    
    async def _handle_create(self, command: str) -> str:
        """Handle file creation"""
        words = command.split()
        
        filename = None
        for i, word in enumerate(words):
            if word.lower() == 'file' and i + 1 < len(words):
                filename = words[i + 1]
                break
        
        if filename:
            return await self.create_file(filename)
        
        return "Specify filename: create file <filename>"
    
    async def _handle_close(self, command: str) -> str:
        """Handle close commands"""
        self.actions_taken.append("Closed editor")
        return "File closed"
    
    # ==================== Core Functions ====================
    
    async def open_file(self, path: str) -> str:
        """Open a file in VS Code"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.vscode_path, path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            self.actions_taken.append(f"Opened file: {path}")
            return f"Opened {path} in VS Code"
            
        except Exception as e:
            return f"Error opening file: {e}"
    
    async def open_folder(self, path: str) -> str:
        """Open a folder in VS Code"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.vscode_path, path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            self.current_workspace = path
            self.actions_taken.append(f"Opened folder: {path}")
            return f"Opened folder {path} in VS Code"
            
        except Exception as e:
            return f"Error opening folder: {e}"
    
    async def create_file(self, path: str, content: str = "") -> str:
        """Create a new file"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            await self.open_file(path)
            
            self.actions_taken.append(f"Created file: {path}")
            return f"Created and opened {path}"
            
        except Exception as e:
            return f"Error creating file: {e}"
    
    async def edit_file(
        self, 
        path: str, 
        old_content: str, 
        new_content: str
    ) -> str:
        """Edit a file by replacing content"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_content not in content:
                return f"Could not find the specified content in {path}"
            
            new_file_content = content.replace(old_content, new_content, 1)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            
            self.actions_taken.append(f"Edited file: {path}")
            return f"Successfully edited {path}"
            
        except Exception as e:
            return f"Error editing file: {e}"
    
    async def run_python(self, path: str) -> str:
        """Run a Python file"""
        try:
            process = await asyncio.create_subprocess_exec(
                'python', path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            self.actions_taken.append(f"Ran Python: {path}")
            
            result = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            
            if errors:
                return f"Output:\n{result}\n\nErrors:\n{errors}"
            return f"Output:\n{result}"
            
        except Exception as e:
            return f"Error running Python: {e}"
    
    async def run_node(self, path: str) -> str:
        """Run a Node.js file"""
        try:
            process = await asyncio.create_subprocess_exec(
                'node', path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            self.actions_taken.append(f"Ran Node: {path}")
            
            result = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            
            if errors:
                return f"Output:\n{result}\n\nErrors:\n{errors}"
            return f"Output:\n{result}"
            
        except Exception as e:
            return f"Error running Node: {e}"
    
    async def run_terminal(self, command: str) -> str:
        """Run an arbitrary terminal command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            self.actions_taken.append(f"Terminal: {command[:30]}...")
            
            result = stdout.decode() if stdout else ""
            errors = stderr.decode() if stderr else ""
            
            return f"Output:\n{result}\n{errors}".strip()
            
        except Exception as e:
            return f"Error running command: {e}"
    
    async def search_workspace(self, query: str) -> str:
        """Search for text in workspace"""
        if not self.current_workspace:
            return "No workspace open. Open a folder first."
        
        try:
            # Use grep/findstr to search
            if os.name == 'nt':
                cmd = f'findstr /s /i /n "{query}" *.*'
            else:
                cmd = f'grep -r -n -i "{query}" .'
            
            process = await asyncio.create_subprocess_shell(
                cmd,
                cwd=self.current_workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            results = stdout.decode()[:2000]  # Limit output
            
            self.actions_taken.append(f"Searched for: {query}")
            return f"Search results for '{query}':\n{results}"
            
        except Exception as e:
            return f"Error searching: {e}"
    
    async def get_file_content(self, path: str) -> str:
        """Read file content"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def list_files(self, path: str = ".") -> List[str]:
        """List files in directory"""
        try:
            return [str(p) for p in Path(path).rglob("*") if p.is_file()]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_actions(self) -> List[str]:
        """Get list of actions taken"""
        return self.actions_taken.copy()
    
    def clear_actions(self):
        """Clear action history"""
        self.actions_taken = []
