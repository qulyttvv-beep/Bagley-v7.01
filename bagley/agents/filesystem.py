"""
📁 File System Agent - Full File/Folder Control
"""

import os
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class FileSystemAgent:
    """
    📁 File System Agent
    
    Full file system operations:
    - Create, read, write, delete files
    - Create, list, delete directories
    - Copy, move, rename
    - Search and filter
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.actions_taken: List[str] = []
        
        logger.info(f"Initialized FileSystemAgent at: {self.base_path}")
    
    async def execute(self, command: str, files: Optional[List[str]] = None) -> str:
        """Execute natural language file command"""
        command_lower = command.lower()
        
        if "read" in command_lower or "show" in command_lower or "content" in command_lower:
            return await self._handle_read(command, files)
        elif "write" in command_lower or "save" in command_lower:
            return await self._handle_write(command, files)
        elif "delete" in command_lower or "remove" in command_lower:
            return await self._handle_delete(command, files)
        elif "create" in command_lower or "new" in command_lower:
            return await self._handle_create(command)
        elif "copy" in command_lower:
            return await self._handle_copy(command, files)
        elif "move" in command_lower or "rename" in command_lower:
            return await self._handle_move(command, files)
        elif "list" in command_lower or "dir" in command_lower or "ls" in command_lower:
            return await self._handle_list(command)
        elif "search" in command_lower or "find" in command_lower:
            return await self._handle_search(command)
        else:
            return "Unknown file operation. Try: read, write, delete, create, copy, move, list, search"
    
    async def _handle_read(self, command: str, files: Optional[List[str]]) -> str:
        """Handle read operations"""
        if files:
            path = files[0]
        else:
            # Extract path from command
            path = self._extract_path(command)
        
        if path:
            return await self.read_file(path)
        return "Please specify a file to read"
    
    async def _handle_write(self, command: str, files: Optional[List[str]]) -> str:
        """Handle write operations"""
        return "Write requires: write_file(path, content)"
    
    async def _handle_delete(self, command: str, files: Optional[List[str]]) -> str:
        """Handle delete operations"""
        if files:
            path = files[0]
        else:
            path = self._extract_path(command)
        
        if path:
            return await self.delete(path)
        return "Please specify what to delete"
    
    async def _handle_create(self, command: str) -> str:
        """Handle creation"""
        if "folder" in command.lower() or "directory" in command.lower():
            path = self._extract_path(command)
            if path:
                return await self.create_directory(path)
        else:
            path = self._extract_path(command)
            if path:
                return await self.create_file(path)
        return "Please specify path to create"
    
    async def _handle_copy(self, command: str, files: Optional[List[str]]) -> str:
        """Handle copy operations"""
        return "Copy requires: copy(source, destination)"
    
    async def _handle_move(self, command: str, files: Optional[List[str]]) -> str:
        """Handle move/rename"""
        return "Move requires: move(source, destination)"
    
    async def _handle_list(self, command: str) -> str:
        """Handle list directory"""
        path = self._extract_path(command) or "."
        return await self.list_directory(path)
    
    async def _handle_search(self, command: str) -> str:
        """Handle search"""
        pattern = command.split()[-1]  # Last word as pattern
        return await self.search(pattern)
    
    def _extract_path(self, command: str) -> Optional[str]:
        """Extract file path from command"""
        words = command.split()
        for word in words:
            if '/' in word or '\\' in word or '.' in word:
                return word
        return None
    
    # ==================== Core Operations ====================
    
    async def read_file(self, path: str) -> str:
        """Read file content"""
        try:
            full_path = self._resolve_path(path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.actions_taken.append(f"Read: {path}")
            return content
            
        except Exception as e:
            return f"Error reading {path}: {e}"
    
    async def write_file(self, path: str, content: str) -> str:
        """Write content to file"""
        try:
            full_path = self._resolve_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.actions_taken.append(f"Wrote: {path}")
            return f"Successfully wrote to {path}"
            
        except Exception as e:
            return f"Error writing {path}: {e}"
    
    async def append_file(self, path: str, content: str) -> str:
        """Append content to file"""
        try:
            full_path = self._resolve_path(path)
            
            with open(full_path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            self.actions_taken.append(f"Appended to: {path}")
            return f"Successfully appended to {path}"
            
        except Exception as e:
            return f"Error appending to {path}: {e}"
    
    async def create_file(self, path: str, content: str = "") -> str:
        """Create a new file"""
        return await self.write_file(path, content)
    
    async def create_directory(self, path: str) -> str:
        """Create a directory"""
        try:
            full_path = self._resolve_path(path)
            full_path.mkdir(parents=True, exist_ok=True)
            
            self.actions_taken.append(f"Created directory: {path}")
            return f"Created directory {path}"
            
        except Exception as e:
            return f"Error creating directory {path}: {e}"
    
    async def delete(self, path: str) -> str:
        """Delete file or directory"""
        try:
            full_path = self._resolve_path(path)
            
            if full_path.is_file():
                full_path.unlink()
                self.actions_taken.append(f"Deleted file: {path}")
                return f"Deleted file {path}"
            elif full_path.is_dir():
                shutil.rmtree(full_path)
                self.actions_taken.append(f"Deleted directory: {path}")
                return f"Deleted directory {path}"
            else:
                return f"{path} not found"
            
        except Exception as e:
            return f"Error deleting {path}: {e}"
    
    async def copy(self, source: str, destination: str) -> str:
        """Copy file or directory"""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)
            
            if src.is_file():
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst)
            
            self.actions_taken.append(f"Copied: {source} -> {destination}")
            return f"Copied {source} to {destination}"
            
        except Exception as e:
            return f"Error copying: {e}"
    
    async def move(self, source: str, destination: str) -> str:
        """Move/rename file or directory"""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)
            
            shutil.move(str(src), str(dst))
            
            self.actions_taken.append(f"Moved: {source} -> {destination}")
            return f"Moved {source} to {destination}"
            
        except Exception as e:
            return f"Error moving: {e}"
    
    async def list_directory(self, path: str = ".") -> str:
        """List directory contents"""
        try:
            full_path = self._resolve_path(path)
            
            items = []
            for item in full_path.iterdir():
                prefix = "📁" if item.is_dir() else "📄"
                items.append(f"{prefix} {item.name}")
            
            self.actions_taken.append(f"Listed: {path}")
            return f"Contents of {path}:\n" + "\n".join(sorted(items))
            
        except Exception as e:
            return f"Error listing {path}: {e}"
    
    async def search(self, pattern: str, path: str = ".") -> str:
        """Search for files matching pattern"""
        try:
            full_path = self._resolve_path(path)
            
            matches = list(full_path.rglob(f"*{pattern}*"))[:20]  # Limit results
            
            if matches:
                result = "\n".join(str(m.relative_to(full_path)) for m in matches)
                return f"Found {len(matches)} matches:\n{result}"
            else:
                return f"No files matching '{pattern}' found"
            
        except Exception as e:
            return f"Error searching: {e}"
    
    async def get_info(self, path: str) -> Dict[str, Any]:
        """Get file/directory info"""
        try:
            full_path = self._resolve_path(path)
            stat = full_path.stat()
            
            return {
                "path": str(full_path),
                "is_file": full_path.is_file(),
                "is_dir": full_path.is_dir(),
                "size_bytes": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "extension": full_path.suffix,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p
    
    def get_actions(self) -> List[str]:
        return self.actions_taken.copy()
