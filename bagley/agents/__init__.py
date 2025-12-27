"""
🤖 Bagley Agents - PC Integration & Automation
VS Code, File System, Browser, and System Control
"""

from bagley.agents.vscode import VSCodeAgent
from bagley.agents.filesystem import FileSystemAgent
from bagley.agents.browser import BrowserAgent
from bagley.agents.system import SystemAgent

__all__ = [
    "VSCodeAgent",
    "FileSystemAgent",
    "BrowserAgent",
    "SystemAgent",
]
