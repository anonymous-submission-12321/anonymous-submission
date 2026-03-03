# accessibility/orca_bridge.py (simplified version)
"""Integration with Orca screen reader."""
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OrcaBridge:
    def __init__(self, command_generator=None):
        self.command_generator = command_generator
        self.is_running = self._check_orca()
    
    def _check_orca(self) -> bool:
        """Check if Orca is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-x", "orca"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def speak(self, text: str) -> bool:
        """Send text to Orca"""
        if not self.is_running:
            return False
        try:
            # Use orca's text-to-speech
            subprocess.run(
                ["orca", "--speak", text],
                timeout=2,
                capture_output=True
            )
            return True
        except:
            return False
    
    def suggest_command(self, query: str) -> Optional[str]:
        """Generate and speak command"""
        if not self.command_generator:
            return None
        command = self.command_generator.recognize(query)
        self.speak(f"Suggested command: {command}")
        return command
