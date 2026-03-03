# accessibility/executor.py
"""Execute shell commands safely, optionally in containers."""
import subprocess
import shlex
from typing import Tuple, Optional

class CommandExecutor:
    def __init__(self, container=None, safe_mode: bool = True):
        self.container = container  # If None, runs locally
        self.safe_mode = safe_mode
        
        # Dangerous command patterns to block in safe mode
        self.dangerous_patterns = [
            "rm -rf /", "rm -rf /*", 
            "mkfs", "dd if=/dev/zero", 
            ":(){", "fork()", 
            "chmod 777 /", "chmod -R 777 /",
            "kill -9 -1", "killall",
            "> /dev/sd", "> /dev/hd"
        ]
        
    def _is_safe(self, command: str) -> Tuple[bool, str]:
        """Check if command is safe to execute"""
        if not self.safe_mode:
            return True, ""
            
        cmd_lower = command.lower()
        for pattern in self.dangerous_patterns:
            if pattern in cmd_lower:
                return False, f"Command blocked: '{pattern}' pattern detected"
        
        return True, ""
        
    def execute(self, command: str) -> Tuple[str, str]:
        """
        Execute command and return (stdout, stderr)
        
        Args:
            command: Shell command to execute
            
        Returns:
            Tuple of (stdout, stderr)
        """
        # Safety check
        safe, reason = self._is_safe(command)
        if not safe:
            return "", reason
        
        if self.container:
            # Execute in container
            output, exit_code = self.container.execute(command)
            if exit_code == 0:
                return output, ""
            else:
                return "", output
        else:
            # Execute locally
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return "", "Command timed out"
            except Exception as e:
                return "", str(e)
