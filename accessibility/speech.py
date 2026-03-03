# accessibility/speech.py (simplified version)
"""Text-to-speech utilities."""
import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)

class SpeechEngine:
    def __init__(self, engine="auto"):
        self.engine = self._detect_engine(engine)
    
    def _detect_engine(self, preferred):
        if preferred != "auto" and shutil.which(preferred):
            return preferred
        for engine in ["espeak", "festival"]:
            if shutil.which(engine):
                return engine
        return None
    
    def speak(self, text):
        if not self.engine:
            return False
        try:
            if self.engine == "espeak":
                subprocess.run(["espeak", text], timeout=5)
            elif self.engine == "festival":
                subprocess.run(["festival", "--tts"], input=text, text=True, timeout=5)
            return True
        except Exception as e:
            logger.error(f"Speech failed: {e}")
            return False
    
    def say_command(self, command):
        self.speak(f"Command: {command}")
