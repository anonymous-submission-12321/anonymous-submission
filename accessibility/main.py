# accessibility/main.py
"""Main integration entry point for accessibility tools."""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accessibility.container import ContainerManager
from accessibility.model_loader import QLoRALoader
from accessibility.command_recognizer import CommandRecognizer
from accessibility.executor import CommandExecutor

# Optional accessibility bridges
try:
    from accessibility.orca_bridge import OrcaBridge
    from accessibility.speech import SpeechEngine
    HAS_ACCESSIBILITY = True
except ImportError:
    HAS_ACCESSIBILITY = False

def parse_args():
    parser = argparse.ArgumentParser(description="Accessible Linux Command Generator")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/qlora-finetuned",
                       help="Path to fine-tuned model checkpoint")
    parser.add_argument("--container", action="store_true", 
                       help="Run commands in Docker container for safety")
    parser.add_argument("--image", type=str, default="qlora-command:latest",
                       help="Docker image name")
    parser.add_argument("--local", action="store_true",
                       help="Run commands locally (not recommended for production)")
    parser.add_argument("--safe-mode", action="store_true", default=True,
                       help="Enable dangerous command blocking")
    parser.add_argument("--orca", action="store_true",
                       help="Integrate with Orca screen reader")
    parser.add_argument("--speech", type=str, choices=["espeak", "festival", "none"],
                       default="none", help="Text-to-speech engine")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing accessibility command tool...")
    
    # Module 1: Container (optional)
    container = None
    if args.container:
        container = ContainerManager(image_name=args.image)
        if not container.start(args.checkpoint):
            print("Failed to start container, falling back to local execution")
            container = None
    
    # Module 2: Model Loader
    loader = QLoRALoader(args.checkpoint)
    model, tokenizer = loader.load()
    
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Module 3: Command Recognizer
    recognizer = CommandRecognizer(model, tokenizer)
    
    # Module 4: Executor
    executor = CommandExecutor(
        container=container,
        safe_mode=args.safe_mode and not args.local
    )
    
    # Optional accessibility bridges
    orca = None
    speech = None
    
    if HAS_ACCESSIBILITY:
        if args.orca:
            from accessibility.orca_bridge import OrcaBridge
            orca = OrcaBridge(recognizer)
            if orca.is_running:
                orca.speak("Command generator ready")
        
        if args.speech != "none":
            from accessibility.speech import SpeechEngine
            speech = SpeechEngine(engine=args.speech)
    
    # Main loop
    print("\nReady. Enter natural language commands (or 'exit' to quit):")
    
    while True:
        try:
            # Get input (supports both CLI and accessibility input)
            if orca and hasattr(orca, 'get_input'):
                user_input = orca.get_input()
            else:
                user_input = input("> ")
                
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
            
            # Recognize
            command = recognizer.recognize(user_input)
            
            # Announce command
            if orca:
                orca.speak(f"Command: {command}")
            if speech:
                speech.say_command(command)
            else:
                print(f"$ {command}")
            
            # Execute
            stdout, stderr = executor.execute(command)
            
            # Output
            if stdout:
                if orca:
                    orca.speak(f"Output: {stdout[:200]}")
                if speech:
                    speech.speak(f"Output: {stdout[:200]}")
                else:
                    print(stdout)
                    
            if stderr:
                if orca:
                    orca.speak(f"Error: {stderr[:200]}")
                if speech:
                    speech.speak(f"Error: {stderr[:200]}")
                else:
                    print(f"Error: {stderr}")
                    
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    loader.unload()
    if container:
        container.stop()
    print("Exited.")

if __name__ == "__main__":
    main()
