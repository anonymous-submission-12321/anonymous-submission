# accessibility/container.py
"""Docker container management for safe command execution."""
import docker
import os
from typing import Optional, Tuple

class ContainerManager:
    def __init__(self, image_name="qlora-command:latest"):
        self.client = docker.from_env()
        self.image_name = image_name
        self.container = None
        
    def start(self, checkpoint_path: str) -> Optional[docker.models.containers.Container]:
        """Start container with mounted model weights"""
        abs_path = os.path.abspath(checkpoint_path)
        
        # Ensure directory exists
        if not os.path.exists(abs_path):
            os.makedirs(abs_path, exist_ok=True)
        
        try:
            self.container = self.client.containers.run(
                self.image_name,
                detach=True,
                tty=True,
                stdin_open=True,
                volumes={abs_path: {'bind': '/checkpoints', 'mode': 'ro'}},
                environment=["CUDA_VISIBLE_DEVICES=0"],
                remove=True  # Auto-remove on stop
            )
            return self.container
        except docker.errors.ImageNotFound:
            print(f"Error: Image {self.image_name} not found. Please build it first.")
            return None
        except Exception as e:
            print(f"Error starting container: {e}")
            return None
        
    def stop(self):
        """Stop and remove container"""
        if self.container:
            try:
                self.container.stop(timeout=5)
            except Exception as e:
                print(f"Error stopping container: {e}")
            finally:
                self.container = None
            
    def execute(self, command: str) -> Tuple[str, int]:
        """Run command inside container"""
        if not self.container:
            return "", 1
        
        try:
            exit_code, output = self.container.exec_run(
                command, 
                stdout=True, 
                stderr=True
            )
            return output.decode('utf-8'), exit_code
        except Exception as e:
            return str(e), 1
