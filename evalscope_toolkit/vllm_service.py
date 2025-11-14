"""vLLM service management"""

import os
import sys
import time
import socket
import signal
import random
import hashlib
import subprocess
import requests
from pathlib import Path
from typing import Optional, Tuple


class VLLMService:
    """Manages vLLM service lifecycle"""
    
    def __init__(self, config):
        """Initialize vLLM service manager
        
        Args:
            config: EvalConfig instance
        """
        self.config = config
        self.process = None
        self.port = None
        self.served_model_name = None
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0
    
    @staticmethod
    def pick_free_port(base_port: int) -> int:
        """Pick a free port starting from base_port"""
        port = base_port + random.randint(0, 100)
        max_attempts = 100
        
        for _ in range(max_attempts):
            if not VLLMService.is_port_in_use(port):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    pass
            port += 1
            if port > 65535:
                port = 20000 + random.randint(0, 10000)
        
        raise RuntimeError(f"Could not find available port (starting from {base_port})")
    
    @staticmethod
    def is_local_path(path: str) -> bool:
        """Check if path is a local directory"""
        return Path(path).exists()
    
    def start(self, model_ref: str, log_dir: Path) -> Tuple[int, str]:
        """Start vLLM service for a model
        
        Args:
            model_ref: Model path or HuggingFace repo ID
            log_dir: Directory for logs
            
        Returns:
            Tuple of (port, served_model_name)
        """
        # Generate served model name
        served_name = Path(model_ref).name if self.is_local_path(model_ref) else model_ref
        self.served_model_name = f"{served_name}_{self.config.user_id}"
        
        # Pick a free port
        self.port = self.pick_free_port(self.config.base_port)
        
        # Generate unique identifier
        model_hash = hashlib.md5(
            f"{served_name}_{self.config.user_id}_{os.getpid()}".encode()
        ).hexdigest()[:8]
        run_tmp = Path(f"/tmp/vllm_{self.config.user_id}_{model_hash}_{os.getpid()}")
        run_tmp.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        env = os.environ.copy()
        env['TMPDIR'] = str(run_tmp)
        env['XDG_RUNTIME_DIR'] = str(run_tmp)
        env['VLLM_INSTANCE_ID'] = f"eval_{self.config.user_id}_{model_hash}_{os.getpid()}"
        env['MASTER_ADDR'] = '127.0.0.1'
        env['MASTER_PORT'] = str(self.pick_free_port(20000))
        env['CUDA_VISIBLE_DEVICES'] = self.config.gpus
        
        vllm_log = log_dir / f"vllm_{model_hash}.log"
        print(f"Starting vLLM {model_ref} on port {self.port} (MASTER_PORT={env['MASTER_PORT']})")
        
        # Build command
        model_arg = str(Path(model_ref).resolve()) if self.is_local_path(model_ref) else model_ref
        
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model_arg,
            '--served-model-name', self.served_model_name,
            '--trust-remote-code',
            '--port', str(self.port),
            '--tensor-parallel-size', str(self.config.tensor_parallel_size),
            '--gpu-memory-utilization', str(self.config.gpu_memory_utilization),
            '--max-num-seqs', str(self.config.max_num_seqs),
            '--disable-log-requests',
            '--disable-log-stats',
        ]
        
        # Add chat template if exists
        if self.config.chat_template and self.config.chat_template.exists():
            cmd.extend(['--chat-template', str(self.config.chat_template)])
        
        # Check if model is accessible
        print(f"Checking model accessibility: {model_ref}")
        if not self.is_local_path(model_ref):
            try:
                response = requests.head(f"https://huggingface.co/{model_ref}", timeout=10)
                if response.status_code == 200:
                    print(f"✓ HuggingFace model accessible: {model_ref}")
                else:
                    print(f"⚠ HuggingFace model may not be accessible: {model_ref} (status: {response.status_code})")
            except Exception as e:
                print(f"⚠ Could not check HuggingFace model: {e}")
        
        # Start process
        with open(vllm_log, 'w') as log_file:
            self.process = subprocess.Popen(
                cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT
            )
        
        print(f"vLLM PID={self.process.pid}, log={vllm_log}")
        
        # Wait for service to be ready
        print("Waiting for vLLM to be ready...")
        for i in range(300):  # 5 minutes timeout
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"vLLM process exited with code {self.process.returncode}")
                if vllm_log.exists():
                    print("\nFull vLLM log:")
                    with open(vllm_log) as f:
                        print(f.read())
                raise RuntimeError(f"vLLM process failed to start (exit code: {self.process.returncode})")
            
            # Check service health
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    print("✓ vLLM service ready")
                    return self.port, self.served_model_name
            except:
                pass
            
            try:
                response = requests.get(f"http://127.0.0.1:{self.port}/v1/models", timeout=2)
                if response.status_code == 200:
                    print("✓ vLLM service ready")
                    return self.port, self.served_model_name
            except:
                pass
            
            # Print progress every 30 seconds
            if i % 15 == 0 and i > 0:
                print(f"  Still waiting... ({i*2}s elapsed)")
                
                # Check log for any errors
                if vllm_log.exists():
                    with open(vllm_log) as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if 'error' in last_line.lower() or 'exception' in last_line.lower():
                                print(f"  Recent log line: {last_line}")
            
            time.sleep(2)
        
        # Timeout, output logs
        print("Error: vLLM not ready within 300 seconds")
        if vllm_log.exists():
            print("\nFull vLLM log:")
            with open(vllm_log) as f:
                print(f.read())
        
        # Kill the process
        self.stop()
        raise RuntimeError("vLLM service failed to start")
    
    def stop(self):
        """Stop vLLM service"""
        if self.process is not None:
            print(f"Stopping vLLM service (PID={self.process.pid})")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except:
                self.process.kill()
            self.process = None
            self.port = None
            self.served_model_name = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
