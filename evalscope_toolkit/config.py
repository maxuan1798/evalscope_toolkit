"""Configuration management for model evaluation"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    
    # Model configuration
    models: List[str] = field(default_factory=lambda: ["unsloth/Llama-3.2-3B-Instruct"])
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ["gsm8k"])
    
    # GPU configuration
    gpus: str = "0"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6
    max_num_seqs: int = 64
    
    # Evaluation parameters
    eval_batch_size: int = 32
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    eval_n: int = 1
    seed: int = 42
    system_prompt: str = ""
    
    # Directory configuration
    workspace: Path = field(default_factory=lambda: Path.cwd())
    data_root: Optional[Path] = None
    log_root: Optional[Path] = None
    chat_template: Optional[Path] = None
    
    # Service configuration
    base_port: int = 8800
    
    # User configuration
    user_id: str = field(default_factory=lambda: os.environ.get('USER', 'user'))
    
    def __post_init__(self):
        """Initialize derived paths"""
        self.workspace = Path(self.workspace)
        
        if self.data_root is None:
            self.data_root = self.workspace / "data"
        else:
            self.data_root = Path(self.data_root)
            
        if self.log_root is None:
            self.log_root = self.workspace / "log"
        else:
            self.log_root = Path(self.log_root)
            
        if self.chat_template is None:
            self.chat_template = self.workspace / "chat_template.jinja"
        else:
            self.chat_template = Path(self.chat_template)
        
        # Create directories
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ['HF_DATASETS_CACHE'] = str(self.data_root / ".hf_cache")
        os.environ['MODELSCOPE_CACHE'] = str(self.data_root / ".modelscope_cache")
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        os.environ['VLLM_HOST_IP'] = '127.0.0.1'
    
    def get_dataset_args(self):
        """Generate dataset configuration arguments"""
        return {
            "gsm8k": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "competition_math": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt,
                "train_split": "train",
                "eval_split": "test"
            },
            "humaneval": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "math_500": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "drop": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "hellaswag": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "eval_split": "val",
                "system_prompt": self.system_prompt
            },
            "mmlu": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "arc": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "truthfulqa": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            },
            "winogrande": {
                "few_shot_num": 0,
                "filters": {"remove_until": "</think>"},
                "system_prompt": self.system_prompt
            }
        }
