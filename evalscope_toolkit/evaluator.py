"""Evaluation orchestration"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

from .config import EvalConfig
from .dataset_manager import DatasetManager
from .vllm_service import VLLMService


class Evaluator:
    """Orchestrates the complete evaluation pipeline"""
    
    def __init__(self, config: EvalConfig):
        """Initialize evaluator
        
        Args:
            config: EvalConfig instance
        """
        self.config = config
        self.dataset_manager = DatasetManager(config.data_root)
        
        # Generate instance ID
        self.instance_id = f"{datetime.now().strftime('%m%d%H%M')}{os.getpid()}"
        
        # Create log directory
        self.base_log_dir = config.log_root / f"outputs_{config.user_id}_{self.instance_id}"
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.overall_time_log = self.base_log_dir / "overall_evaluation_times.log"
        
        print("=" * 60)
        print("Evaluation Instance Configuration:")
        print("=" * 60)
        print(f"USER_ID:      {config.user_id}")
        print(f"INSTANCE_ID:  {self.instance_id}")
        print(f"BASE_PORT:    {config.base_port}")
        print(f"GPUS:         {config.gpus}")
        print(f"TP_SIZE:      {config.tensor_parallel_size}")
        print(f"DATA_ROOT:    {config.data_root}")
        print(f"LOG_ROOT:     {config.log_root}")
        print(f"BASE_LOG_DIR: {self.base_log_dir}")
        print("=" * 60)
    
    def prepare_datasets(self):
        """Download and verify datasets"""
        print("\n" + "=" * 60)
        print("Preparing datasets...")
        print("=" * 60)
        
        # Download datasets
        self.dataset_manager.download_datasets(self.config.datasets)
        
        # Verify datasets
        verification = self.dataset_manager.verify_datasets(self.config.datasets)
        
        if verification['missing']:
            raise RuntimeError(f"Missing datasets: {verification['missing']}")
        
        print("\n✓ All datasets ready")
    
    def run_evaluation_for_model(self, model_ref: str) -> dict:
        """Run evaluation for a specific model
        
        Args:
            model_ref: Model path or HuggingFace repo ID
            
        Returns:
            Dictionary with evaluation results and timing info
        """
        # Get model name for logging
        name = Path(model_ref).name if VLLMService.is_local_path(model_ref) else model_ref
        work_dir = self.base_log_dir / name
        work_dir.mkdir(parents=True, exist_ok=True)
        time_log = work_dir / "evaluation_times.log"
        
        results = {
            'model': name,
            'datasets': {},
            'total_duration': 0
        }
        
        # Start vLLM service
        vllm_service = VLLMService(self.config)
        
        try:
            port, served_model_name = vllm_service.start(model_ref, self.base_log_dir)
            
            # Get dataset configuration
            dataset_args = self.config.get_dataset_args()
            dataset_args_json = json.dumps(dataset_args)
            
            # Iterate through datasets
            for dataset in self.config.datasets:
                dataset = dataset.strip()
                t0 = time.time()
                
                print(f"\n[{name}] Starting evaluation for {dataset}")
                
                # Build evalscope command
                gen_config = json.dumps({
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_new_tokens,
                    "n": self.config.eval_n,
                    "seed": self.config.seed
                })
                
                cmd = [
                    sys.executable, '-m', 'evalscope.cli', 'eval',
                    '--model', served_model_name,
                    '--generation-config', gen_config,
                    '--api-url', f'http://127.0.0.1:{port}/v1/chat/completions',
                    '--api-key', 'EMPTY',
                    '--eval-type', 'openai_api',
                    '--work-dir', str(work_dir / dataset),
                    '--datasets', dataset,
                    '--dataset-args', dataset_args_json,
                    '--dataset-dir', str(self.config.data_root),
                    '--eval-batch-size', str(self.config.eval_batch_size),
                    '--stream'
                ]
                
                # Execute evaluation
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=False, text=True)
                
                t1 = time.time()
                dur = int(t1 - t0)
                
                print(f"[{name}] Completed {dataset}, duration={dur} seconds")
                
                # Store results
                results['datasets'][dataset] = {
                    'duration': dur,
                    'start_time': datetime.fromtimestamp(t0).isoformat(),
                    'end_time': datetime.fromtimestamp(t1).isoformat(),
                    'work_dir': str(work_dir / dataset)
                }
                results['total_duration'] += dur
                
                # Log time
                with open(time_log, 'a') as f:
                    f.write(f"Model: {name}, Dataset: {dataset}, Duration: {dur}s, "
                           f"Start: {datetime.fromtimestamp(t0)}, End: {datetime.fromtimestamp(t1)}\n")
                
                with open(self.overall_time_log, 'a') as f:
                    f.write(f"[{self.config.user_id}] Model: {name}, Dataset: {dataset}, Duration: {dur}s\n")
                
                time.sleep(2)
        
        finally:
            # Clean up vLLM service
            vllm_service.stop()
        
        return results
    
    def run(self) -> dict:
        """Run complete evaluation pipeline
        
        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "=" * 60)
        print("Starting Evaluation Pipeline")
        print("=" * 60)
        print(f"Models: {self.config.models}")
        print(f"Datasets: {self.config.datasets}")
        print("=" * 60)
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Run evaluation for each model
        all_results = {
            'instance_id': self.instance_id,
            'user_id': self.config.user_id,
            'models': {},
            'log_dir': str(self.base_log_dir)
        }
        
        for idx, model in enumerate(self.config.models):
            print(f"\n{'='*60}")
            print(f"Evaluating model {idx+1}/{len(self.config.models)}: {model}")
            print(f"{'='*60}\n")
            
            try:
                results = self.run_evaluation_for_model(model)
                all_results['models'][model] = results
            except Exception as e:
                print(f"Error: Model {model} evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                all_results['models'][model] = {
                    'error': str(e),
                    'model': model
                }
        
        # Save summary
        summary_file = self.base_log_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Evaluation Pipeline Completed!")
        print(f"{'='*60}")
        print(f"Results saved to: {self.base_log_dir}")
        print(f"Summary: {summary_file}")
        print(f"Overall log: {self.overall_time_log}")
        
        return all_results
    
    def show_results(self):
        """Display evaluation results"""
        if self.overall_time_log.exists():
            print("=" * 60)
            print("Overall Evaluation Time Statistics:")
            print("=" * 60)
            with open(self.overall_time_log) as f:
                print(f.read())
        else:
            print("No evaluation logs generated yet")
        
        # Show directory structure
        print("\nEvaluation Result Directory Structure:")
        print("=" * 60)
        
        if self.base_log_dir.exists():
            for root, dirs, files in os.walk(self.base_log_dir):
                level = root.replace(str(self.base_log_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f'{subindent}{file}')
