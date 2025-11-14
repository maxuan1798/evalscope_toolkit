"""Command-line interface for Evalscope Toolkit"""

import sys
import argparse
from pathlib import Path

from .config import EvalConfig
from .evaluator import Evaluator
from .dataset_manager import DatasetManager


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Evalscope Toolkit - Simplified model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a model on gsm8k dataset
  evalscope-toolkit --models "unsloth/Llama-3.2-3B-Instruct" --datasets gsm8k
  
  # Evaluate multiple models on multiple datasets
  evalscope-toolkit --models "model1,model2" --datasets "gsm8k,humaneval"
  
  # Use multiple GPUs
  evalscope-toolkit --models "large-model" --datasets gsm8k --gpus "0,1" --tp-size 2
  
  # List supported datasets
  evalscope-toolkit --list-datasets
        """
    )
    
    # Model and dataset configuration
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models (local paths or HuggingFace IDs)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to evaluate"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (default: '0')"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.6,
        help="GPU memory utilization (default: 0.6)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)"
    )
    
    # Directory configuration
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace directory (default: current directory)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Data cache directory (default: workspace/data)"
    )
    parser.add_argument(
        "--log-root",
        type=str,
        help="Log directory (default: workspace/log)"
    )
    
    # Utility commands
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all supported datasets and exit"
    )
    parser.add_argument(
        "--download-datasets",
        type=str,
        help="Download specified datasets (comma-separated) and exit"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_datasets:
        print("Supported datasets:")
        print("=" * 60)
        for dataset_name in DatasetManager.list_supported_datasets():
            info = DatasetManager.get_dataset_info(dataset_name)
            print(f"\n📊 {dataset_name}")
            print(f"   {info['description']}")
            print(f"   ModelScope: {info['ms_name']}")
            if info['subset_name']:
                print(f"   Subset: {info['subset_name']}")
            print(f"   Split: {info['split']}")
        return 0
    
    if args.download_datasets:
        datasets = [d.strip() for d in args.download_datasets.split(',')]
        workspace = Path(args.workspace)
        data_root = Path(args.data_root) if args.data_root else workspace / "data"
        
        dm = DatasetManager(data_root)
        dm.download_datasets(datasets)
        return 0
    
    # Validate required arguments
    if not args.models or not args.datasets:
        parser.error("--models and --datasets are required")
    
    # Parse models and datasets
    models = [m.strip() for m in args.models.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Create configuration
    config_kwargs = {
        'models': models,
        'datasets': datasets,
        'gpus': args.gpus,
        'tensor_parallel_size': args.tp_size,
        'gpu_memory_utilization': args.gpu_mem_util,
        'eval_batch_size': args.eval_batch_size,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'workspace': Path(args.workspace),
    }
    
    if args.data_root:
        config_kwargs['data_root'] = Path(args.data_root)
    if args.log_root:
        config_kwargs['log_root'] = Path(args.log_root)
    
    config = EvalConfig(**config_kwargs)
    
    # Run evaluation
    try:
        evaluator = Evaluator(config)
        results = evaluator.run()
        
        print("\n" + "=" * 60)
        print("✓ Evaluation completed successfully!")
        print("=" * 60)
        print(f"Results saved to: {evaluator.base_log_dir}")
        
        return 0
    
    except Exception as e:
        print(f"\n⚠ Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
