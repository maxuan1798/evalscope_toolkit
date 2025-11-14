"""Dataset download and management using ModelScope"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional


class DatasetManager:
    """Manages dataset download and caching using ModelScope"""
    
    # Evalscope official dataset configurations (ModelScope)
    DATASET_CONFIGS = {
        'gsm8k': {
            'ms_name': 'AI-ModelScope/gsm8k',
            'subset_name': 'main',
            'split': 'test',
            'description': 'Grade School Math 8K - Math reasoning problems',
        },
        'competition_math': {
            'ms_name': 'AI-ModelScope/competition_math',
            'subset_name': None,
            'split': 'test',
            'description': 'Competition-level mathematics problems',
        },
        'humaneval': {
            'ms_name': 'AI-ModelScope/humaneval',
            'subset_name': None,
            'split': 'test',
            'description': 'HumanEval - Code generation evaluation',
        },
        'math_500': {
            'ms_name': 'AI-ModelScope/math_500',
            'subset_name': None,
            'split': 'test',
            'description': 'Math 500 dataset',
        },
        'drop': {
            'ms_name': 'AI-ModelScope/drop',
            'subset_name': None,
            'split': 'validation',
            'description': 'Reading comprehension dataset',
        },
        'hellaswag': {
            'ms_name': 'AI-ModelScope/hellaswag',
            'subset_name': None,
            'split': 'validation',
            'description': 'Commonsense reasoning',
        },
        'mmlu': {
            'ms_name': 'AI-ModelScope/mmlu',
            'subset_name': 'all',
            'split': 'test',
            'description': 'Massive Multitask Language Understanding',
        },
        'arc': {
            'ms_name': 'AI-ModelScope/ai2_arc',
            'subset_name': 'ARC-Challenge',
            'split': 'test',
            'description': 'AI2 Reasoning Challenge',
        },
        'truthfulqa': {
            'ms_name': 'AI-ModelScope/truthful_qa',
            'subset_name': 'generation',
            'split': 'validation',
            'description': 'TruthfulQA - Truthfulness evaluation',
        },
        'winogrande': {
            'ms_name': 'AI-ModelScope/winogrande',
            'subset_name': 'winogrande_xl',
            'split': 'validation',
            'description': 'Winogrande - Commonsense reasoning',
        }
    }
    
    def __init__(self, cache_dir: Path):
        """Initialize dataset manager
        
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = Path(cache_dir)
        self.ms_cache_dir = self.cache_dir / ".modelscope_cache"
        self.ms_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a single dataset
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.DATASET_CONFIGS:
            print(f"⚠ Unknown dataset: {dataset_name}")
            print(f"  Available datasets: {', '.join(self.DATASET_CONFIGS.keys())}")
            return False
        
        config = self.DATASET_CONFIGS[dataset_name]
        print(f"\n📥 Downloading {dataset_name}: {config['description']}")
        print(f"  ModelScope: {config['ms_name']}")
        if config['subset_name']:
            print(f"  Subset: {config['subset_name']}")
        print(f"  Split: {config['split']}")
        
        try:
            # Import ModelScope
            try:
                from modelscope.msdatasets import MsDataset
            except ImportError:
                print("Installing modelscope package...")
                import subprocess
                import sys
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'modelscope'], check=True)
                from modelscope.msdatasets import MsDataset
            
            # Check if dataset is already cached
            cache_info_file = self.ms_cache_dir / f"{dataset_name}_info.json"
            if cache_info_file.exists():
                with open(cache_info_file) as f:
                    cache_info = json.load(f)
                print(f"  Found cached dataset (downloaded: {cache_info.get('download_time', 'unknown')})")
                print(f"  Samples: {cache_info.get('samples', 'unknown')}")
                print(f"  Verifying cache...")
            
            # Download dataset using ModelScope MsDataset
            load_kwargs = {
                'split': config['split'],
                'cache_dir': str(self.ms_cache_dir)
            }
            
            if config['subset_name']:
                load_kwargs['subset_name'] = config['subset_name']
            
            print(f"  Loading from ModelScope...")
            dataset = MsDataset.load(config['ms_name'], **load_kwargs)
            
            # Get dataset size
            try:
                dataset_size = len(dataset)
            except:
                dataset_size = sum(1 for _ in dataset)
            
            # Save cache info
            cache_info = {
                'dataset_name': dataset_name,
                'ms_name': config['ms_name'],
                'subset_name': config['subset_name'],
                'split': config['split'],
                'samples': dataset_size,
                'download_time': datetime.now().isoformat(),
            }
            with open(cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2)
            
            print(f"✓ {dataset_name} downloaded successfully")
            print(f"  Samples: {dataset_size}")
            print(f"  Cache info: {cache_info_file}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_datasets(self, dataset_names: List[str]) -> None:
        """Download multiple datasets
        
        Args:
            dataset_names: List of dataset names to download
        """
        print("=" * 60)
        print("Dataset Download and Preparation (ModelScope - evalscope official)")
        print("=" * 60)
        print(f"Datasets to prepare: {dataset_names}")
        print(f"Cache directory: {self.ms_cache_dir}")
        
        for dataset_name in dataset_names:
            self.download_dataset(dataset_name.strip())
        
        print("\n" + "=" * 60)
        print("Dataset preparation completed!")
        print("=" * 60)
    
    def verify_datasets(self, dataset_names: List[str]) -> dict:
        """Verify that datasets are available in cache
        
        Args:
            dataset_names: List of dataset names to verify
            
        Returns:
            Dictionary with 'available' and 'missing' lists
        """
        print("\n" + "=" * 60)
        print("Dataset Cache Verification")
        print("=" * 60)
        
        available_datasets = []
        missing_datasets = []
        
        for dataset_name in dataset_names:
            dataset_name = dataset_name.strip()
            cache_info_file = self.ms_cache_dir / f"{dataset_name}_info.json"
            
            if cache_info_file.exists():
                with open(cache_info_file) as f:
                    cache_info = json.load(f)
                available_datasets.append({
                    'name': dataset_name,
                    'samples': cache_info.get('samples', 'unknown'),
                    'download_time': cache_info.get('download_time', 'unknown'),
                    'ms_name': cache_info.get('ms_name', 'unknown')
                })
            else:
                missing_datasets.append(dataset_name)
        
        if available_datasets:
            print("\n✓ Available datasets:")
            for dataset in available_datasets:
                print(f"  - {dataset['name']}: {dataset['samples']} samples")
                print(f"    Source: {dataset['ms_name']}")
                print(f"    Downloaded: {dataset['download_time']}")
        
        if missing_datasets:
            print(f"\n⚠ Missing datasets: {', '.join(missing_datasets)}")
            print("  These datasets need to be downloaded")
        else:
            print("\n✓ All required datasets are cached and ready!")
        
        print("\n" + "=" * 60)
        print("Cache verification completed!")
        print("=" * 60)
        
        return {
            'available': available_datasets,
            'missing': missing_datasets
        }
    
    @classmethod
    def list_supported_datasets(cls) -> List[str]:
        """List all supported datasets
        
        Returns:
            List of supported dataset names
        """
        return list(cls.DATASET_CONFIGS.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Optional[dict]:
        """Get information about a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset configuration dict or None if not found
        """
        return cls.DATASET_CONFIGS.get(dataset_name)
