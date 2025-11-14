"""Utility functions for notebook setup and dependency management"""

import sys
import subprocess
import importlib.util


def check_dependency(package_name: str) -> bool:
    """Check if a package is installed
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except:
        return False


def install_package(package_name: str, install_name: str = None, upgrade: bool = False) -> bool:
    """Install a package using pip
    
    Args:
        package_name: Name of the package (for checking)
        install_name: Name to use for pip install (if different)
        upgrade: Whether to upgrade if already installed
        
    Returns:
        True if successful, False otherwise
    """
    if install_name is None:
        install_name = package_name
    
    print(f"Installing {package_name}...")
    try:
        cmd = [sys.executable, '-m', 'pip', 'install']
        if upgrade:
            cmd.append('--upgrade')
        cmd.append(install_name)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False


def check_gpu_availability() -> bool:
    """Check GPU availability
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        print("✓ NVIDIA GPU detected")
        return True
    except:
        print("⚠ No NVIDIA GPU detected - vLLM will use CPU mode")
        return False


def install_vllm() -> bool:
    """Install vLLM with proper CUDA support
    
    Returns:
        True if successful, False otherwise
    """
    print("Installing vLLM...")
    return install_package("vllm")


def install_evalscope() -> bool:
    """Install evalscope with dependencies
    
    Returns:
        True if successful, False otherwise
    """
    print("Installing evalscope...")
    
    # Install core dependencies first
    core_deps = ['requests', 'tqdm', 'fsspec', 'dill', 'multiprocess', 'datasets']
    for dep in core_deps:
        if not check_dependency(dep):
            install_package(dep)
    
    # Install modelscope
    if not check_dependency('modelscope'):
        install_package('modelscope')
    
    # Install evalscope
    return install_package('evalscope')


def setup_dependencies() -> bool:
    """Setup all required dependencies
    
    Returns:
        True if all dependencies installed successfully
    """
    print("=" * 60)
    print("Setting up dependencies...")
    print("=" * 60)
    
    # Check GPU
    check_gpu_availability()
    
    # Check and install torch
    if not check_dependency('torch'):
        print("Installing PyTorch...")
        install_package('torch')
    else:
        print("✓ torch already installed")
    
    # Install vLLM
    if not check_dependency('vllm'):
        if not install_vllm():
            print("⚠ Failed to install vLLM")
            return False
    else:
        print("✓ vLLM already installed")
    
    # Install evalscope
    if not check_dependency('evalscope'):
        if not install_evalscope():
            print("⚠ Failed to install evalscope")
            return False
    else:
        print("✓ evalscope already installed")
    
    # Verify critical packages
    print("\nVerifying installations...")
    critical_packages = ['torch', 'vllm', 'evalscope', 'modelscope']
    all_ok = True
    
    for package in critical_packages:
        if check_dependency(package):
            print(f"✓ {package}: installed")
        else:
            print(f"✗ {package}: not available")
            all_ok = False
    
    if all_ok:
        print("\n" + "=" * 60)
        print("✓ All dependencies installed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠ Some dependencies failed to install")
        print("=" * 60)
    
    return all_ok


def download_toolkit_from_github(repo_url: str = None, branch: str = "main") -> bool:
    """Download evalscope_toolkit from GitHub repository
    
    Args:
        repo_url: GitHub repository URL (if None, assumes already downloaded)
        branch: Branch to download from
        
    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path
    
    # Check if toolkit already exists
    toolkit_path = Path.cwd() / "evalscope_toolkit"
    if toolkit_path.exists():
        print("✓ evalscope_toolkit already exists")
        return True
    
    if repo_url is None:
        print("⚠ evalscope_toolkit not found and no repo URL provided")
        return False
    
    print(f"Downloading evalscope_toolkit from {repo_url}...")
    
    # Use git to clone or download
    try:
        # Try git clone first
        subprocess.run(
            ['git', 'clone', '-b', branch, '--depth', '1', repo_url, 'temp_repo'],
            check=True,
            capture_output=True
        )
        
        # Move evalscope_toolkit to current directory
        import shutil
        shutil.move('temp_repo/evalscope_toolkit', 'evalscope_toolkit')
        shutil.rmtree('temp_repo')
        
        print("✓ evalscope_toolkit downloaded successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("⚠ Git clone failed, trying alternative method...")
        
        # Try wget/curl download
        try:
            import urllib.request
            import zipfile
            
            # Download zip file
            zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
            print(f"Downloading {zip_url}...")
            
            urllib.request.urlretrieve(zip_url, "temp.zip")
            
            # Extract
            with zipfile.ZipFile("temp.zip", 'r') as zip_ref:
                zip_ref.extractall("temp_extract")
            
            # Move evalscope_toolkit
            import shutil
            import os
            
            # Find the extracted directory
            extracted_dir = None
            for item in os.listdir("temp_extract"):
                if os.path.isdir(os.path.join("temp_extract", item)):
                    extracted_dir = os.path.join("temp_extract", item)
                    break
            
            if extracted_dir:
                toolkit_src = os.path.join(extracted_dir, "evalscope_toolkit")
                if os.path.exists(toolkit_src):
                    shutil.move(toolkit_src, "evalscope_toolkit")
                    shutil.rmtree("temp_extract")
                    os.remove("temp.zip")
                    print("✓ evalscope_toolkit downloaded successfully")
                    return True
            
            print("⚠ Could not find evalscope_toolkit in downloaded archive")
            return False
            
        except Exception as e:
            print(f"⚠ Download failed: {e}")
            return False
    
    except Exception as e:
        print(f"⚠ Failed to download toolkit: {e}")
        return False
