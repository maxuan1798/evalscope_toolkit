from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evalscope-toolkit",
    version="1.0.0",
    author="InfiMax",
    author_email="maxuan1798@gmail.com",
    description="A simplified toolkit for model evaluation using vLLM and Evalscope",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxuan1798/evalscope-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "torch>=2.0.0",
        "vllm>=0.6.0",
        "evalscope>=0.4.0",
        "modelscope>=1.9.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evalscope-toolkit=evalscope_toolkit.cli:main",
        ],
    },
)
