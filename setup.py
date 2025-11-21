from setuptools import setup, find_packages

setup(
    name="embedding-model",
    version="0.1.0",
    description="Industrial Defect Embedding Model with SSL and SupCon",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.8",
)

