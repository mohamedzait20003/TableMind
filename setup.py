"""
Setup script for the DoT model package.

This script allows the package to be installed in development mode
and provides metadata for the project.
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dot-model",
    version="0.1.0",
    author="ECE-570 Project Team",
    author_email="student@purdue.edu",
    description="Differentiable Optimized Transformer for Table Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/dot-model",
    package_dir={"": "src"},
    packages=["core", "data", "training"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dot-train=scripts.train:main",
            "dot-eval=scripts.evaluate:main",
        ],
    },
)