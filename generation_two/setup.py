"""
Setup script for Generation Two
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="generation-two",
    version="1.0.0",
    description="WorldQuant Brain Alpha Mining System - Generation Two",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WorldQuant Miner",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/worldquant-miner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "generation-two=generation_two.gui.run_gui:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
