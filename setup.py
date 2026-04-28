"""
License Plate Detection System - Setup Script
"""

from setuptools import setup, find_packages

setup(
    name='license-plate-detection',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
)
