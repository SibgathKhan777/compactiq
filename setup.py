"""
setup.py — Install PyCompat as a pip package
=============================================

Install locally:
    pip install -e .

Install from GitHub:
    pip install git+https://github.com/SibgathKhan777/bank.git

Then use anywhere:
    from pycompat_model import PyCompatModel
    model = PyCompatModel.load("/path/to/model")
"""

from setuptools import setup, find_packages

setup(
    name="pycompat-model",
    version="1.0.0",
    description="AI-powered Python package compatibility predictor",
    long_description=open("model/README.md").read() if __import__("os").path.exists("model/README.md") else "",
    long_description_content_type="text/markdown",
    author="SibgathKhan777",
    url="https://github.com/SibgathKhan777/bank",
    py_modules=["pycompat_model"],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "api": ["flask>=3.0.0", "flask-cors>=4.0.0"],
        "huggingface": ["huggingface_hub>=0.20.0"],
    },
    entry_points={
        "console_scripts": [
            "pycompat=pycompat_model:main_cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
