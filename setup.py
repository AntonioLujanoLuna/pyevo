from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysnes",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python implementation of Separable Natural Evolution Strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pysnes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "examples": [
            "pillow>=8.0.0",
            "matplotlib>=3.3.0",
            "imageio>=2.9.0",
            "tqdm>=4.65.0",
        ],
        "video": [
            "imageio[ffmpeg]",
        ],
        "gpu": [
            "cupy>=12.0.0",
        ],
        "all": [
            "pillow>=8.0.0",
            "matplotlib>=3.3.0",
            "imageio>=2.9.0",
            "tqdm>=4.65.0",
            "imageio[ffmpeg]",
        ],
    },
)