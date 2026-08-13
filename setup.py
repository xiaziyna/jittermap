from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="jittermap",
    version="0.1.0",
    description="Stellar surface mapping from astrometric jitter and photometry",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Jamila Taaki",
    author_email="xiaziyna@gmail.com",
    url="https://github.com/xiaziyna/jittermap",
    project_urls={
        "Documentation": "https://jittermap.readthedocs.io",
        "Paper": "https://doi.org/10.3847/1538-4357/ae66f7",
    },
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "jittermap": [
            "data/wigner/*.npz",
            "data/kernels/*.npz",
        ],
    },
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
