#!/usr/bin/env python
from pathlib import Path
from setuptools import find_packages, setup

here = Path(__file__).parent

with open(here / "README.md") as f:
    readme = f.read()
with open(here / "requirements.txt") as f:
    reqs = f.read()

setup(
    name="igms",
    version="0.0.1",
    description="Implementation of MMD and related implicit generative models for PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/djsutherland/igms",
    author="DJ Sutherland",
    author_email="mail@djsutherland.ml",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=reqs,
    classifiers=["License :: OSI Approved :: Apache Software License"],
)
