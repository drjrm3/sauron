#!/usr/bin/env python3

from setuptools import setup
import numpy
setup(
    name="sauron",
    version="1.1.2",
    description="The protein ring finder.",
    author="J. Robert Michael, PhD",
    author_email="jrobert.michael@stjude.org",
    url="https://github.com/drjrm3/sauron.git",
    packages=["sauron"],
    license="Apache License 2.0",
    test_suite="tests",
    install_requires=[
        "scipy", "numpy", "matplotlib", "tiffile"
    ],
    python_requires=">=3.4",
    scripts=["scripts/saruman", "scripts/sauron"]
)
