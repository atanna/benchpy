#! /usr/bin/env python

from setuptools import setup
from Cython.Build import cythonize

DISTNAME = "benchpy"
DESCRIPTION = __doc__
LONG_DESCRIPTION = open("README.rst").read()
MAINTAINER = "Ann Atamanova"
MAINTAINER_EMAIL = "anne.atamanova@gmail.com"
LICENSE = "MIT"

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License"
    "Intended Audience :: Developers",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
]


import benchpy

VERSION = benchpy.__version__


setup_options = dict(
    name="benchpy",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url="https://github.com/atanna/benchpy",
    packages=["benchpy"],
    classifiers=CLASSIFIERS,
    ext_modules=cythonize("benchpy/_speedups.pyx"),
    install_requires=["numpy", "scipy", "prettytable"],
    extras_require={
        "docs": ["Sphinx", "numpydoc"]
    }
)


if __name__ == "__main__":
    setup(**setup_options)
