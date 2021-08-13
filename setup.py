#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='advchain',
    version='0.0.0',
    description='Adversarial data augmentation with chained transformation',
    author='Chen Chen',
    author_email='chen.chen15@imperial.ac.uk',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/cherise215/advchain',
    install_requires=['torch'],
    packages=find_packages(),
)
