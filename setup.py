#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Mehrad Ansari",
    author_email="mehrad.ans@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
    ],
    description="Chemist AI Agent for Inverse Design of Materials with Natural Language Prompts",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    dependency_links=[
        "https://data.dgl.ai/wheels/torch-2.3/cu121/dgl-2.2.1+cu121-cp311-cp311-manylinux1_x86_64.whl"
    ],
    license="Apache-2.0",
    include_package_data=True,
    keywords="dziner",
    name="dziner",
    packages=find_packages(include=["dziner", "dziner.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/mehradnas92/dziner",
    version="0.1",
    zip_safe=False,
)
