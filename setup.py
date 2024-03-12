from setuptools import setup, find_packages
import os
from pathlib import Path

location = Path(os.path.abspath(os.path.dirname(__file__)))
# with open(os.path.join(location, "requirements.txt"), "r") as f:
#    requirements = f.read().splitlines()

# For some reason tox fails during the above file read
REQUIREMENTS = ["absl-py>=1.0.0,<2.0.0", "GitPython>=3.0.0,<4.0.0"]

with open(location / "Readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyr_benchmark_wrangling",
    version="0.0.11",
    author="PyRepair Team",
    author_email="nikhil.parasaram.19@ucl.ac.uk",
    description="A package to facilitate data-wrangling for APR tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PyRepair/pyr_benchmark_wrangling",
    packages=find_packages(include=["BugsInPy*", "diff_utils*", "static_library*"]),
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "lmeasures=BugsInPy.lmeasures:main",
            "bgp=BugsInPy.bgp:main",
            "sample_bip=BugsInPy.sample_bip:main",
            "run_custom_patch=BugsInPy.run_custom_patch:main",
            "diff_utils=diff_utils.diff_utils:main",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",  # Updated to Apache License
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="Apache License 2.0",
)
