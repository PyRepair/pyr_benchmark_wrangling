[![PyPI](https://img.shields.io/pypi/v/pyr_benchmark_wrangling)](https://pypi.python.org/pypi/pyr_benchmark_wrangling)
[![Run Tests and Deploy](https://github.com/PyRepair/pyr_benchmark_wrangling/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/PyRepair/pyr_benchmark_wrangling/actions/workflows/test.yml)
[![Downloads](https://static.pepy.tech/badge/pyr_benchmark_wrangling)](https://pepy.tech/project/pyr_benchmark_wrangling)

# Introduction
Pyrepair Benchmark Wrangling is a Python package designed to facilitate and streamline the process of the data-wrangling benchmarks for Automated Program Repair (APR) tools. This package provides an easy-to-use command-line interface to interact with two main components: 
- `BugsInPy` - a package to run BugsInPy benchmark (https://dl.acm.org/doi/abs/10.1145/3368089.3417943)
- `diff_utils` - a set of utilities for handling diffs

## Features

- `lmeasures`: A command-line tool to compute and report various metrics and measures for the benchmarks.
- `bgp`: A command-line interface to interact with the BugsInPy benchmark suite.
- `sample_bip`: A utility to sample bugs from the BugsInPy benchmark suite.
- `run_custom_patch`: A tool to apply custom patches to the bugs in the BugsInPy dataset.
- `diff_utils`: A command-line utility to handle diff files and changes.

# Installation
You can either directly install Pyr Benchmark Wrangling via pip, or Docker.
After cloning the repository, switch to pyr_benchmark_wrangling:

```bash
cd pyr_benchmark_wrangling
```

## Direct Installation


### System Requirements:

Before using the Pyr Benchmark Wrangling, make sure your system meets the following system requirements:

- Python 3.7 and Python 3.8
- Development packages for Python 3.7 and Python 3.8
- libffi7 library

On a Debian-based system, you can install these requirements using `apt-get`:

```bash
sudo apt-get install python3.7 python3.7-dev python3.8 python3.8-dev libffi7
```

### Pip Command:


You can install Pyr Benchmark Wrangling by running the following command:

```bash
pip install .
```

## Using Docker

Pyr Benchmark Wrangling's Docker Space Requirements:
- Lite image: 2.8 GB
- Full image: 20 GB

The difference between `lite` and `full` image is that the virtual environments are lazily constructed in `lite`, and downloaded in `full`.



To build the lite image (2.8 GB), use the following command:

```bash
docker build --target lite -t pyr:lite .
```

This will execute all instructions in the Dockerfile up until the lite stage is complete.
The lite image automatically runs `update_bug_records` and `clone`'s all repositories

To build the full image (20 GB), use the following command:

```bash

docker build --target full -t pyr:full .
```
This will execute all instructions in the Dockerfile.
The full image automatically runs `update_bug_records`, `clone`'s all repositories, installs all required environments.



# BugsInPy CLI
The BugsInPy CLI is a command-line tool designed for interacting with and running Python bugs from the BugsInPy dataset. This script streamlines the process of setting up bug repositories, cloning specific bugs or repositories, preparing the environment, running tests, and more. Below, you'll find an overview of the available commands and their functionalities:
This tool requires Python3.10 and above

## Usage

### `setup` Command

The `setup` command is used to set up the BugsInPy repository. This step is essential before working with any bugs. It clones the BugsInPy repository to your local system.

```bash
bgp setup
```

### `clone` Command

The `clone` command allows you to clone specific bugs or repositories based on your requirements. You can specify the bugs to clone using the `--bug_list` flag or repositories using the `--repo_list` flag.

**Example:**
```bash
bgp clone --bugids repo1:id1,repo2:id2,...,repo3:id3
```
### `checkout_buggy` and `checkout_fixed` Commands

These commands are used to checkout the buggy or fixed version of a specific bug repository. You provide the bug ID to identify the repository.

- To checkout the buggy version:
```bash
bgp checkout_buggy --bugids repo:<bug_id>
```
- To checkout the fixed version:
```bash
bgp checkout_fixed --bugids repo:<bug_id>
```


### `extract_features` Command

The `extract_features` command extracts features of a specific bug. 

**Example:**
```bash
bgp extract_features --bugids repo:<bug_id>
```

### `prep` Command

The `prep` command prepares the environment for a specific bug. It installs the required dependencies and performs sanity checks to ensure the bug can be tested successfully.
The commands `setup` and `clone` should be run before `prep`

**Example:**
```bash
bgp prep --bugids repo:<bug_id>
```

### `run_test` Command

The `run_test` command runs the tests for a specific list of bugs. It executes the test commands associated with the bug.
The commands `setup`, `clone` and `prep` should be run before `run_test`.

```bash
bgp run_test --bugids repo:<bug_id>
```


### `delete_bug_repo` Command

The `delete_bug_repo` command deletes a specific bug repository from your local system.

**Example:**
```bash
bgp delete_bug_repo <bug_id>
```
## Additional Notes

- The CLI provides options to control the verbosity of the prep step and set the log level.
- Mutually exclusive flags such as `--bug_list`, `--repo_list` are available for listing bugs/repos on which the commands should be run.
- You can adjust the timeout for various system calls using the `--timeout` flag.

## Unsupported repos
The following repositories are un-supported:
- Spacy: Due to the requirement of python version < "3.4"


# diff_utils
`diff_utils` is a Python module for analyzing and extracting data from unified diff outputs generated by tools such as Git. The module provides functionalities to compute localization measures on single file diffs, across multiple file diffs, and extract the modified line numbers and file names from diffs.

## Features

- Compute hunk statistics such as count, gaps, and spans from a single file diff.
- Aggregate hunk information across multiple file diffs to calculate comprehensive statistics.
- Extract modified files and their respective line changes from a unified diff.
- Write the extracted data to CSV files for further analysis.


## Usage

The `diff_utils` module provides a set of functions that can be used independently or through a command-line interface.

## Command-line Interface

The module can be run as a script to perform actions based on the arguments provided:

-    --measure: Accepts comma-delimited list of diff files to measure localization metrics.
-    --locations: Accepts comma-delimited list of diff files from which to extract location sets.
-    --quiet: Quiet mode, which suppresses the standard output.
-    --output: Specifies the output CSV file name.


Example Usage:

```bash
> diff_utils --measure "diff_file1.txt,diff_file2.txt" --output "measures_output.csv"
> diff_utils --locations "diff_file1.txt,diff_file2.txt" --output "locations_output.csv"
```
## Module Functions

You can also use the functions provided by `diff_utils` in a Python script:

```python
from diffutils import measure_localisation_diff_file, locations_from_diff_file

# Measure localization metrics for a given diff file
metrics = measure_localisation_diff_file("diff_file.txt")

# Extract locations from a diff file
locations = locations_from_diff_file("diff_file.txt")
```
