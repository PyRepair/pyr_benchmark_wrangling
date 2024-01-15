import ast
import builtins
import subprocess
import os
import re
from typing import Dict, Set, Tuple, List, Optional
from BugsInPy.utils import checkout
from BugsInPy.test_runner import move_test_file
from diff_utils import locations_from_diff
import ast
import keyword
import shutil

from pathlib import Path
from dataclasses import asdict, dataclass, field
from os.path import commonpath

from .dynamic_features import extract_dynamic_features
from .static_features import (
    extract_buggy_function,
    extract_functions_and_variables_from_file,
)


def extract_features(bug_id, bug_record, repo_path, separate_envs):
    buggy_commit_id = bug_record["buggy_commit_id"]
    move_test_file(bug_id, bug_record, repo_path, separate_envs)

    features = extract_buggy_function(bug_record, repo_path)
    features = extract_functions_and_variables_from_file(features)
    features = extract_dynamic_features(
        features, bug_record, repo_path, bug_id, separate_envs
    )
    return features
