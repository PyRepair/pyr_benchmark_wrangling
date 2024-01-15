# Copyright 2023 The pyrepair and triangulate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for working with the BugsInPy benchmark."""

import configparser
import logging
import os
from pathlib import Path
import json

# import pprint as pp
import re
import shutil
import sys
import subprocess
import textwrap
from typing import Dict, List
from BugsInPy.class_extractor import extract_classes
from BugsInPy.feature_extractor import extract_features
from BugsInPy.bgp_config import BGPConfig, BugRecord
from BugsInPy.test_runner import (
    get_test_command_and_env,
    prep,
    run_test,
    ignore_venv,
    TestStatus,
)

import argparse


from BugsInPy.utils import checkout, GIT_INSTALLED, git_clone, clone, delete
from BugsInPy.exceptions import InvalidExecutionOrderError, BugNotFoundError


def extract_github_url(file_path: Path) -> str:
    """Extract the github_url value from BugsInPy's project.info file."""

    github_url_pattern = re.compile(r'github_url="([^"]+)"')
    with open(file_path, "r", encoding="utf-8") as f:  # pylint: disable=invalid-name
        content = f.read()
        match = github_url_pattern.search(content)
        if match:
            return match.group(1)
        raise ValueError(f"{file_path} does not contain a GitHub URL.")


def build_bug_record(bug_number: str, project_dir: Path) -> BugRecord:
    """Build a bug record from BugsInPy's repo and bug information."""

    record: BugRecord = {
        "benchmark_url": "",
        "bip_python_version": "",
        "buggy_commit_id": "",
        "failing_test_command": "",
        "fixed_commit_id": "",
        "fixing_patch": [],  # This stores a diff
        "test_file": "",
    }

    info_file_path = project_dir / "project.info"
    if os.path.isfile(info_file_path):
        record["benchmark_url"] = extract_github_url(info_file_path)
    else:
        raise EnvironmentError(f"{info_file_path} is not a file.")

    bug_dir = project_dir / "bugs" / bug_number
    with open(bug_dir / "bug.info", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split("=", 1)
                match key.strip():
                    case "buggy_commit_id":
                        record["buggy_commit_id"] = value.strip().strip('"')
                    case "fixed_commit_id":
                        record["fixed_commit_id"] = value.strip().strip('"')
                    case "python_version":
                        # We rename b/c we may need to use a different python version
                        # to run the test due to bitrot.
                        record["bip_python_version"] = value.strip().strip('"')
                    case "test_file":
                        record["test_file"] = value.strip().strip('"')
    with open(bug_dir / "bug_patch.txt", "r", encoding="utf-8", newline=None) as file:
        record["fixing_patch"] = file.readlines()
    with open(bug_dir / "run_test.sh", "r", encoding="utf-8", newline=None) as file:
        record["failing_test_command"] = ";".join(file.readlines()).strip()
    return record


def build_bug_dict() -> Dict[str, BugRecord]:
    """Build a dictionary of BugsInPy's repo and bug information."""
    bug_records: Dict[str, BugRecord] = {}

    if not os.path.exists(BGPConfig.BIP_ROOT):
        git_clone(BGPConfig.BIP_GIT_URL, BGPConfig.BIP_ROOT)
    ignore_venv(BGPConfig.BIP_ROOT)

    for project_name in os.listdir(BGPConfig.BIP_PROJECTS_DIR):
        project_dir = BGPConfig.BIP_PROJECTS_DIR / project_name

        if os.path.isdir(project_dir):
            for bug_number in os.listdir(project_dir / "bugs"):
                if not bug_number.isdigit():
                    continue  # ignore detritus in BugsInPY
                bug_id = f"{project_name}:{bug_number}"
                bug_record = build_bug_record(bug_number, project_dir)
                bug_records[bug_id] = bug_record

    return bug_records


def bgp_setup() -> None:
    """Set up this bgp.py script itself."""

    # Check BGP's Python interpreter dependencies
    python_versions_to_check = ["3.7", "3.8", "3.9"]
    found_versions = []

    for version in python_versions_to_check:
        if shutil.which(f"python{version}"):
            found_versions.append(version)

    python_versions_message = (
        "BugsInPy's bugs were downloaded 19 June, 2020.  Roughly, 7% occurred"
        " in projects using 3.6; 30% in 3.7, 63% in 3.8. Python versions <3.8"
        " are deprecated. At the time of this writing, it was still possible"
        " to manually install Python 3.7. For some bugs that occurred when the"
        " subject project was using Python <3.8, BGP achieves its goal of"
        " executing the regression test with Python3.8."
    )

    not_found_versions = list(set(python_versions_to_check) - set(found_versions))
    if not_found_versions:
        wrapped_text = textwrap.fill(python_versions_message, width=80)
        print(wrapped_text)

        print(
            "\nCould not find the following Python versions that BPG needs"
            f" in the PATH: {not_found_versions[0]}",
            end="",
        )
        for version in not_found_versions[1:]:
            print(f", {version}", end="")
        print(".")
        sys.exit(1)

    # Check BGP's Python Module dependencies
    if not GIT_INSTALLED:
        print("GitPython is not installed. Please install", end=" ")
        print("the modules in BGP's requirements.txt.")
        sys.exit(1)

    bug_records = build_bug_dict()
    with open(BGPConfig.BUG_RECORDS, "w") as bug_records_file:
        json.dump(bug_records, bug_records_file)

    # BGPConfig.BUG_RECORDS is versioned, so it should exist
    if not os.path.exists(BGPConfig.BUG_RECORDS):
        # This is an error message for the future
        # Currently the bug data file is not checked in
        print("The BugsInPy benchmark's bug data was not found. This is", end=" ")
        print(
            "suprising, as this file is checked into the repo.  If you wish ",
            end=" ",
        )
        print("to rebuild it from scratch, issue 'bgp rebuild_bug_database'.")
        sys.exit(1)

    bgp_goal = (
        "BGP's goal is _not_ cleanly build projects, but rather to build them"
        " in order to run the regression test(s) on the fixed and buggy"
        " versions of the subject and confirm that the test passes on the"
        " fixed version and fails on the buggy version. For this reason,"
        " prepping these bugs is quite noisy and generates many errors that"
        " BGP ignores."
    )
    wrapped_text = textwrap.fill(bgp_goal, width=80)
    print()
    print(wrapped_text)
    print()


def main():
    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument(
        "command",
        choices=[
            "setup",
            "clone",
            "checkout_buggy",
            "checkout_fixed",
            "prep",
            "run_test",
            "delete",
            "extract_features",
            "prep_ignore_venv",
            "update_bug_records",
            "delete_bug_repo",
            "get_test_command",
        ],
        help="Command to execute",
    )

    parser.add_argument(
        "--bugids", nargs="*", default=[], help="List of repo and bug id pairs"
    )
    parser.add_argument("--repo_list", nargs="*", default=[], help="List of repos")
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="restart the command from scratch",
    )
    parser.add_argument(
        "--repo_dir", action="store_true", help="Directory containing repos"
    )
    parser.add_argument(
        "--envs-dir",
        default=None,
        help="Custom directory path for envs. Use absolute path.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="An atomic timeout for BGP in seconds."
    )
    parser.add_argument("--test_output", default=None, help="The test output xml file.")
    parser.add_argument(
        "--feature-json", default=None, help="The json output xml file."
    )
    parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the log level",
    )
    parser.add_argument(
        "--verbose_prep", action="store_true", help="Control prep's verbosity."
    )

    args = parser.parse_args()

    command = args.command
    if (
        (args.bugids and (args.repo_list or args.repo_dir))
        or (args.repo_list and (args.bugids or args.repo_dir))
        or (args.repo_dir and (args.bugids or args.repo_list))
        or (args.test_output and command != "run_test")
    ):
        raise ValueError("Mutually exclusive args provided.")
        sys.exit(1)

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.envs_dir:
        BGPConfig.BIP_ENVIRONMENT_DIR = Path(args.envs_dir) / "envs"
        BGPConfig.BIP_CLONED_REPOS = Path(args.envs_dir) / "repos"
    if command == "setup":
        bgp_setup()
        extract_classes(BGPConfig.BIP_ENVIRONMENT_CLASSES)
        sys.exit()

    if command == "update_bug_records":
        bug_records = build_bug_dict()
        with open(BGPConfig.BUG_RECORDS, "w") as bug_records_file:
            json.dump(bug_records, bug_records_file)
        extract_classes(BGPConfig.BIP_ENVIRONMENT_CLASSES)
        sys.exit()

    if not os.path.exists(BGPConfig.BUG_RECORDS):
        raise EnvironmentError(
            "The BugsInPy bug database not found; Issue"
            " 'update_bug_records' to build it?"
        )

    with open(BGPConfig.BUG_RECORDS, "r") as bug_record_file:
        bug_records = json.load(bug_record_file)

    test_status_records: Dict[str, Dict[str, str]] = {}
    if os.path.exists(BGPConfig.TEST_STATUS_RECORDS):
        with open(BGPConfig.TEST_STATUS_RECORDS, "r") as test_status_records_file:
            test_status_records = json.load(test_status_records_file)

    # Set bugid_list to the bugs 1) passed via --bugids, 2) checked out into
    # the directory containing cloned bug repos (--repodir), or 3) fall
    # through to all bugs in the BugsInPy database.
    bugid_list = args.bugids
    if len(args.repo_list) > 0:
        bugid_list = bug_records.keys()
        bugid_list = [
            bugid for bugid in bugid_list if bugid.split(":")[0] in args.repo_list
        ]
    elif args.repo_dir:
        if not BGPConfig.BIP_CLONED_REPOS.is_dir():
            raise ValueError(f"{args.repo_dir} is not a directory.")
        pattern = r"^([^:])+:(\d+)"
        bugid_list = [
            bugid.name
            for bugid in BGPConfig.BIP_CLONED_REPOS.rglob("*")
            if re.match(pattern, bugid.name)
        ]
    elif len(bugid_list) == 0:
        # Run command on all bugs
        bugid_list = bug_records.keys()
    total_failed = []
    test_status_change = False
    features = {}
    separate_envs = args.envs_dir is not None
    for bug_id in bugid_list:
        if args.envs_dir:
            # For windows
            repo_path = BGPConfig.BIP_CLONED_REPOS / bug_id.replace(":", "_")
        else:
            repo_path = BGPConfig.BIP_CLONED_REPOS / bug_id.split(":")[0]
        if not bug_id in bug_records:
            raise BugNotFoundError(f"The bug {bug_id} does not exist in BugsInPy.")
        bug_record = bug_records[bug_id]
        if bug_id in test_status_records:
            test_status_record = test_status_records[bug_id]
        else:
            test_status_record = None
        test_status_change = False
        if args.verbose_prep:
            pip_output_redirection = None
        else:
            pip_output_redirection = subprocess.DEVNULL
        match command:
            # We check out each bug into its own repo to isolate venv's.
            case "clone":
                clone(bug_id, repo_path, bug_records[bug_id], restart=args.restart)
            case "checkout_buggy":
                commit_id = bug_record["buggy_commit_id"]
                checkout(bug_id, repo_path, commit_id, separate_envs=separate_envs)
            case "checkout_fixed":
                commit_id = bug_record["fixed_commit_id"]
                checkout(bug_id, repo_path, commit_id, separate_envs=separate_envs)
            case "extract_features":
                features[bug_id] = extract_features(
                    bug_id, bug_record, repo_path, separate_envs
                )
            case "prep":
                test_status_change, updated_test_status_record = prep(
                    bug_id,
                    repo_path,
                    bug_record,
                    test_status_record,
                    args.timeout,
                    pip_output_redirection,
                    restart=args.restart,
                    separate_envs=separate_envs,
                )
                if test_status_change:
                    test_status_records[bug_id] = updated_test_status_record
                test_status = updated_test_status_record["test_status"]
                if test_status != TestStatus.PASS:
                    total_failed.append((bug_id, test_status))
            case "get_test_command":
                env, test_command = get_test_command_and_env(
                    bug_id,
                    repo_path,
                    bug_record,
                    test_status_record,
                    args.timeout,
                    pip_output_redirection,
                    separate_envs=separate_envs,
                )
                print(f"Python Path:{env}\ntest_command:{test_command}")
            case "prep_ignore_venv":
                ignore_venv(repo_path)
            case "run_test":
                if not test_status_record or not "python_path" in test_status_record:
                    raise EnvironmentError(f"Bug {bug_id} not prepped.")
                run_test(
                    bug_id,
                    repo_path,
                    test_status_record["python_path"],
                    bug_record["failing_test_command"],
                    failing=False,
                    timeout=args.timeout,
                    xml_output=args.test_output,
                )
            case "delete_bug_repo":
                delete(repo_path)
            case _:
                raise ValueError(f"{command} is unknown.")

    if test_status_change:
        with open(BGPConfig.TEST_STATUS_RECORDS, "w") as test_status_records_file:
            json.dump(test_status_records, test_status_records_file)
    if features:
        if args.feature_json:
            with open(args.feature_json, "w") as f:
                json.dump(features, f)
        else:
            print(json.dumps(features))
    if command == "prep":
        print(total_failed)


if __name__ == "__main__":
    main()
