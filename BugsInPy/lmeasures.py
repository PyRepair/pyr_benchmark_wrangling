#!/usr/bin/env python3

"""Collect bug-specific localisation relevant statistics from the BugsInPy 
benchmark."""

import ast
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import pprint as pp
import re
from statistics import mean, median
import subprocess
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set, Union
import sys
from BugsInPy.bgp import BGPConfig

from . import bgp as bgp
import diff_utils as du

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class TestMeasures(TypedDict):
    test_count: int
    test_method_sloc_total: int
    test_method_sloc_max: int
    test_method_sloc_mean: float
    test_method_sloc_median: float
    test_method_assert_count_total: int
    test_method_assert_count_max: int
    test_method_assert_count_mean: float
    test_method_assert_count_median: float


class ProjectMeasures(TypedDict):
    project_total_files: int
    project_max_lines: int
    project_mean_lines: float
    project_median_lines: float


SKIPPED_BUGS = {
    "spacy": "The test extraction does not seem to work",
    "tornado:3": "no test file",
    "scrapy:13": "No test file",
    "keras:12": "Keras:12 has exact same buggy commit hash and fixed commit hash",
    "pandas:61": "Some error",
    "pandas:59": "No test file found",
    "pandas:5": "No test file found",
}


def parse_run_test_sh(run_test_sh: Path) -> list[Dict[str, str]]:
    """BugsInPy's run_test.sh scripts work with three different testing
    frameworks:  pytest 74%, tox ~2%, and unittest ~24%.  This script
    routes its input to the appropriate parser."""

    bugid = run_test_sh.parts[-4] + ":" + run_test_sh.parts[-2]

    patterns = [
        r"(pytest|py\.test)",
        r"tox",
        r"unittest",
    ]

    try:
        with open(run_test_sh, "r", encoding="utf-8") as file:
            run_test_lines = file.read().rstrip().split("\n")
        # Check precondition
        for line in run_test_lines:
            line = line.strip()
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, line):
                    matched_patterns.append(pattern)

            if len(matched_patterns) == 0:
                raise ValueError(
                    f"{bugid}: {run_test_sh} contains a line without a"
                    f" matching pattern: '{line}'."
                )
            if len(matched_patterns) > 1:
                raise ValueError(
                    f"{bugid}: {run_test_sh} contains a line that matches"
                    f" multiple patterns: '{line}'."
                )

    except FileNotFoundError:
        print(f"File not found: {run_test_sh}")
    except Exception as e:  # pylint: disable=broad-exception-caught,invalid-name
        print(f"Error: {e}")
    if re.search("pytest", matched_patterns[0]):
        parse_command = parse_pytest
    elif "tox" in matched_patterns:
        parse_command = parse_tox
    else:  # "unittest" in matched_patterns must hold.
        parse_command = parse_unittest

    bugid = run_test_sh.parts[-4] + ":" + run_test_sh.parts[-2]

    # Some run_test.sh use shell's ';' to join commands on a single line.
    test_commands: List[str] = []
    for item in run_test_lines:
        lines = item.split(";")
        test_commands.extend(line.strip() for line in lines)

    if len(test_commands) > 1:
        logging.debug(f"{bugid} has multiple test commands.")

    parsed_test_commands = []
    for test_command in test_commands:
        test_parts = parse_command(bugid, test_command)
        parsed_test_commands.append(test_parts)

    if not parsed_test_commands:
        raise ValueError(
            f"In {bugid}, {test_commands} lines violates run_test.sh's expected format."
        )

    return parsed_test_commands


def parse_tox(bugid: str, test_command: str) -> Dict[str, str]:
    """Parse a run_test.sh line that uses tox.  The expected format is
        'tox <slash_delimited_path_to_test_file>::<test_function>'.
    Note that BIP does not contain any class names in its handful of tox
    commands.  This function would need to be updated to handle optional
    class names in the test command.
    """

    parts = test_command.split()

    if len(parts) != 2 or parts[0] != "tox":
        raise ValueError(
            f"{bugid}:  The tox command {test_command} does not have thecorrect shape."
        )

    components = parts[1].split("::")

    test_parts: Dict[str, str] = {}
    test_parts["test_path_from_repo_root"] = components[0]
    test_parts["function_name"] = components[-1]

    return test_parts


def parse_unittest(bugid: str, test_command: str) -> Dict[str, str]:
    """Parse run_test.sh lines that use unittest.  This function assumes
    that the test_commands contains a single unittest invocation of the form:
        'python(3(.\d)+)* -m unittest [-q] test_string'
    where test_string has the form
        'path.to.test.file.classname.testname.'
    """

    parts = test_command.split()

    if len(parts) < 4 or parts[1] != "-m" or parts[3] == "unittest":
        raise ValueError(
            f"{bugid}:  The unittest command {test_command} does not have"
            "the correct shape."
        )

    test_string = parts[-1]
    components = test_string.split(".")
    if len(components) < 3:
        raise ValueError(
            f"{bugid}:  Test command {test_command} does not have the correct"
            " shape: its test parameter has too few components."
        )

    test_parts: Dict[str, str] = {}
    test_parts["test_path_from_repo_root"] = "/".join(components[:-2]) + ".py"
    test_parts["class_name"] = components[-2]
    test_parts["function_name"] = components[-1]

    return test_parts


def parse_pytest(bugid: str, test_command: str) -> Dict[str, str]:
    """When using pytest, BugsInPy's run_test.sh script is a list of pytest
    commands, where each line has this format:
    '<test method> <args> <test_path>::[<class_name>::]<function_name>[params]'
    where test_method is one of py.test, pytest, or python3 -m pytest
    and test_path is relative to the root of the buggy project's repository.

    Pytest can take test_path as a directory.  In this case,
    '::<class_name>::<function_name>[params]' does not appear.  This script
    errors on this format.

    Some programmers, when they see a problem, think "I know I'll use
    a regex."  Then they have two problems.
    """

    # Ensure we have a test_path and function_name, check if optional
    # class_name is present.
    dcolon_count = test_command.count("::")
    if dcolon_count > 2:
        msg = (
            f"In {bugid}, the pytest command '{test_command}' has"
            " an unexpected format. Its count of '::' is"
            f" {dcolon_count}, which is greater than 2."
        )
        raise ValueError(msg)

    # Discard parameters
    param_free_command = re.sub(r"\[.*?\]$", "", test_command.strip())

    # Some commands are just '\n'
    if param_free_command == "":
        return {}

    # Split on last whitespace to handle potential test method arguments
    pattern = r"(.*) ([^ ]+)$"
    match = re.search(pattern, param_free_command)
    if match:
        test_method_args = match.group(1)
        path_class_function = match.group(2)
    else:
        msg = (
            f"In {bugid}, the pytest command '{test_command}'"
            " has an unexpected format. It does not split test method"
            " and args from path, class and function name"
            " on its last space."
        )
        logging.debug(msg)

    msg = (
        f"{bugid}'s pytest test method and any args passed to it"
        f" were '{test_method_args}'."
    )
    logging.debug(msg)

    test_parts: Dict[str, str] = {}
    # Extract test path and optionally its class and function names.
    if dcolon_count == 0:
        # Input only contains a test file.
        test_parts["test_path_from_repo_root"] = path_class_function
    else:
        if dcolon_count == 1:
            pattern = r"^(.*?)::(.*?)$"
        elif dcolon_count == 2:
            pattern = r"^(.*?)::(.*?)::(.*?)$"
        matches = re.match(pattern, path_class_function)
        if matches:
            test_parts["test_path_from_repo_root"] = matches.group(1)
            if dcolon_count == 1:
                test_parts["function_name"] = matches.group(2)
            else:  # dcolon_count is 2
                test_parts["class_name"] = matches.group(2)
                test_parts["function_name"] = matches.group(3)
        else:
            raise ValueError(
                f"In {bugid}, the pytest command {test_command} violates"
                " the expected format."
            )
    return test_parts


def count_lines_assert(root: Union[ast.AST, ast.Module]) -> Tuple[int, int]:
    """Count the sloc and the assert in the AST subtree starting from root."""
    if isinstance(root, ast.Module):
        # Happens when ast is build on complete test files
        sloc = root.body[-1].end_lineno - root.body[0].lineno + 1
    else:
        sloc = root.end_lineno - root.lineno + 1
    assert_call_count = sum(
        isinstance(subnode, ast.Call)
        and (
            (isinstance(subnode.func, ast.Name) and "assert" in subnode.func.id.lower())
            or (
                isinstance(subnode.func, ast.Attribute)
                and "assert" in subnode.func.attr.lower()
            )
        )
        for subnode in ast.walk(root)
    )
    assert_statement_count = sum(
        isinstance(subnode, ast.Assert) for subnode in ast.walk(root)
    )
    return sloc, assert_call_count + assert_statement_count


def get_test_lines_asserts(
    test_path: Path, class_name: str, function_name: Optional[str]
) -> Tuple[int, int]:
    """Use AST to count a test's SLOC and number of assert statements."""

    with open(test_path, "r", encoding="utf-8") as file:
        test_lines = file.read()

    tree = ast.parse(test_lines, filename=str(test_path))

    if not function_name:
        return count_lines_assert(tree)

    test_function = None
    for node in ast.walk(tree):
        if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
            for subnode in node.body:
                if (
                    isinstance(subnode, ast.FunctionDef)
                    and subnode.name == function_name
                ):
                    test_function = subnode
                    break
        elif (
            not class_name
            and isinstance(node, ast.FunctionDef)
            and node.name == function_name
        ):
            test_function = node
            break

    if not test_function:
        bugid = test_path.parts[-3] + ":" + test_path.parts[-1]
        msg = (
            f"In {bugid}, the test {function_name} was not found"
            f" in {test_path}.  Perhaps you have not checked out"
            " the fixed version of the repo?"
        )
        logging.warning(msg)
        return 0, 0

    return count_lines_assert(test_function)


def compute_test_file_measures(
    run_test_path: Path, repo_root_path: Path
) -> TestMeasures:
    """Compute localisation relevant test file measures."""

    result_dict: TestMeasures = {
        "test_count": 0,
        "test_method_sloc_total": 0,
        "test_method_sloc_max": 0,
        "test_method_sloc_mean": 0.0,
        "test_method_sloc_median": 0,
        "test_method_assert_count_total": 0,
        "test_method_assert_count_max": 0,
        "test_method_assert_count_mean": 0.0,
        "test_method_assert_count_median": 0,
    }

    parsed_test_commands = parse_run_test_sh(run_test_path)
    result_dict["test_count"] = len(parsed_test_commands)
    sloc_counts = []
    assert_counts = []
    for command in parsed_test_commands:
        test_path = repo_root_path / command["test_path_from_repo_root"]
        if not test_path.exists():
            project_repo_path = test_path.parent.parent.parent
            if not project_repo_path.exists():
                message = (
                    f"The test's repo {project_repo_path} has not been checked out."
                )
            else:
                message = f"The test path {test_path} does not not exist."
            logging.warning(message)
            continue
        class_name = command.get("class_name", None)

        sloc, assert_count = get_test_lines_asserts(
            test_path, class_name, command.get("function_name", None)
        )
        sloc_counts.append(sloc)
        assert_counts.append(assert_count)

    if not sloc_counts:
        raise EnvironmentError(f"No test files found in repo {repo_root_path}.")

    result_dict["test_method_sloc_total"] = sum(sloc_counts)
    result_dict["test_method_sloc_max"] = max(sloc_counts)
    result_dict["test_method_sloc_mean"] = mean(sloc_counts)
    result_dict["test_method_sloc_median"] = median(sloc_counts)
    result_dict["test_method_assert_count_total"] = sum(assert_counts)
    result_dict["test_method_assert_count_max"] = max(assert_counts)
    result_dict["test_method_assert_count_mean"] = mean(assert_counts)
    result_dict["test_method_assert_count_median"] = median(assert_counts)

    return result_dict


def count_lines_in_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return len(file.readlines())
    except Exception as e:  # pylint: disable=invalid-name
        raise ValueError(f"Error reading {file_path}") from e


def compute_project_repo_measures(bug_repo_path: Path) -> ProjectMeasures:
    """Extract localisation relevant repository-wide measures"""

    file_paths = []

    for root, _, files in os.walk(bug_repo_path):
        if "venv" in root:
            continue
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = Path(root) / file
            if file_path.is_file():
                file_paths.append(file_path)

    total_files = len(file_paths)

    line_counts = [count_lines_in_file(file_path) for file_path in file_paths]

    measures: ProjectMeasures = {
        "project_total_files": total_files,
        "project_max_lines": max(line_counts),
        "project_mean_lines": mean(line_counts),
        "project_median_lines": median(line_counts),
    }

    return measures


def compute_measures(bug_info_dir: Path, bug_repo_path: Path) -> Dict[str, Any]:
    """Factored this code out of main and functionalised it to end a pylint complaint."""
    measures: Dict[str, Any] = {}

    measures = dict(du.measure_localisation_diff_file(bug_info_dir / "bug_patch.txt"))
    with open(bug_info_dir / "bug_patch.txt", "r") as f:
        measures["location"] = du.locations_from_diff([f.read()])
    bugsinpy_projects_at_github = (
        "https://github.com/soarsmu/BugsInPy/blob/master/projects/"
    )

    project = bug_info_dir.parts[-3]
    bug_number = bug_info_dir.parts[-1]
    bug_id = project + ":" + bug_number

    patch_url = (
        bugsinpy_projects_at_github + project + "/bugs/" + bug_number + "/bug_patch.txt"
    )
    measures["bug_fix_url"] = patch_url

    buggy_commit_id = ""
    fixed_commit_id = ""
    with open(BGPConfig.BIP_BENCHMARKS_DIR / "bgp_bug_records.json", "r") as json_file:
        bug_records = json.load(json_file)
        bug_record = bug_records[bug_id]
        buggy_commit_id = bug_record["buggy_commit_id"]
        fixed_commit_id = bug_record["fixed_commit_id"]

    if bug_repo_path.exists():
        bgp.checkout(bug_id, bug_repo_path, fixed_commit_id)
        measures.update(
            compute_test_file_measures(bug_info_dir / "run_test.sh", bug_repo_path)
        )
        bgp.checkout(bug_id, bug_repo_path, buggy_commit_id)
        measures.update(compute_project_repo_measures(bug_repo_path))

    return measures


def split_bug_ids(value):
    """Used to split a list command line argument."""
    return value.split(",")


def main():  # pylint: disable=missing-function-docstring
    pyrepair_benchmarks_dir = BGPConfig.BIP_BENCHMARKS_DIR
    parser = argparse.ArgumentParser(
        description="Calculate localisation relevant measures for in bug in BugsInPy."
    )
    parser.add_argument(
        "--bugsinpy_root",
        type=str,
        help="Path to BugsInPy",
        default=f"{pyrepair_benchmarks_dir}/BugsInPy/projects",
    )
    parser.add_argument(
        "--project_repos",
        type=str,
        help="Path to BugsInPy project repositories",
        default=f"{pyrepair_benchmarks_dir}/BugsInPy_Cloned_Repos",
    )
    parser.add_argument(
        "--bugids",
        "-b",
        type=split_bug_ids,
        help="Comma delimited list of repo:id formatted bug ids in BugsInPy",
        default=None,
    )
    parser.add_argument(
        "--locations",
        "-l",
        type=bool,
        help="A list of locations in the dictionary.",
        default=False,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help=(
            "Number of bugs to run lmeasures on. Default is 3000. Useful for debugging."
        ),
    )

    args = parser.parse_args()

    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.getLogger().setLevel(log_level_map[args.log_level])

    bugsinpy_root_path = Path(args.bugsinpy_root)
    measures_dict = {}

    if args.bugids:
        bug_info_dirs = [
            bugsinpy_root_path / project / "bugs" / bugid
            for project, bugid in (bug.split(":") for bug in args.bugids)
        ]
    else:
        bug_info_dirs = list(bugsinpy_root_path.glob("**/bugs/*"))

    for idx, bug_info_dir in enumerate(bug_info_dirs):
        if idx >= args.limit:
            break
        try:
            # Bugs have integer IDs, stray files should be ignored
            int(bug_info_dir.parts[-1])
        except:  # pylint: disable=bare-except
            continue
        if bug_info_dir.parts[-3] in SKIPPED_BUGS:
            # Currently unsupported projects
            continue
        key = bug_info_dir.parts[-3] + ":" + bug_info_dir.parts[-1]
        if key in SKIPPED_BUGS:
            # Currently unsupported bugs
            continue

        bug_repo_path = Path(args.project_repos) / bug_info_dir.parts[-3]

        measures_dict[key] = compute_measures(bug_info_dir, bug_repo_path)

    pp.pprint(measures_dict)

    csv_file = "localisation_measures.csv"
    # This script always computes all the diff measures from bgp_bug_records,
    # which contains the fix diffs, but it only computes the other measures
    # cloned repos.  This means that there can be a field mismatch during the
    # csv write below.  The following assignment collects all the field names
    # from the TypeDicts in use, even when not all of the bugs have been cloned.
    largest_dict = max(measures_dict.values(), key=lambda d: len(d))
    fieldnames = ["bug_id"] + list(largest_dict.keys())

    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for bug_id, values in measures_dict.items():
            writer.writerow({"bug_id": bug_id, **values})


if __name__ == "__main__":
    main()
