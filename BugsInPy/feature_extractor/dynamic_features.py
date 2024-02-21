from pyclbr import Function
from tracemalloc import start
from BugsInPy.test_runner import prep, run_test
from BugsInPy.bgp_config import BGPConfig, BugRecord
from BugsInPy.utils import checkout
from .file_instrumenter import FileInstrumentor
from .static_features import (
    extract_buggy_function,
    extract_functions_and_variables_from_file,
)
from static_library import code_to_node
from diff_utils import locations_from_diff
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import ast
import contextlib
import io
import json
import logging
import os
import re
import subprocess


import ast


class LastStatementChecker(ast.NodeVisitor):
    def __init__(self):
        self.is_last_statement_return = False

    def check_last_statement(self, node):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("Node is not a function definition.")

        if not node.body:
            return False

        last_stmt = node.body[-1]
        self.visit(last_stmt)
        return self.is_last_statement_return

    def visit_Return(self, node):
        self.is_last_statement_return = True


def function_ends_with_return(node) -> bool:
    try:
        checker = LastStatementChecker()
        return checker.check_last_statement(node.body[0])
    except (SyntaxError, ValueError):
        return False


class ReturnFinder(ast.NodeVisitor):
    def __init__(self):
        self.return_lines = []

    def visit_Return(self, node):
        self.return_lines.append(node.lineno)


def get_return_lines(node):
    """
    Extracts the line numbers of return statements from the given function's source code.

    Args:
    node : The AST node of the function.

    Returns:
    List[int]: A list of line numbers where return statements are found.
    """
    if not node:
        return []

    # Create a ReturnFinder instance and visit each node
    finder = ReturnFinder()
    finder.visit(node)

    return finder.return_lines


def is_upper_camel_case(s: str) -> bool:
    """Check if a string is in upper camel case (PascalCase)."""
    return re.match(r"^[A-Z][a-zA-Z0-9]*$", s) is not None


def is_lower_camel_case(s: str) -> bool:
    """Check if a string is in lower camel case."""
    return re.match(r"^[a-z][a-zA-Z0-9]*$", s) is not None


def is_camel_case(s: str) -> bool:
    return is_upper_camel_case(s) or is_lower_camel_case(s)


@dataclass
class TestData:
    test_path: str
    test_function: str
    test_function_code: Optional[str]
    test_error: Optional[str]
    full_test_error: Optional[str]
    traceback: Optional[str]
    test_error_location: Optional[str]
    test_function_decorators: List[str]


def get_test_status_record(bug_id):
    test_status_records: Dict[str, Dict[str, str]] = {}
    if os.path.exists(BGPConfig.TEST_STATUS_RECORDS):
        with open(BGPConfig.TEST_STATUS_RECORDS, "r") as test_status_records_file:
            test_status_records = json.load(test_status_records_file)
    if bug_id in test_status_records:
        return test_status_records[bug_id]
    else:
        return None


def extract_test_code_and_node(file_path, test_name):
    with open(file_path, "r") as file:
        file_contents = file.read()

    # Parse the file content into an AST
    tree = ast.parse(file_contents)

    # Function to recursively extract the source code of a node
    def get_source_code(node):
        if hasattr(node, "lineno"):
            # Check for decorators and adjust start line if necessary
            start_line = (
                min(decorator.lineno for decorator in node.decorator_list) - 1
                if node.decorator_list
                else node.lineno - 1
            )
            end_line = (
                node.end_lineno if hasattr(node, "end_lineno") else start_line + 1
            )
            return "\n".join(file_contents.splitlines()[start_line:end_line])
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            return get_source_code(node), node

    return None, None


def extract_traceback(output_str) -> Optional[str]:
    """
    Extracts the stacktrace from the given output string.
    The stacktrace starts after a line with multiple underscores and
    ends before another such line or the end of the error message.
    """
    # Split the output into lines
    lines = output_str.split("\n")

    # Variables to hold the start and end of the traceback
    start = None
    end = None

    # Iterate over the lines to find the start and end markers
    for i, line in enumerate(lines):
        if "_ " * 10 in line and start is None:
            start = i + 1  # Start capturing after this line
        elif "_ " * 10 in line and start is not None:
            end = i  # End capturing before this line
            break
        elif line.strip().startswith("E               ") and start is not None:
            end = i + 1  # Include this line as the end of the traceback
            break
    if start is not None and end is not None:
        # Extract the lines that constitute the stacktrace
        return "\n".join(lines[start:end]).strip()
    else:
        return None


def extract_failing_tests(
    xml_output_path: Path, repo_path: Path, test_paths: List[str]
) -> List[Dict]:
    failing_tests = []
    # Parse the XML file
    tree = ET.parse(xml_output_path)
    root = tree.getroot()

    # Iterate over test cases and check for failures
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        if failure is not None:
            full_test_function = testcase.get("name")
            test_function = full_test_function.split("[")[
                0
            ]  # Extract base test function name
            test_classname = testcase.get("classname").replace(
                ".", "/"
            )  # Convert class name to path
            full_test_error = failure.text  # Full error message and traceback

            if is_camel_case(test_classname.split("/")[-1]):
                test_path = "/".join(test_classname.split("/")[:-1])
            else:
                test_path = test_classname

            full_test_path = repo_path / f"{test_path}.py"
            test_code, test_node = extract_test_code_and_node(
                full_test_path, test_function
            )
            traceback = extract_traceback(full_test_error)
            exception_location = extract_exception_location(full_test_error, traceback)
            if test_node:
                decorators = [
                    ast.unparse(decorator) for decorator in test_node.decorator_list
                ]
            else:
                decorators = []
            failing_tests.append(
                asdict(
                    TestData(
                        test_path=str(full_test_path),
                        test_function=test_function,
                        test_function_code=test_code,
                        full_test_error=full_test_error,
                        test_error=failure.get("message"),  # Short error message
                        traceback=traceback,
                        test_error_location=exception_location,
                        test_function_decorators=decorators,
                    )
                )
            )
    return failing_tests


def extract_exception_location(full_error, traceback) -> Optional[str]:
    if traceback is None:
        return None
    try:
        traceback_end_index = full_error.index(traceback) + len(traceback)

        exception_location = full_error[traceback_end_index:].strip()

        return exception_location
    except ValueError:
        return None


def get_chunk_lines(function, locations):
    start_line_of_function = function["start_line"]

    function_code_lines = function["function_code"].split("\n")
    end_line_of_function = start_line_of_function + len(function_code_lines) - 1

    chunk_start_line, chunk_end_line = None, None

    for line in locations:
        if start_line_of_function <= line <= end_line_of_function:
            if chunk_start_line is None or line < chunk_start_line:
                chunk_start_line = line
            if chunk_end_line is None or line > chunk_end_line:
                chunk_end_line = line

    return chunk_start_line, chunk_end_line


def extract_dynamic_values(
    features,
    bug_record,
    repo_path: Path,
    bug_id: str,
    test_status_record: Dict[str, str],
    angelic: bool = False,
):
    # Set the feature name to modify
    if angelic:
        feature_key = "angelic_variable_values"
        # The feature information on fixed commit is unavailable, hence requires construction
        new_features = extract_buggy_function(bug_record, repo_path)
        new_features = extract_functions_and_variables_from_file(new_features)
    else:
        feature_key = "variable_values"
        new_features = features

    # Iterate over buggy functions in features and instrument their start and return points
    for file_path, file_data in new_features.items():
        if "test_data" == file_path:
            continue
        for function in file_data["buggy_functions"]:
            # Initialize FileInstrumentor for each buggy file
            instrumentor = FileInstrumentor(repo_path / file_path)
            start_line = function["start_line"]
            for n_start, line in enumerate(
                function["function_code"].split("\n"), start_line
            ):
                if "def " in line and "@" not in line:
                    break
            end_line = function["end_line"]
            # Add print points at the start of the function
            instrumentor.add_print_point(
                function["function_name"],
                n_start,
                function["filtered_variables"],
                "start",
            )
            func_node = code_to_node[function["function_code"]]
            # Add print points at each return statement
            return_lines = get_return_lines(func_node)
            if not function_ends_with_return(func_node):
                instrumentor.add_print_point(
                    function["function_name"],
                    func_node.end_lineno,
                    function["filtered_variables"],
                    "end_2",
                )
            for rel_line in return_lines:
                # Adjust the line number to be relative to the start of the file
                instrumentor.add_print_point(
                    function["function_name"],
                    rel_line,
                    function["filtered_variables"],
                    "end",
                )

            # Instrument the code
            instrumentor.instrument()

            # Define the path for the test output and run tests
            test_output_path = Path(BGPConfig.BIP_BENCHMARKS_DIR) / "output.xml"
            run_test(
                bug_id,
                repo_path,
                python_path=test_status_record["python_path"],
                failing_test_commands=bug_record["failing_test_command"],
                xml_output=test_output_path,
                timeout=60,
                test_output_stdout=subprocess.DEVNULL,
            )

            # Extract variable information from the test output XML file
            extracted_data = instrumentor.extract_variable_info_from_json()
            # Restore the original code
            instrumentor.restore_original_file()
            feat_to_modify = None
            if angelic:
                for buggy_function in features[file_path]["buggy_functions"]:
                    if buggy_function["function_name"] == function["function_name"]:
                        feat_to_modify = buggy_function
                        break
            else:
                feat_to_modify = function
            # feat_to_modify does not exist if the function does not exist in the fixed commit
            if feat_to_modify:
                if feature_key not in feat_to_modify:
                    feat_to_modify[feature_key] = []
                feat_to_modify[feature_key].extend(extracted_data)
    return features


def extract_angelic_values(
    features,
    bug_record,
    repo_path: Path,
    bug_id: str,
    test_status_record: Dict[str, str],
    separate_envs: bool,
):
    fixed_commit = bug_record["fixed_commit_id"]

    checkout(bug_id, repo_path, fixed_commit, separate_envs=separate_envs)
    extract_dynamic_values(
        features, bug_record, repo_path, bug_id, test_status_record, angelic=True
    )
    return features


def extract_dynamic_features(
    features: Dict,
    bug_record: BugRecord,
    repo_path: Path,
    bug_id: str,
    separate_envs: bool,
):
    logging.disable(logging.CRITICAL)
    test_status_record = get_test_status_record(bug_id)
    _, test_status_record = prep(
        bug_id,
        repo_path,
        bug_record,
        test_status_record=test_status_record,
        pip_output_redirection=subprocess.DEVNULL,
        timeout=600,
        test_output_stdout=subprocess.DEVNULL,
        separate_envs=separate_envs,
    )
    if test_status_record["does_test_run"] == "False":
        _, test_status_record = prep(
            bug_id,
            repo_path,
            bug_record,
            test_status_record=test_status_record,
            pip_output_redirection=subprocess.DEVNULL,
            timeout=600,
            test_output_stdout=subprocess.DEVNULL,
            restart=True,
            separate_envs=separate_envs,
        )

    test_output_path = BGPConfig.BIP_BENCHMARKS_DIR / "output.xml"

    run_test(
        bug_id,
        repo_path,
        python_path=test_status_record["python_path"],
        failing_test_commands=bug_record["failing_test_command"],
        xml_output=test_output_path,
        timeout=60,
        test_output_stdout=subprocess.DEVNULL,
    )

    test_paths = bug_record["test_file"].split(";")
    test_paths = [test_path for test_path in test_paths if test_path.strip() != ""]
    failing_tests = extract_failing_tests(
        test_output_path, repo_path, test_paths=test_paths
    )

    try:
        os.remove(test_output_path)
    except Exception as e:
        print(f"An error occurred: {e}")
    features["test_data"] = failing_tests
    features = extract_dynamic_values(
        features, bug_record, repo_path, bug_id, test_status_record
    )
    features = extract_angelic_values(
        features, bug_record, repo_path, bug_id, test_status_record, separate_envs
    )

    return features
