import argparse
import ast
import json
import os

from ast import NodeTransformer
from ast import unparse
from git import Repo, NoSuchPathError
from pathlib import Path
import subprocess
from typing import Dict

from BugsInPy.utils import checkout

from BugsInPy.bgp import BGPConfig, run_test, InvalidExecutionOrderError


class ReplaceFunctionNode(ast.NodeTransformer):
    def __init__(self, target_lineno: int, replacement_node: ast.FunctionDef):
        self.target_lineno = target_lineno
        self.replacement_node = replacement_node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Check if the function has decorators
        start_lineno = node.lineno
        if node.decorator_list:
            # Adjust the start line number to the first decorator's line number
            start_lineno = node.decorator_list[0].lineno

        # Replace the node if it starts at the target line number
        if start_lineno == self.target_lineno:
            return self.replacement_node
        return node


def replace_code(bug_data: Dict, repo_bug_id: str, file_path: Path) -> None:
    lineno = int(bug_data["start_line"])
    replacement_code = bug_data["replace_code"]

    # Parse the original source code
    with file_path.open("r", encoding="utf-8") as source_file:
        source_code = source_file.read()
    tree = ast.parse(source_code)

    # Parse the replacement code to get its AST
    replacement_tree = ast.parse(replacement_code)
    if not isinstance(replacement_tree.body[0], ast.FunctionDef):
        raise ValueError("Replacement code does not contain a function definition.")

    # Use the NodeTransformer to replace the original function with the replacement
    transformer = ReplaceFunctionNode(lineno, replacement_tree.body[0])
    modified_tree = transformer.visit(tree)

    # Unparse the modified AST to get the source code
    modified_code = unparse(modified_tree)

    # Write the modified code back to the file
    with file_path.open("w", encoding="utf-8") as source_file:
        source_file.write(modified_code)


def parse_json(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
        return data


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def add_imports(bug_data: Dict, repo_bug_id: str, file_path: Path) -> None:
    imports_list = bug_data.get("import_list", [])

    # Parse the original source code
    with file_path.open("r", encoding="utf-8") as source_file:
        source_code = source_file.read()

    tree = ast.parse(source_code)

    # Find the insertion point for new imports
    # This is usually at the beginning of the file, but after any module docstring or __future__ imports
    insert_index = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_index = max(
                insert_index, source_code.rfind("\n", 0, node.end_lineno)
            )
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            insert_index = max(
                insert_index, source_code.rfind("\n", 0, node.end_lineno)
            )
        else:
            break

    # Prepare the new import section
    new_imports = "\n".join(imports_list) + "\n"
    if insert_index > 0:
        new_imports = "\n" + new_imports

    # Insert the new imports into the source code
    new_source_code = (
        source_code[:insert_index] + new_imports + source_code[insert_index:]
    )

    # Write the modified source code back to the file
    with file_path.open("w", encoding="utf-8") as source_file:
        source_file.write(new_source_code)


def main():
    parser = argparse.ArgumentParser(
        description="Process a JSON file and output to specified directory."
    )
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output file JSON",
    )
    parser.add_argument(
        "--envs-dir",
        default=None,
        help="Custom directory path for envs. Use absolute path.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="A timeout for test runs."
    )

    args = parser.parse_args()

    data = parse_json(args.input_json)
    if args.envs_dir:
        BGPConfig.BIP_ENVIRONMENT_DIR = Path(args.envs_dir) / "envs"
        BGPConfig.BIP_CLONED_REPOS = Path(args.envs_dir) / "repos"

    with open(BGPConfig.BUG_RECORDS, "r") as bug_record_file:
        bug_records = json.load(bug_record_file)

    test_status_records: Dict[str, Dict[str, str]] = {}
    if os.path.exists(BGPConfig.TEST_STATUS_RECORDS):
        with open(BGPConfig.TEST_STATUS_RECORDS, "r") as test_status_records_file:
            test_status_records = json.load(test_status_records_file)

    bug_list = bug_records.keys()
    output_dict = {}
    for repo, repo_data in data.items():
        for bug_data in repo_data:
            bug_id = bug_data["bugID"]
            repo_bug_id = f"{repo}:{bug_id}"
            if repo_bug_id not in bug_list:
                raise ValueError(f"The following repo: {repo_bug_id} does not exist!")
            if args.envs_dir:
                repo_path = BGPConfig.BIP_CLONED_REPOS / repo_bug_id.replace(":", "_")
            else:
                repo_path = BGPConfig.BIP_CLONED_REPOS / repo
            file_path = repo_path / bug_data["file_name"]

            # checkout(bug_id, repo_path, bug_records[repo_bug_id]["buggy_commit_id"])
            try:
                repo = Repo(repo_path)
            except NoSuchPathError as e:
                msg = (
                    f"Please clone {bug_id}'s repository before running"
                    " either the `checkout` or `prep` commands."
                )
                raise InvalidExecutionOrderError(msg) from e
            if repo_bug_id in test_status_records:
                test_status_record = test_status_records[repo_bug_id]
            else:
                test_status_record = None

            test_data_no_exists = (
                not test_status_record
                or not "python_path" in test_status_record
                or not Path(test_status_record["python_path"]).exists()
            )
            if test_data_no_exists and not args.envs_dir:
                raise EnvironmentError(f"Bug {repo_bug_id} not prepped.")

            if args.envs_dir:
                python_path = str(
                    Path(args.envs_dir)
                    / "envs"
                    / repo_bug_id.replace(":", "_")
                    / "bin"
                    / "python"
                )
            else:
                python_path = test_status_record["python_path"]
            # Save before replacing
            with file_path.open("r", encoding="utf-8") as source_file:
                store_source_code = source_file.read()

            replace_code(bug_data, repo_bug_id, file_path)
            return_code = run_test(
                repo_bug_id,
                repo_path,
                python_path,
                bug_records[repo_bug_id]["failing_test_command"],
                failing=False,
                timeout=args.timeout,
                test_output_stdout=subprocess.DEVNULL,
            )
            if return_code != 0 and bug_data.get("import_list", []) != []:
                add_imports(bug_data, repo_bug_id, file_path)
                return_code = run_test(
                    repo_bug_id,
                    repo_path,
                    python_path,
                    bug_records[repo_bug_id]["failing_test_command"],
                    failing=False,
                    timeout=args.timeout,
                    test_output_stdout=subprocess.DEVNULL,
                )

            output_dict[repo_bug_id] = return_code
            with file_path.open("w", encoding="utf-8") as source_file:
                source_file.write(store_source_code)

    write_json(args.output_file, output_dict)


if __name__ == "__main__":
    main()
