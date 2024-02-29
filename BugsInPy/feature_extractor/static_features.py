from diff_utils import locations_from_diff
from typing import Dict
from pathlib import Path
from static_library import (
    get_function_data_from_file,
    get_class_data,
    filter_unused_imports,
    extract_import_statements,
    filter_python_keywords,
    expand_to_complete_structure,
    merge_non_function_lines,
    extract_functions_from_file,
    extract_function_signatures_from_file,
    extract_variables_from_file,
)
from dataclasses import asdict


def extract_buggy_function(bug_record, repo_path):
    locations = locations_from_diff(bug_record["fixing_patch"])
    visited_code = set()
    features: Dict = {}
    non_function_lines = []
    visited_lines = set()
    for file_path_str, lineno in locations:
        file_path = repo_path / Path(file_path_str)
        if "test_" in file_path.name:
            continue
        if (file_path, lineno) in visited_lines:
            continue
        if str(file_path) not in features:
            features[str(file_path)] = {}
            features[str(file_path)]["buggy_functions"] = []

        function_data = get_function_data_from_file(file_path, lineno)
        if function_data.start_line:
            for l in range(function_data.start_line, function_data.end_line + 1):
                visited_lines.add((file_path, l))

        # Check if the line is in any function
        if function_data.code is not None:
            variables = {}
            for var in function_data.variables:
                variables[var] = list(function_data.variables[var])
            if function_data.code not in visited_code:
                class_info = get_class_data(file_path, lineno)
                import_statements = extract_import_statements(file_path)

                used_imports = filter_unused_imports(
                    import_statements, function_data.code
                )
                features[str(file_path)]["buggy_functions"].append(
                    {
                        "function_name": function_data.name,
                        "function_code": function_data.code,
                        "decorators": (
                            function_data.decorators
                        ),  # Assuming this is the new field
                        "docstring": function_data.docstring,
                        "start_line": function_data.start_line,
                        "end_line": function_data.end_line,
                        "variables": variables,
                        "filtered_variables": filter_python_keywords(variables),
                        "diff_line_number": lineno,
                        "class_data": asdict(class_info) if class_info else None,
                        "used_imports": used_imports,
                    }
                )
                visited_code.add(function_data.code)

        else:
            # Handle lines outside functions
            non_function_lines.append((file_path_str, lineno))

    snippet_ranges = merge_non_function_lines(non_function_lines, repo_path)

    for file_path_str, intervals in snippet_ranges.items():
        file_path = repo_path / Path(file_path_str)

        with open(file_path, "r") as file:
            lines = file.readlines()
            for start_line, end_line in intervals:
                expanded_start, expanded_end = expand_to_complete_structure(
                    file_path, start_line, end_line
                )
                snippet_code = "".join(lines[expanded_start - 1 : expanded_end]).strip()
                if snippet_code.strip() == "":
                    continue
                features[str(file_path)].setdefault("snippets", []).append(
                    {
                        "snippet_code": snippet_code,
                        "start_line": expanded_start,
                        "end_line": expanded_end,
                    }
                )
    return features


def extract_functions_and_variables_from_file(features: Dict):
    files = features.keys()
    for file_path in files:
        if file_path == "test_data":
            continue
        functions = extract_functions_from_file(file_path)
        variables = extract_variables_from_file(file_path)
        for var in variables:
            variables[var] = list(variables[var])
        features[file_path]["inscope_functions"] = functions
        features[file_path]["variables_in_file"] = variables
        features[file_path]["filtered_variables_in_file"] = filter_python_keywords(
            variables
        )
    return features
