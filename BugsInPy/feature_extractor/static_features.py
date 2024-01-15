import ast
import builtins
import re
from tracemalloc import start
from typing import Dict, List, Optional
from diff_utils import locations_from_diff
import ast
import keyword
from pathlib import Path
from dataclasses import asdict, dataclass, field


@dataclass
class FunctionData:
    start_line: Optional[int]
    end_line: Optional[int]
    name: Optional[str]
    code: Optional[str]
    variables: Dict[str, str]
    function_calls: List[str]
    decorators: List[str]
    raw_code: Optional[str]
    docstring: str


def remove_indentation(source_code):
    lines = source_code.split("\n")
    # Filter out blank lines to avoid counting them in determining the minimum indentation
    non_blank_lines = [line for line in lines if line.strip()]

    # Find the minimum indentation level (excluding blank lines)
    min_indentation = min(len(line) - len(line.lstrip()) for line in non_blank_lines)

    # Remove the minimum indentation from each line
    unindented_lines = [
        line[min_indentation:] if len(line) > min_indentation else line
        for line in lines
    ]
    return "\n".join(unindented_lines)


def filter_python_keywords(variables):
    """
    Filters out variables that match Python keywords.

    Args:
    variables (Dict[str, Set[int]]): A dictionary with variable names as keys.

    Returns:
    Dict[str, Set[int]]: A filtered dictionary with non-keyword variable names.
    """
    # Get a set of all Python keywords
    python_keywords = set(keyword.kwlist)
    builtin_vars = dir(builtins)
    # Filter out variables that are Python keywords
    filtered_variables = {
        var: lines
        for var, lines in variables.items()
        if var not in python_keywords and var not in builtin_vars
    }

    return filtered_variables


@dataclass
class ClassInfo:
    signature: Optional[str]
    docstring: Optional[str]
    constructor_docstring: Optional[str]
    functions: List[str] = field(default_factory=list)
    constructor_variables: List[str] = field(default_factory=list)
    class_level_variables: List[str] = field(default_factory=list)
    class_decorators: List[str] = field(default_factory=list)
    function_signatures: List[str] = field(default_factory=list)
    class_level_variable_names: List[str] = field(default_factory=list)
    constructor_variable_names: List[str] = field(default_factory=list)


def extract_function_signatures(code: str) -> List[str]:
    function_signatures = []

    try:
        tree = ast.parse(code)

        def get_signature(node):
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    # Include type hint if present
                    type_hint = ast.unparse(arg.annotation)
                    arg_str += f": {type_hint}"
                args.append(arg_str)

            defaults = [ast.unparse(d) for d in node.args.defaults]
            defaults = ["=" + d for d in defaults]
            params = [
                arg + default
                for arg, default in zip(
                    args, [""] * (len(args) - len(defaults)) + defaults
                )
            ]

            if node.args.vararg:
                vararg = "*" + node.args.vararg.arg
                if node.args.vararg.annotation:
                    vararg += f": {ast.unparse(node.args.vararg.annotation)}"
                params.append(vararg)

            if node.args.kwarg:
                kwarg = "**" + node.args.kwarg.arg
                if node.args.kwarg.annotation:
                    kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
                params.append(kwarg)

            signature = f"{node.name}({', '.join(params)})"
            if node.returns:
                # Include return type hint if present
                return_type_hint = ast.unparse(node.returns)
                signature += f" -> {return_type_hint}"
            return signature

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature = get_signature(node)
                function_signatures.append(signature)

    except SyntaxError:
        pass

    return function_signatures


def extract_functions(code: str) -> List[str]:
    functions = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Start line is initially set to the function's start line
                start_line = node.lineno

                # Check decorators for an earlier start line
                for decorator in node.decorator_list:
                    start_line = min(start_line, decorator.lineno)

                # End line is the last line of the function
                end_line = node.end_lineno

                # Extract the function code from the original source
                function_code_lines = code.splitlines()[start_line - 1 : end_line]
                function_code = "\n".join(function_code_lines)
                function_code = remove_indentation(function_code)
                functions.append(function_code)

    except SyntaxError:
        pass

    return functions


def extract_functions_from_file(file_path):
    with open(file_path, "r") as file:
        return extract_functions(file.read())


def extract_function_signatures_from_file(file_path):
    with open(file_path, "r") as file:
        return extract_function_signatures(file.read())


class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}

    def visit_Name(self, node):
        # Add simple variable names with line numbers
        var_name = node.id
        line_number = node.lineno
        if var_name not in self.variables:
            self.variables[var_name] = set()
        self.variables[var_name].add(line_number)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Recursively build the attribute chain, e.g., x.y.z
        attributes = self.build_attribute_chain(node)
        attr_name = ".".join(attributes)
        line_number = node.lineno
        if attr_name not in self.variables:
            self.variables[attr_name] = set()
        self.variables[attr_name].add(line_number)
        self.generic_visit(node)

    def build_attribute_chain(self, node):
        # Recursively build the attribute chain
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            chain.append(node.id)
        return reversed(chain)  # Reverse to get the chain in the correct order


def extract_variables_from_file(file_path):
    with open(file_path, "r") as file:
        file_contents = convert_py2_to_py3(file.read())

    tree = ast.parse(file_contents)
    collector = VariableCollector()
    collector.visit(tree)

    return collector.variables


class FunctionCallCollector(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = []

    def visit_Call(self, node):
        self.function_calls.append(ast.unparse(node))
        self.generic_visit(node)


def convert_py2_to_py3(code):
    # Replace Python 2-specific Unicode raw string literals (Ur"string" or uR"string")
    # with raw string literals (r"string"), which are valid in Python 3.
    # This regex will match u/U followed by r/R and then a string literal.
    code = re.sub(r'u[rR]"', 'r"', code)
    code = re.sub(r"u[rR]'", "r'", code)
    code = re.sub(r'U[rR]"', 'r"', code)
    code = re.sub(r"U[rR]'", "r'", code)

    return code


code_to_node = {}


def get_function_data_from_file(filename, line_number):
    with open(filename, "r") as file:
        lines = file.readlines()
        code = convert_py2_to_py3("".join(lines))
        tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, "end_lineno") else None

            if (
                start_line
                <= line_number
                <= (end_line if end_line is not None else line_number)
            ):
                decorators = [
                    ast.unparse(decorator) for decorator in node.decorator_list
                ]

                # Adjust start_line to include decorators
                if node.decorator_list:
                    start_line = min(
                        decorator.lineno for decorator in node.decorator_list
                    )

                function_source = (
                    "".join(lines[start_line - 1 : end_line])
                    if end_line is not None
                    else lines[start_line - 1]
                )
                function_source = remove_indentation(function_source)
                code_to_node[function_source] = node
                variable_collector = VariableCollector()
                variable_collector.visit(node)
                variables = variable_collector.variables
                function_call_collector = FunctionCallCollector()
                function_call_collector.visit(node)
                function_calls = function_call_collector.function_calls

                # Extract docstring
                docstring = ast.get_docstring(node)

                return FunctionData(
                    start_line,
                    end_line,
                    node.name,
                    function_source,
                    variables,
                    function_calls,
                    decorators,
                    ast.unparse(node),
                    docstring,  # Add docstring to the return value
                )

    return FunctionData(
        None, None, None, None, {}, [], [], None, None
    )  # Add None for docstring in the default return


class ConstructorVariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                self.variables.add(target.attr)
        self.generic_visit(node)


def get_constructor_variable_names(class_node):
    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            collector = ConstructorVariableCollector()
            collector.visit(node)
            return collector.variables
    return set()


class ConstructorVariableStmtCollector(ast.NodeVisitor):
    def __init__(self):
        self.assignment_lines = []

    def visit_Assign(self, node):
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                # Store the entire line of assignment if the target is an instance attribute (self.variable)
                assignment_line = ast.unparse(node)
                self.assignment_lines.append(assignment_line)
        self.generic_visit(node)


def get_constructor_variables(class_node):
    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            collector = ConstructorVariableStmtCollector()
            collector.visit(node)
            return collector.assignment_lines
    return []


def get_class_level_variable_names(class_node):
    class_level_variables = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign):
            # Handle annotated assignments
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                type_hint = ast.unparse(node.annotation)
                class_level_variables[var_name] = type_hint
        elif isinstance(node, ast.Assign):
            # Handle regular assignments
            for target in node.targets:
                if isinstance(target, ast.Name):
                    class_level_variables[target.id] = None  # No type hint available

    return class_level_variables


def get_class_level_variables(class_node):
    class_level_assignments = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) or isinstance(node, ast.Assign):
            # Add the entire line of the assignment to the list
            assignment_line = ast.unparse(node)
            class_level_assignments.append(assignment_line)

    return class_level_assignments


def get_class_decorators(node: ast.ClassDef) -> List[str]:
    decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
    return decorators


def get_class_data(file_path: Path, lineno: int) -> Optional[ClassInfo]:
    with open(file_path, "r") as file:
        file_contents = file.read()

    tree = ast.parse(file_contents)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.lineno <= lineno <= (
            node.end_lineno or float("inf")
        ):
            class_snippet = ast.unparse(node)
            functions = extract_functions(class_snippet)
            function_signatures = extract_function_signatures(class_snippet)
            base_classes = [ast.unparse(b) for b in node.bases]
            class_signature = f"class {node.name}({', '.join(base_classes)})"
            class_docstring = ast.get_docstring(node)

            init_docstring = None
            constructor_variables = get_constructor_variables(node)
            constructor_variable_names = get_constructor_variable_names(node)
            class_level_variable_names = get_class_level_variable_names(node)
            class_level_variables = get_class_level_variables(node)
            class_decorators = get_class_decorators(node)
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode.name == "__init__":
                    init_docstring = ast.get_docstring(subnode)
                    break

            return ClassInfo(
                signature=class_signature,
                docstring=class_docstring,
                constructor_docstring=init_docstring,
                functions=functions,
                function_signatures=function_signatures,
                constructor_variables=list(constructor_variables),
                class_level_variables=list(class_level_variables),
                class_decorators=class_decorators,
                class_level_variable_names=list(class_level_variable_names),
                constructor_variable_names=list(constructor_variable_names),
            )

    return None


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


def find_encapsulating_node_range(tree, start_line, end_line):
    def find_node_containing_line(node, target_line):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            if node.lineno <= target_line <= node.end_lineno:
                return node
        for child in ast.iter_child_nodes(node):
            result = find_node_containing_line(child, target_line)
            if result:
                return result
        return None

    def find_smallest_enclosing_block(node):
        # Look specifically for assignment, import, loop, or if condition nodes
        while node:
            if isinstance(
                node,
                (ast.Assign, ast.Import, ast.ImportFrom, ast.For, ast.While, ast.If),
            ):
                return node
            node = getattr(node, "parent", None)
        return None

    def set_parents(node, parent=None):
        if not hasattr(node, "parent"):
            node.parent = parent
        for child in ast.iter_child_nodes(node):
            set_parents(child, node)

    set_parents(tree)

    start_node = find_node_containing_line(tree, start_line)
    end_node = find_node_containing_line(tree, end_line)

    start_block = find_smallest_enclosing_block(start_node) if start_node else None
    end_block = find_smallest_enclosing_block(end_node) if end_node else None

    overall_start = start_block.lineno if start_block else start_line
    overall_end = end_block.end_lineno if end_block else end_line

    return overall_start, overall_end


def expand_to_complete_structure(file_path, start_line, end_line):
    with open(file_path, "r") as file:
        source = file.read()

    tree = ast.parse(convert_py2_to_py3(source))
    expanded_start, expanded_end = find_encapsulating_node_range(
        tree, start_line, end_line
    )

    return expanded_start, expanded_end


def merge_non_function_lines(non_function_lines, repo_path):
    sorted_lines = sorted(non_function_lines, key=lambda x: (x[0], x[1]))
    snippet_ranges = {}

    for file_path, line_no in sorted_lines:
        if file_path not in snippet_ranges:
            snippet_ranges[file_path] = [(line_no, line_no)]
        else:
            intervals = snippet_ranges[file_path]
            last_start, last_end = intervals[-1]

            if not is_function_in_range(
                repo_path / Path(file_path), last_start, line_no
            ):
                intervals[-1] = (last_start, line_no)
            else:
                intervals.append((line_no, line_no))
    return snippet_ranges


def is_function_in_range(file_path, start_line, end_line):
    with open(file_path, "r") as file:
        source = file.read()

    tree = ast.parse(convert_py2_to_py3(source))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if hasattr(node, "lineno"):
                function_start = node.lineno
                function_end = (
                    node.end_lineno if hasattr(node, "end_lineno") else function_start
                )

                if start_line <= function_end and end_line >= function_start:
                    return True
    return False


def extract_functions_and_variables_from_file(features: Dict):
    files = features.keys()
    for file_path in files:
        if file_path == "test_data":
            continue
        functions = extract_functions_from_file(file_path)
        function_signatures = extract_function_signatures_from_file(file_path)
        variables = extract_variables_from_file(file_path)
        for var in variables:
            variables[var] = list(variables[var])
        features[file_path]["inscope_functions"] = functions
        features[file_path]["inscope_function_signatures"] = function_signatures
        features[file_path]["variables_in_file"] = variables
        features[file_path]["filtered_variables_in_file"] = filter_python_keywords(
            variables
        )
    return features
