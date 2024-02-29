import ast
import builtins
import re
from typing import Dict, List, Optional
import ast
import keyword
from pathlib import Path
from dataclasses import dataclass, field


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
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not isinstance(node.parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature = get_signature(node)
                function_signatures.append(signature)

    except SyntaxError:
        pass

    return function_signatures


def extract_functions(code: str) -> List[str]:
    functions = []

    try:
        tree = ast.parse(code)
        # Attach parents to nodes
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not isinstance(node.parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
        for arg, default in zip(args, [""] * (len(args) - len(defaults)) + defaults)
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


def extract_functions_and_scope(code: str) -> Dict[str, List[Dict[str, str]]]:
    result = {"file_scope_functions": [], "file_scope_classes": []}

    try:
        tree = ast.parse(code)
        # Attach parents to nodes
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                # Ignore functions within classes, they'll be handled separately
                if isinstance(node.parent, ast.ClassDef) or isinstance(
                    node.parent, (ast.AsyncFunctionDef, ast.FunctionDef)
                ):
                    continue
                start_line = node.lineno
                end_line = node.end_lineno
                for decorator in node.decorator_list:
                    start_line = min(start_line, decorator.lineno)

                function_code_lines = code.splitlines()[start_line - 1 : end_line]
                function_code = "\n".join(function_code_lines)
                function_code = remove_indentation(function_code)
                signature = get_signature(node)

                result["file_scope_functions"].append(
                    {"code": function_code, "signature": signature}
                )

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "class_declaration": f"class {node.name}:",
                    "functions": [],
                }

                for child in node.body:
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and not isinstance(
                        child.parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        start_line = child.lineno
                        end_line = child.end_lineno
                        start_line = child.lineno
                        end_line = child.end_lineno
                        for decorator in child.decorator_list:
                            start_line = min(start_line, decorator.lineno)

                        function_code_lines = code.splitlines()[
                            start_line - 1 : end_line
                        ]
                        function_code = "\n".join(function_code_lines)
                        function_code = remove_indentation(function_code)
                        signature = get_signature(child)

                        class_info["functions"].append(
                            {"code": function_code, "signature": signature}
                        )

                result["file_scope_classes"].append(class_info)

    except SyntaxError:
        pass

    return result


def extract_function_code(node: ast.AST, code: str) -> str:
    """Extracts the code of a function or method from its AST node."""
    start_line = node.lineno
    # Check decorators for an earlier start line
    for decorator in node.decorator_list:
        start_line = min(start_line, decorator.lineno)
    end_line = node.end_lineno

    # Extract the function code from the original source
    function_code_lines = code.splitlines()[start_line - 1 : end_line]
    function_code = "\n".join(function_code_lines)
    function_code = remove_indentation(function_code)
    return function_code


def extract_functions_from_file(file_path):
    with open(file_path, "r") as file:
        return extract_functions_and_scope(file.read())


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


def filter_unused_imports(import_statements: List[str], code: str) -> List[str]:
    direct_imports = set()  # Direct imports (import math)
    attribute_imports = (
        {}
    )  # Attribute imports with optional aliases (from os import path as p)
    aliases = {}  # Aliases mapping

    for statement in import_statements:
        try:
            parsed = ast.parse(statement)
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        direct_imports.add(name.name)
                        if name.asname:
                            aliases[name.asname] = name.name
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imported_name = (
                            f"{node.module}.{name.name}" if node.module else name.name
                        )
                        attribute_imports[name.asname or name.name] = imported_name
                        if name.asname:
                            aliases[name.asname] = imported_name
        except SyntaxError:
            pass

    # Find used imports in the code
    used_imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in aliases:  # Check if the name is an alias
                    used_imports.add(aliases[node.id])
                elif (
                    node.id in attribute_imports
                ):  # Check if the name is from attribute import
                    used_imports.add(attribute_imports[node.id])
                elif node.id in direct_imports:  # Check if the name is a direct import
                    used_imports.add(node.id)
    except SyntaxError:
        pass

    # Filter and return the used import statements
    filtered_imports = []
    for stmt in import_statements:
        parsed = ast.parse(stmt)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in used_imports or name.asname in used_imports:
                        filtered_imports.append(stmt)
                        break
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    full_name = (
                        f"{node.module}.{name.name}" if node.module else name.name
                    )
                    if full_name in used_imports or name.asname in used_imports:
                        filtered_imports.append(stmt)
                        break
    return filtered_imports


def extract_import_statements(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()

    tree = ast.parse(file_content)
    import_statements = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_statement = ast.unparse(node)
            import_statements.append(import_statement)

    return import_statements


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
