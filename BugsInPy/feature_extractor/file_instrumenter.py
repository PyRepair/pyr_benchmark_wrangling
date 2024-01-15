from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import xml.etree.ElementTree as ET


from BugsInPy.bgp_config import BGPConfig

import ast
import json
import shutil
import os


@dataclass
class VariableInfo:
    variable_value: str
    variable_type: str
    variable_shape: str


def create_attribute_check(var):
    parts = var.split(".")
    allowed_len = "(str, list, tuple, set, dict, frozenset, bytes, bytearray, range)"
    if len(parts) == 1:
        # It's a simple variable, not an attribute
        list_check = (
            f" + '|' + (str(len({var})) if hasattr({var}, '__len__') and "
            f" isinstance({var}, {allowed_len}) else '')"
        )
        shape_check = (
            f" + '|' + (repr(getattr({var}, 'shape', 'No shape')) if 'shape' in"
            f" dir({var}) else '')"
        )
        return (
            f"\"{var}\": (type({var}).__name__ + '|' +"
            f" repr({var}){list_check}{shape_check} if '{var}' in locals() else 'None')"
        )
    else:
        # It's an attribute access, like a.b
        attr_expression = f"getattr({parts[0]}, '{'.'.join(parts[1:])}', None)"
        list_check = (
            f" + '|' + (str(len({attr_expression})) if"
            f" hasattr({attr_expression},'__len__') and "
            f" isinstance({attr_expression}, {allowed_len}) else '')"
        )
        shape_check = (
            f" + '|' + (repr(getattr({attr_expression}, 'shape')) if"
            f" hasattr({attr_expression}, 'shape') else '')"
        )
        return (
            f"\"{var}\": (type({attr_expression}).__name__ + '|' +"
            f" repr({attr_expression}){list_check}{shape_check} if '{parts[0]}' in"
            " locals() else 'None')"
        )


class InstrumentationTransformer(ast.NodeTransformer):
    default_instrumentation_output = BGPConfig.BIP_ROOT / "instrumentation.json"

    def __init__(self, print_points, filename=default_instrumentation_output):
        self.print_points = print_points
        self.filename = filename
        # Ensure the JSON file exists
        self.create_rewrite_json()
        self.instrumented_nodes = set()

    def create_rewrite_json(self):
        with open(self.filename, "w") as file:
            json.dump({}, file)  # Create an empty JSON object in the file

    def visit_Module(self, node):
        # Find the position after any __future__ imports
        future_imports = [
            i
            for i, n in enumerate(node.body)
            if isinstance(n, ast.ImportFrom) and n.module == "__future__"
        ]
        insert_position = future_imports[-1] + 1 if future_imports else 0

        # Insert the json import
        json_import = ast.Import(
            names=[ast.alias(name="json", asname="json_no_collision")]
        )
        node.body.insert(insert_position, json_import)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)  # Visit all child nodes
        new_node = self.instrument_function(node)  # Apply instrumentation to this node
        return new_node or node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)  # Visit all child nodes
        new_node = self.instrument_function(node)  # Apply instrumentation to this node
        return new_node or node

    def instrument_function(self, node):
        if node.name not in self.print_points:
            return None

        for line, variables, program_point in self.print_points[node.name]:
            # Check if the line is within the node's range

            if not (node.lineno <= line <= getattr(node, "end_lineno", node.lineno)):
                continue

            instrumentation_code = self.create_instrumentation_code(
                node, variables, line, program_point
            )
            instrumentation_stmt = ast.parse(instrumentation_code)

            # Apply instrumentation
            if "end_2" in program_point:
                node.body.extend(instrumentation_stmt.body)
            elif "start" in program_point:
                node.body[:0] = instrumentation_stmt.body
            else:
                node = self.insert_before_return(node, line, instrumentation_stmt)

        return node

    def insert_before_return(self, node, line, instrumentation_stmt):
        """
        Insert instrumentation statement before a return statement at a specific line.
        """

        # Define an inner visitor class to handle the insertion
        class InsertBeforeReturnVisitor(ast.NodeTransformer):
            def visit_Return(self, return_node):
                if return_node.lineno == line:
                    # Insert instrumentation before the return statement
                    return [instrumentation_stmt, return_node]
                return return_node

        visitor = InsertBeforeReturnVisitor()
        return visitor.visit(node)

    def insert_instrumentation(self, node, line, instrumentation_stmt, program_point):
        """
        Recursively insert instrumentation statement into the AST node at the specified line.
        Handles cases where the 'line' might point to a decorator of a function.
        """

        # Avoid instrumenting the same node multiple times
        if (line, program_point) in self.instrumented_nodes:
            return node

        # Check if the node is at the specified line
        if hasattr(node, "lineno"):
            if node.lineno == line:
                # Insert instrumentation statement before the node
                self.instrumented_nodes.add((line, program_point))
                return [instrumentation_stmt, node]

        if isinstance(node, ast.If):
            # Process the 'body' of the if
            node.body = self.process_nested_blocks(
                node.body, line, instrumentation_stmt, program_point
            )

            # Process the 'orelse' part (which covers elif and else)
            node.orelse = self.process_nested_blocks(
                node.orelse, line, instrumentation_stmt, program_point
            )

        # Recursively handle child nodes in a function body or similar structure
        elif hasattr(node, "body") and isinstance(node.body, list):
            new_body = []
            for child in node.body:
                result = self.insert_instrumentation(
                    child, line, instrumentation_stmt, program_point
                )
                if result:
                    new_body.extend(result if isinstance(result, list) else [result])
                else:
                    new_body.append(child)
            node.body = new_body
            return node

        return node

    def process_nested_blocks(self, block, line, instrumentation_stmt, program_point):
        new_block = []
        for child in block:
            result = self.insert_instrumentation(
                child, line, instrumentation_stmt, program_point
            )
            if result:
                new_block.extend(result if isinstance(result, list) else [result])
            else:
                new_block.append(child)
        return new_block

    def create_instrumentation_code(self, node, variables, line, program_point):
        # Replace this with actual code generation logic based on the line number and node information
        # Generate dictionary content for the specified variables
        dict_content = (
            "{" + ", ".join([create_attribute_check(var) for var in variables]) + "}"
        )

        # Differentiate between start and end program points
        if "start" == program_point:
            # Store the start data
            # Use nonsense names to prevent collisions
            # fabcd_nonsense: File pointer
            # data_bcd_nonsense_32: The json dict
            # entry_365: The entry IO tuple to modify
            instrumentation_code = (
                f'fabcd_nonsense = open("{self.filename}", "r");'
                "data_bcd_nonsense_32 = json_no_collision.load(fabcd_nonsense); "
                "fabcd_nonsense.close(); "
                f"start_databcd_nonsense_32 = {dict_content}; "
                f'data_bcd_nonsense_32["{program_point}"] = start_databcd_nonsense_32; '
                f'fabcd_nonsense = open("{self.filename}", "w"); '
                "json_no_collision.dump(data_bcd_nonsense_32, fabcd_nonsense); "
                "fabcd_nonsense.close()"
            )
        elif "end" in program_point:
            # Store the tuple (start_data, end_data) when end is reached
            instrumentation_code = (
                f'fabcd_nonsense = open("{self.filename}", "r");data_bcd_nonsense_32 ='
                " json_no_collision.load(fabcd_nonsense); fabcd_nonsense.close();"
                " start_databcd_nonsense_32 = data_bcd_nonsense_32.get('start',"
                " {}); entry_365 = data_bcd_nonsense_32.get('data', []);"
                f" entry_365.append((start_databcd_nonsense_32, {dict_content}));"
                ' data_bcd_nonsense_32["data"] = entry_365; fabcd_nonsense ='
                f' open("{self.filename}", "w");'
                " json_no_collision.dump(data_bcd_nonsense_32, fabcd_nonsense);"
                " fabcd_nonsense.close()"
            )
        else:
            raise ValueError("Invalid program point specified")

        return instrumentation_code


class FileInstrumentor:
    def __init__(self, filename):
        self.filename = filename
        self.backup_filename = str(filename) + ".bak"
        self.print_points: Dict[str, Dict[int, Tuple[Set[str], str]]] = {}

    def add_print_point(self, func_name, line_number, variables, program_point):
        """
        Add a print point in a function at a specific line number with variables to print.
        """
        if func_name not in self.print_points:
            self.print_points[func_name] = {}
        self.print_points[func_name][line_number] = (variables, program_point)

    def backup_original_file(self):
        """
        Creates a backup of the original file.
        """
        shutil.copyfile(self.filename, self.backup_filename)

    def instrument(self):
        """
        Reads the file, converts it to AST, inserts print statements for variables, and overwrites the original file.
        """
        self.backup_original_file()

        with open(self.filename, "r") as file:
            source_code = file.read()

        # Parse the source code into an AST
        tree = ast.parse(source_code)

        # Prepare print points for AST transformation
        print_points_ast: Dict[str, List[Tuple[int, Set[str], str]]] = {}
        for func_name, line_vars in self.print_points.items():
            for line, vars_data in line_vars.items():
                # Assuming func_name is already in the correct format
                if func_name not in print_points_ast:
                    print_points_ast[func_name] = []
                print_points_ast[func_name].append((line, vars_data[0], vars_data[1]))

        # Apply the instrumentation
        transformer = InstrumentationTransformer(print_points_ast)
        transformed_tree = transformer.visit(tree)

        # Convert the AST back to source code
        instrumented_code = ast.unparse(transformed_tree)
        # Overwrite the original file with the instrumented code
        with open(self.filename, "w") as file:
            file.write(instrumented_code)

    @staticmethod
    def process_data(data):
        processed_data = {}
        for var_name, var_details in data.items():
            details = var_details.split("|", 3)  # Split into at most 4 parts

            var_type = details[0]
            var_value = details[1] if len(details) > 1 else None
            var_shape = details[2] if len(details) > 2 else None
            var_len = details[3] if len(details) > 3 else None
            if var_shape == "":
                var_shape = None
            if var_len == "":
                var_len = None
            var_shape = var_shape or var_len
            # Create VariableInfo instance and convert it to a dictionary
            processed_data[var_name] = asdict(
                VariableInfo(
                    variable_value=var_value,
                    variable_type=var_type,
                    variable_shape=var_shape,
                )
            )
        return processed_data

    def extract_variable_info_from_json(
        self, json_file: Path = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Extracts variable information from a JSON file produced by instrumentation.

        Args:
        json_file: Path to the JSON file.

        Returns:
        Dict[str, Dict[str, VariableInfo]]: A dictionary with line numbers as keys and variable information as values.
        """

        if json_file is None:
            json_file = InstrumentationTransformer.default_instrumentation_output
        with open(json_file, "r") as file:
            json_data = json.load(file)
            data = json_data.get("data", {})
            input = json_data.get("start", {})

        if data == {}:
            return [(FileInstrumentor.process_data(input), {})]

        extracted_data = []
        for input, output in data:
            processed_input = FileInstrumentor.process_data(input)
            processed_output = FileInstrumentor.process_data(output)
            extracted_data.append((processed_input, processed_output))

        return extracted_data

    def restore_original_file(self):
        """
        Restores the original file from the backup.
        """
        shutil.move(self.backup_filename, self.filename)
