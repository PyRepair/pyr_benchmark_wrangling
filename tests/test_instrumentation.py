import pytest
from unittest.mock import mock_open, patch
import shutil
import os
from pathlib import Path
from BugsInPy.feature_extractor.file_instrumenter import FileInstrumentor, VariableInfo

start = "start"
end = "end"


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=(
        "def test_func1():\n    a = 1\n    print(a)\ndef test_func2():\n    b = 2\n   "
        " print(b)"
    ),
)
def test_instrument(mock_file):
    instrumentor = FileInstrumentor("test_file.py")

    # Adding print points to multiple functions
    instrumentor.add_print_point("test_func1", 2, ["a"], start)
    instrumentor.add_print_point("test_func2", 2, ["b"], end)
    instrumentor.instrument()

    mock_file.assert_called()  # Check if open was called
    handle = mock_file()

    # Check the written content
    written_content = handle.write.call_args[0][0]
    assert written_content.count("fabcd_nonsense.close()") == 2


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="""
def _maybe_empty_lines(self, current_line: Line) -> Tuple[int, int]:
    max_allowed = 1
    if current_line.depth == 0:
        max_allowed = 2
    if current_line.leaves:
        # Consume the first leaf's extra newlines.
        first_leaf = current_line.leaves[0]
        before = first_leaf.prefix.count("")
        before = min(before, max_allowed)
        first_leaf.prefix = ""
    else:
        before = 0
    depth = current_line.depth
    while self.previous_defs and self.previous_defs[-1] >= depth:
        self.previous_defs.pop()
        before = 1 if depth else 2
    is_decorator = current_line.is_decorator
    if is_decorator or current_line.is_def or current_line.is_class:
        if not is_decorator:
            self.previous_defs.append(depth)
        if self.previous_line is None:
            # Don't insert empty lines before the first line in the file.
            return 0, 0
        if self.previous_line and self.previous_line.is_decorator:
            # Don't insert empty lines between decorators.
            return 0, 0

        if is_decorator and self.previous_line and self.previous_line.is_comment:
            # Don't insert empty lines between decorator comments.
            return 0, 0

        newlines = 2
        if current_line.depth:
            newlines -= 1
        return newlines, 0
    if current_line.is_flow_control:
        return before, 1
    if (
        self.previous_line
        and self.previous_line.is_import
        and not current_line.is_import
        and depth == self.previous_line.depth
    ):
        return (before or 1), 0
    if (
        self.previous_line
        and self.previous_line.is_yield
        and (not current_line.is_yield or depth != self.previous_line.depth)
    ):
        return (before or 1), 0
    return before, 0
""",
)
def test_instrument_large_function(mock_file):
    instrumentor = FileInstrumentor("test_file.py")
    vars = [
        "Line",
        "max_allowed",
        "current_line.depth",
        "current_line",
        "current_line.leaves",
        "first_leaf",
        "before",
        "first_leaf.prefix.count",
        "first_leaf.prefix",
        "depth",
        "self.previous_defs",
        "self",
        "self.previous_defs.pop",
        "is_decorator",
        "current_line.is_decorator",
        "current_line.is_def",
        "current_line.is_class",
        "self.previous_defs.append",
        "self.previous_line",
        "self.previous_line.is_decorator",
        "newlines",
        "current_line.is_flow_control",
        "self.previous_line.is_import",
        "current_line.is_import",
        "self.previous_line.depth",
        "self.previous_line.is_yield",
        "current_line.is_yield",
        "Tuple",
    ]
    start = "start"
    end = "end"
    # Adding multiple print points
    instrumentor.add_print_point("_maybe_empty_lines", 2, vars, start)
    instrumentor.add_print_point("_maybe_empty_lines", 21, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 22, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 23, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 24, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 27, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 31, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 36, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 38, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 45, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 51, vars, end)
    instrumentor.add_print_point("_maybe_empty_lines", 52, vars, end)

    instrumentor.instrument()

    mock_file.assert_called()  # Check if open was called
    handle = mock_file()

    # Check the written content
    written_content = handle.write.call_args[0][0]
    # Assert that the expected print points are inserted
    # The exact content to assert will depend on your specific implementation
    print(written_content)
    assert written_content.count("fabcd_nonsense.close()") == 18
