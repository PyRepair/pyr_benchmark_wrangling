import pytest
from pathlib import Path
from BugsInPy.run_custom_patch import get_diff 


@pytest.mark.parametrize("lineno, replacement_code, source_code, expected_diff", [
    (1, "def example_function():\n    print('hello world')\n", "def example_function():\n    print('hello world')\n", ""),
    (
        1, 
        "def example_function_modified():\n    print('goodbye world')\n", 
        "def example_function():\n    print('hello world')\n", 
        "--- Original\n+++ Replacement\n@@ -1,2 +1,2 @@\n-def example_function():\n-    print('hello world')\n+def example_function_modified():\n+    print('goodbye world')\n"
     ),
])
def test_get_diff(lineno, replacement_code, source_code, expected_diff):
    diff = get_diff(lineno, replacement_code, source_code)
    assert diff == expected_diff