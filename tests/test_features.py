import pytest
import ast
import re
from unittest.mock import mock_open, patch

from diff_utils import locations_from_single_file_diff


@pytest.mark.parametrize(
    "patch, expected",
    [
        (
            """diff --git a/cookiecutter/generate.py b/cookiecutter/generate.py
index 37365a4..c526b97 100644
--- a/cookiecutter/generate.py
+++ b/cookiecutter/generate.py
@@ -82,7 +82,7 @@ def generate_context(
     context = OrderedDict([])
 
     try:
-        with open(context_file) as file_handle:
+        with open(context_file, encoding='utf-8') as file_handle:
             obj = json.load(file_handle, object_pairs_hook=OrderedDict)
     except ValueError as e:
         # JSON decoding error.  Let's throw a new exception that is more
        """,
            [85, 86],
        ),
    ],
)
def test_extract_changed_line_numbers_from_patch(patch, expected):
    assert locations_from_single_file_diff(patch.split("\n")) == expected
