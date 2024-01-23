import pytest
from BugsInPy.feature_extractor.static_features import filter_unused_imports  # Replace with the actual module name

@pytest.mark.parametrize("imports, code, expected", [
    (["import math", "import os", "from collections import Counter"], 
     """
def example():
    x = math.sqrt(16)
    y = Counter([1, 2, 2, 3, 3, 3])
    return x, y
""",
     ["import math", "from collections import Counter"]),
    (["import math", "import os", "from collections import Counter as C"], 
     """
def example():
    x = math.sqrt(16)
    y = C([1, 2, 2, 3, 3, 3])
    return x, y
""",
     ["import math", "from collections import Counter as C"]),

    (["import math", "import os"], 
     """
def example():
    return 'No imports used'
""",
     []),

    (["import math as m", "import os"], 
     """
def example():
    x = m.sqrt(16)
    return x
""",
     ["import math as m"]),

    (["import math", "import os"], 
     """
def example():
    # Example comment: import math
    return "import os"
""",
     []),

    # Test with multiline import
    (["from collections import (Counter, defaultdict)", "import sys"], 
     """
def process():
    c = Counter()
    return c
""",
     ["from collections import (Counter, defaultdict)"]),

    # Test with complex code and multiple used imports
    (["import math", "import json", "from os import path, environ"], 
     """
def complex_function():
    value = math.pi
    file_path = path.join('directory', 'filename')
    config = json.loads('{}')
    return value, file_path, config
""",
     ["import math", "import json", "from os import path, environ"]),

    # Add more test cases as needed
])
def test_filter_unused_imports(imports, code, expected):
    assert filter_unused_imports(imports, code) == expected