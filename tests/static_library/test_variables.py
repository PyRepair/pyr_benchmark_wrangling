import pytest
from static_library import extract_variables_from_file

@pytest.mark.parametrize("file_contents,expected_variables", [
    # Test with a simple variable
    (
        "a = 1\nb = 2\na = 3",  # file_contents
        {'a': {1, 3}, 'b': {2}}  # expected_variables
    ),
    # Test with attributes
    (
        "obj.attr = 1\nobj.attr2 = obj.attr\nobj2 = obj.attr2",
        {'obj': {1, 2, 3}, 'obj.attr': {1, 2}, 'obj.attr2': {2, 3}, 'obj2': {3}}
    ),
    # Test with nested attributes
    (
        "x.y.z = 1\nx.y = x.y.z",
        {'x': {1, 2}, 'x.y.z': {1, 2}, 'x.y': {1, 2}}
    ),
])
def test_variable_collector(file_contents, expected_variables, mocker):
    # Mock open to return file_contents
    mocker.patch("builtins.open", mocker.mock_open(read_data=file_contents))

    # Call the function to test
    variables = extract_variables_from_file("dummy_path.py")
    # Assert the expected outcome
    assert variables == expected_variables