import pytest
from diff_utils import convert_lines_by_file_to_intervals_by_file


@pytest.mark.parametrize(
    "location_set, expected_result",
    [
        (
            [
                ("X", 2),
                ("Y", 5),
                ("X", 3),
                ("X", 4),
                ("Y", 6),
                ("X", 7),
                ("Y", 8),
                ("Z", 1),
            ],
            [
                ("X", (2, 4)),
                ("X", (7, 7)),
                ("Y", (5, 6)),
                ("Y", (8, 8)),
                ("Z", (1, 1)),
            ],
        ),
        (
            [
                ("fileA", 1),
                ("fileB", 2),
                ("fileB", 3),
                ("fileC", 1),
                ("fileC", 2),
            ],
            [("fileA", (1, 1)), ("fileB", (2, 3)), ("fileC", (1, 2))],
        ),
        ([("fileA", 3), ("fileA", 1), ("fileA", 2)], [("fileA", (1, 3))]),
        (
            [("fileA", 1), ("fileA", 1000)],
            [("fileA", (1, 1)), ("fileA", (1000, 1000))],
        ),
        ([("fileA", 1)], [("fileA", (1, 1))]),
    ],
)
def test_convert_lines_by_file_to_intervals_example2(location_set, expected_result):
    result = convert_lines_by_file_to_intervals_by_file(location_set)
    assert result == expected_result
