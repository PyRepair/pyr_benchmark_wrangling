#!/usr/bin/env python3

import argparse
import csv
import difflib
from enum import Enum, auto
from pathlib import Path
import pprint as pp
import re
from statistics import mean, median
import sys
from typing import Any, Dict, List, Tuple, TypedDict, Set
import statistics


class HunkInfo(TypedDict):
    hunk_count: int
    gaps: List[int]
    max_hunk_gap: int
    mean_hunk_gap: float
    median_hunk_gap: float
    max_hunk_span: int


class MultiFileDiffInfo(TypedDict):
    diff_file_count: int
    diff_hunk_count: int
    diff_max_hunk_gap: List[float]
    diff_mean_hunk_gap: float
    diff_median_hunk_gap: float
    diff_max_hunk_span: int


# Precondition: diff_lines contains a unified diff for a single file AND
# the first diff operand is older in the version history than the second.
def measure_localisation_single_file_diff(diff_lines: List[str]) -> HunkInfo:
    """Compute localisation relevant measures for a single file's diff
    in a bug fix."""

    measures: HunkInfo = {
        "hunk_count": 1,
        "gaps": [],
        "max_hunk_gap": 0,
        "mean_hunk_gap": 0,
        "median_hunk_gap": 0,
        "max_hunk_span": 0,
    }

    pattern = re.compile(r"^@@ -(\d+),(\d+) \+\d+,\d+ @@.*$")
    hunk_intervals: List[Tuple[int, int]] = []

    for line in diff_lines:
        match_line = pattern.search(line)
        if match_line:
            start_line = int(match_line.group(1))
            length = int(match_line.group(2))
            hunk_intervals.append((start_line, start_line + length - 1))

    hunk_count = len(hunk_intervals)

    if hunk_count == 0:
        raise ValueError("diff_lines is not a unified diff.")

    if hunk_count > 1:
        measures["hunk_count"] = hunk_count
        gaps = [
            hunk_intervals[i + 1][0] - hunk_intervals[i][1]
            for i in range(hunk_count - 1)
        ]
        if not gaps:
            gaps = [0]
        measures["gaps"] = gaps
        measures["max_hunk_gap"] = max(gaps)
        measures["mean_hunk_gap"] = mean(gaps)
        measures["median_hunk_gap"] = median(gaps)
        measures["max_hunk_span"] = hunk_intervals[-1][1] - hunk_intervals[0][0]

    return measures


def dicts_to_lists(dict_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert a list of dicts into a Dict whose values are a list of the values
    bound to a key in the input list of dicts."""
    result_dict: Dict[str, List[Any]] = {}

    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)

    return result_dict


def measure_localisation_across_file_diffs(
    file_diffs: List[List[str]],
) -> MultiFileDiffInfo:
    """Compute bug measures across all files in a bug fix diff."""
    multifile_diff_info: MultiFileDiffInfo = {
        "diff_file_count": 1,
        "diff_hunk_count": 1,
        "diff_max_hunk_gap": [],
        "diff_max_hunk_span": 0,
        "diff_mean_hunk_gap": 0.0,
        "diff_median_hunk_gap": 0.0,
    }
    file_measures = []

    for file_diff in file_diffs:
        file_measures.append(measure_localisation_single_file_diff(file_diff))
    measures_dict = dicts_to_lists([dict(item) for item in file_measures])

    multifile_diff_info["diff_file_count"] = len(file_diffs)
    multifile_diff_info["diff_hunk_count"] = sum(measures_dict["hunk_count"])
    multifile_diff_info["diff_max_hunk_gap"] = max(measures_dict["max_hunk_gap"])
    multifile_diff_info["diff_mean_hunk_gap"] = mean(measures_dict["mean_hunk_gap"])
    multifile_diff_info["diff_median_hunk_gap"] = median(
        measures_dict["median_hunk_gap"]
    )
    multifile_diff_info["diff_max_hunk_span"] = max(measures_dict["max_hunk_span"])

    return multifile_diff_info


def measure_localisation_diff_lines(
    diff_lines: List[str],
) -> MultiFileDiffInfo:
    measures = measure_localisation_across_file_diffs(
        extract_single_file_diffs(diff_lines)
    )

    return measures


def measure_localisation_diff_file(diff_file: str | Path) -> MultiFileDiffInfo:
    with open(diff_file, "r") as f:
        diff_lines = f.readlines()

    measures = measure_localisation_across_file_diffs(
        extract_single_file_diffs(diff_lines)
    )

    return measures


def count_adds_dels_mods(diff_lines: List[str], k: int) -> Set[str]:
    """Count the adds, deletes, and modifications, where a modification is defined
    by Levenshtein edit distance less than k."""

    raise NotImplementedError()


def extract_modified_files(diff_lines: List[str]) -> List[str]:
    """Extract the names of the files modified in a git diff in git unified format."""

    modified_files = []
    pattern = re.compile(r"^diff --git a/(.+?)\s+b/.+$")

    for line in diff_lines:
        match_file = pattern.match(line)
        if match_file:
            modified_files.append(match_file.group(1))

    return modified_files


def extract_single_file_diffs(diff_lines: List[str]) -> List[List[str]]:
    """A git unified diff may contain diffs for a set of files. This method splits
    up a diff into a list of file-specific diffs."""

    single_file_diffs: List[List[str]] = []
    current_single_file_diff: List[str] = []

    for line in diff_lines:
        if line.startswith("diff --git a/"):
            if current_single_file_diff:
                single_file_diffs.append(current_single_file_diff.copy())
                current_single_file_diff = []
        current_single_file_diff.append(line)

    if current_single_file_diff:
        single_file_diffs.append(current_single_file_diff)

    return single_file_diffs


def locations_from_single_file_diff(diff_lines: List[str]) -> List[int]:
    """Convert a git unified diff for a single file into the set of modified
    line_numbers. For additions it adds the line numbers that bracket where the
    addition is made."""

    class States(Enum):
        HUNK = auto()
        ADDS = auto()
        DELS = auto()
        SHARED = auto()

    locations: Set[int] = set()
    i = 0
    current_state = States.HUNK
    pattern = re.compile(r"^@@ -(\d+),(\d+)")

    # Iterate through the input string
    for line in diff_lines[4:]:
        match line[0]:
            case "@":
                match_hunk = pattern.match(line)
                if match_hunk:
                    i = int(match_hunk.group(1))
                else:
                    raise ValueError("Invalid diff input.")
                current_state = States.HUNK
            case "+":
                if current_state != States.ADDS and not i - 1 in locations:
                    locations.add(i - 1)
                current_state = States.ADDS
            case "-":
                current_state = States.DELS
                locations.add(i)
                i += 1
            case " ":
                if current_state == States.ADDS:
                    locations.add(i)
                current_state = States.SHARED
                i += 1
            case _:
                raise ValueError("Invalid diff input.")

    return sorted(locations)


def locations_from_diff(diff_lines: List[str]) -> List[Tuple[str, int]]:
    """Convert a git unified diff into a set of (file, line_number) pairs
    for every location modified by the diff.

    Precondition:  diff_lines contains a unified diff.
    """

    location_set: List[Tuple[str, int]] = []
    for single_file_diff in extract_single_file_diffs(diff_lines):
        fname = extract_modified_files(single_file_diff)[0]
        locations = locations_from_single_file_diff(single_file_diff)
        for location in locations:
            location_set.append((fname, location))
    return location_set


def location_intervals_from_diff(
    diff_lines: List[str],
) -> List[Tuple[str, Tuple[int, int]]]:
    location_set = locations_from_diff(diff_lines)
    return convert_lines_by_file_to_intervals_by_file(location_set)


def convert_lines_by_file_to_intervals_by_file(
    location_set: List[Tuple[str, int]]
) -> List[Tuple[str, Tuple[int, int]]]:
    result: List[Tuple[str, Tuple[int, int]]] = []

    if not location_set:
        return result

    location_set.sort(key=lambda x: (x[0], x[1]))

    current_key, current_interval_start = location_set[0]

    if len(location_set) == 1:
        result.append((current_key, (current_interval_start, current_interval_start)))
        return result

    previous_value = current_interval_start
    for key, value in location_set[1:]:
        if key != current_key or value != previous_value + 1:
            result.append((current_key, (current_interval_start, previous_value)))
            current_key = key
            current_interval_start = value
        previous_value = value

    if key == current_key:
        result.append((current_key, (current_interval_start, value)))
    else:
        result.append((key, (value, value)))

    return result


def locations_from_diff_file(diff_file: str) -> List[Tuple[str, int]]:
    with open(diff_file, "r") as f:
        diff_lines = f.readlines()

    return locations_from_diff(diff_lines)


def write_localisation_measures_to_csv(data: Dict[str, Any], csv_file: str) -> None:
    flat_data = []

    for key, values in data.items():
        for i, (file, value) in enumerate(values, start=1):
            flattened_entry = {
                "file": key,
                f"location{i}_file": file,
                f"location{i}_value": value,
            }
            flat_data.append(flattened_entry)

    if not flat_data:
        return

    with open(csv_file, "w", newline="") as f:
        fieldnames = flat_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)


def write_location_set_to_csv(
    data: Dict[str, List[Tuple[str, int]]], csv_file: str
) -> None:
    flat_data = []

    for key, values in data.items():
        for i, (file, value) in enumerate(values, start=1):
            flattened_entry = {
                "file": key,
                f"location{i}_file": file,
                f"location{i}_value": value,
            }
            flat_data.append(flattened_entry)

    if not flat_data:
        return

    with open(csv_file, "w", newline="") as f:
        fieldnames = flat_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate localisation measures or extract the localisation set"
            " from diffs."
        )
    )
    parser.add_argument(
        "--measure",
        type=str,
        help="Comma-delimited list of diff files to measure",
    )
    parser.add_argument(
        "--locations",
        type=str,
        help="Comma-delimited list of diff files from which to extract location sets",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode (default: False)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diff_utils.csv",
        help="Output filename (csv)",
    )
    args = parser.parse_args()

    if args.measure:
        measurements: Dict[str, Any] = {}
        diff_files = args.measure.split(",")
        for diff_file in diff_files:
            measurements[diff_file] = measure_localisation_diff_file(diff_file)
        if args.output:
            write_localisation_measures_to_csv(measurements, args.output)
        if not args.quiet:
            for file, value in measurements.items():
                pp.pprint(f"Diff File: {file}\n{value}\n")
    elif args.locations:
        location_sets: Dict[str, List[Tuple[str, int]]] = {}
        diff_files = args.locations.split(",")
        for diff_file in diff_files:
            location_sets[diff_file] = locations_from_diff_file(diff_file)
        if args.output:
            write_location_set_to_csv(location_sets, args.output)
        if not args.quiet:
            for file, value in location_sets.items():
                pp.pprint(f"Diff File: {file}\n{value}\n")
    else:
        print("No action specified. Please use --measures or --locations.")
        return


if __name__ == "__main__":
    main()
