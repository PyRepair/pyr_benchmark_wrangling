#! python3

import argparse
import json
import re
import sys
import os
import pprint as pp

from pathlib import Path
from typing import Dict, List, Tuple, Set

from BugsInPy.bgp import BugRecord

BUG_RECORDS = Path(__file__).parent / "bgp_bug_records.json"

# Kludge to import diff_utils package
current_script_directory = os.path.dirname(os.path.abspath(__file__))
prefix = os.path.split(current_script_directory)[0]
diff_utils_path = os.path.join(prefix, "diff_utils", "diff_utils.py")
sys.path.append(os.path.dirname(diff_utils_path))
import diff_utils as du


def construct_bugid_list_from_directory(input_directory: str) -> List[str]:
    bugid_list = []
    pattern = r".*:\d+$"

    with os.scandir(input_directory) as entries:
        for entry in entries:
            if entry.is_dir():
                bug_id = os.path.basename(entry.path)
                if re.match(pattern, bug_id):
                    bugid_list.append(bug_id)

    return bugid_list


def construct_bugid_list_from_argument(
    comma_delimited_bugid_list: str,
) -> List[str]:
    bugid_list = []
    pattern = r".*:\d+$"

    bug_ids = comma_delimited_bugid_list.split(",")
    for bug_id in bug_ids:
        if re.match(pattern, bug_id):
            bugid_list.append(bug_id)
        else:
            raise ValueError("Invalid bug_id in argument.")

    return bugid_list


def extract_localisation_set(
    bug_records: Dict[str, BugRecord], bugid_list: List[str]
) -> Dict[str, List[Tuple[str, int]]]:
    results = {}

    for bug_id in bugid_list:
        results[bug_id] = du.locations_from_diff(bug_records[bug_id]["fixing_patch"])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract localisation sets from BIP repos."
    )
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--input_directory", type=str, help="Path to the input directory"
    )
    group.add_argument(
        "--bugids",
        type=str,
        help="Comma-delimited list of bug ids (<string>:<int>)",
    )

    args = parser.parse_args()

    if args.input_directory and args.bugids:
        print(
            "Error: Both --input_directory and --bugids are specified."
            " Please choose one."
        )
        sys.exit()

    bugid_list = []
    if args.input_directory:
        bugid_list = construct_bugid_list_from_directory(args.input_directory)
    elif args.bugids:
        bugid_list = construct_bugid_list_from_argument(args.bugids)
    else:
        print("Invalid argument specified.")
        sys.exit()

    with open(BUG_RECORDS, "r") as bug_record_file:
        bug_records = json.load(bug_record_file)

    result = extract_localisation_set(bug_records, bugid_list)
    pp.pprint(result)


if __name__ == "__main__":
    main()
