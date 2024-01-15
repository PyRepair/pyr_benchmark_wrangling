#! /usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Dict, List

import pprint as pp


def sample_bugs(dictionary: Dict[str, Any], k: int) -> List[str]:
    return random.sample(list(dictionary.keys()), k)


def main():
    json_file = (
        Path(os.path.dirname(os.path.abspath(__file__))) / "bgp_bug_records.json"
    )
    parser = argparse.ArgumentParser(description="Sample BIP bugs.")
    parser.add_argument(
        "--json_file",
        type=str,
        default=json_file,
        help="Path to the BGP json file",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Number of bugs to sample",
    )
    args = parser.parse_args()

    with open(args.json_file, "r") as file:
        data: Dict[str, Any] = json.load(file)

    sampled_bugs: List[str] = sample_bugs(data, args.sample_size)

    pp.pprint(sampled_bugs)


if __name__ == "__main__":
    main()
