import os
import collections
import json
import logging
import shutil

from pathlib import Path
from typing import Set, List, Dict

from BugsInPy.bgp_config import BGPConfig


BASE_DIR = BGPConfig.BIP_PROJECTS_DIR


def filter_requirements(delimiter, requirements):
    # Filter out comments and blank lines
    return [
        line
        for line in requirements.split(delimiter)
        if line.strip() and not line.startswith("#")
    ]


def read_requirements_file(path):
    with open(path, "rb") as file:
        content_bytes = file.read()
        try:
            requirements = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            requirements = content_bytes.decode("utf-16")
            requirements = requirements.replace("\r\n", "\n")

    # Filter out comments and blank lines, then sort
    requirements = filter_requirements("\n", requirements)
    reqs = sorted(
        line.strip()
        for line in requirements
        if line.strip() and not line.startswith("#")
    )
    repo_name = path.split(os.path.sep)[-4]
    reqs = [req for req in reqs if not (req.startswith("-e") and repo_name in req)]
    return tuple(reqs)


def gather_requirements():
    requirements_dict = collections.defaultdict(list)

    # Enumerate through each project and its bugs
    for project_name in os.listdir(BASE_DIR):
        project_path = os.path.join(BASE_DIR, project_name, "bugs")

        if os.path.isdir(project_path):
            for bug_id in os.listdir(project_path):
                req_file = os.path.join(project_path, bug_id, "requirements.txt")

                if os.path.isfile(req_file):
                    reqs = read_requirements_file(req_file)
                    # Use the requirements as a key and append the project and bug ID as a tuple
                    requirements_dict[reqs].append((project_name, bug_id))

    return requirements_dict


def save_classes_to_files(classes, base_directory):
    # Ensure the base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # This will store the mapping "repo:bug" to className
    repo_bug_to_classname = {}

    # Iterate over the classes dictionary
    for idx, (reqs, projects_and_bugs) in enumerate(classes.items()):
        # Create a dictionary for the current class with its requirements
        class_info = {"class_index": idx, "requirements": reqs}
        # Save the class info to a JSON file
        class_filename = f"class_{idx}.json"
        class_filepath = base_directory / class_filename
        with open(class_filepath, "w") as class_file:
            json.dump(class_info, class_file, indent=4)

        # For each project and bug, map to the current class index
        for project, bug in projects_and_bugs:
            key = f"{project}:{bug}"
            repo_bug_to_classname[key] = idx

    # Save the repo_bug_to_classname mapping to a JSON file
    mapping_filename = "repo_bug_to_classname_mapping.json"
    mapping_filepath = base_directory / mapping_filename
    with open(mapping_filepath, "w") as mapping_file:
        json.dump(repo_bug_to_classname, mapping_file, indent=4)


def merge_classes(classes):
    """
    Merge classes of requirements to minimize the number of unique classes.
    :param classes: Dictionary with tuple of requirements as key and associated bugs as value.
    :return: Dictionary of merged classes.
    """
    # Convert classes dictionary to a list of sets for easier merging
    class_sets = [set(reqs) for reqs in classes.keys()]

    # Sort classes based on size
    sorted_classes = sorted(class_sets, key=len, reverse=True)

    merged_classes: List[Set[str]] = sorted_classes
    """
    while sorted_classes:
        current_class = sorted_classes.pop()
        was_merged = False

        for i, other_class in enumerate(merged_classes):
            # If current_class is a subset of another, no need to create a new class.
            if current_class.issubset(other_class):
                was_merged = True
                break
            # If another_class is a subset of current_class, replace it.
            elif other_class.issubset(current_class):
                merged_classes[i] = current_class
                was_merged = True
                break

        if not was_merged:
            merged_classes.append(current_class)
    """
    # Convert back to the original format (dictionary with tuple as key)
    merged_dict = {}
    for merged_class in merged_classes:
        merged_key = tuple(sorted(merged_class))
        # Combine the associated projects_and_bugs lists for the merged classes
        merged_value = []
        for reqs, projects_and_bugs in classes.items():
            if set(reqs).issubset(merged_class):
                merged_value.extend(projects_and_bugs)

        # Remove duplicates and keep unique values
        merged_value = list(set(merged_value))
        merged_dict[merged_key] = merged_value

    return merged_dict


def requirements_diff(reqs1, reqs2):
    # Create dictionaries from requirements
    reqs_dict1 = {
        req.split("==")[0]: req.split("==")[1] if "==" in req else None for req in reqs1
    }
    reqs_dict2 = {
        req.split("==")[0]: req.split("==")[1] if "==" in req else None for req in reqs2
    }

    only_in_reqs1 = {k: v for k, v in reqs_dict1.items() if k not in reqs_dict2}
    only_in_reqs2 = {k: v for k, v in reqs_dict2.items() if k not in reqs_dict1}
    version_diffs = {
        k: (reqs_dict1[k], reqs_dict2[k])
        for k in reqs_dict1
        if k in reqs_dict2 and reqs_dict1[k] != reqs_dict2[k]
    }

    return {
        "only_in_reqs1": only_in_reqs1,
        "only_in_reqs2": only_in_reqs2,
        "version_diffs": version_diffs,
    }


"""
def merge_similar_classes(classes):
    merged_classes: Dict = {}
    for reqs, projects_and_bugs in classes.items():
        package_set = set(
            [req.split("==")[0] if "==" in req else req.split(">=")[0] for req in reqs]
        )
        found = False
        for mreqs, mprojects_and_bugs in merged_classes.items():
            mpackage_set = set(
                [
                    req.split("==")[0] if "==" in req else req.split(">=")[0]
                    for req in mreqs
                ]
            )
            if package_set == mpackage_set:
                merged_classes[mreqs].extend(projects_and_bugs)
                found = True
                break
        if not found:
            merged_classes[reqs] = projects_and_bugs
    return merged_classes
"""


def extract_classes(extracted_path=None):
    try:
        shutil.rmtree(BGPConfig.BIP_ENVIRONMENT_CLASSES)
        shutil.rmtree(BGPConfig.BIP_ENVIRONMENT_DIR)
    except OSError as e:
        print(f"An exception occured when removing directories: {e}")
    classes = gather_requirements()
    classes = merge_classes(classes)
    total_unique_classes = len(classes)
    logging.debug(f"Total unique classes of requirements: {total_unique_classes}\n")

    # Initialize a default dictionary to hold project stats
    project_stats: Dict = collections.defaultdict(
        lambda: {"bug_count": 0, "unique_reqs_classes": set()}
    )
    seen_bugs = set()

    for reqs, projects_and_bugs in classes.items():
        reqs_tuple = tuple(sorted(reqs))
        for project, bug in projects_and_bugs:
            # Create a unique identifier for the bug across all projects
            unique_bug_identifier = (project, bug)
            if unique_bug_identifier not in seen_bugs:
                # Increment bug count for the project
                project_stats[project]["bug_count"] += 1
                # Mark this bug as seen
                seen_bugs.add(unique_bug_identifier)
            # Add the hashable tuple of requirements to the set for this specific project
            project_stats[project]["unique_reqs_classes"].add(reqs_tuple)

    for project, stats in project_stats.items():
        total_bugs = stats["bug_count"]
        unique_req_classes = len(stats["unique_reqs_classes"])
        logging.debug(f"Project: {project}")
        logging.debug(f"  - Total bugs: {total_bugs}")
        logging.debug(
            f"  - Number of unique requirement classes: {unique_req_classes}\n"
        )
        save_classes_to_files(classes, extracted_path)
