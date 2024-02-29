from BugsInPy.test_runner import move_test_file


from .dynamic_features import extract_dynamic_features
from .static_features import (
    extract_buggy_function,
    extract_functions_and_variables_from_file,
)


def extract_features(bug_id, bug_record, repo_path, separate_envs):
    buggy_commit_id = bug_record["buggy_commit_id"]
    move_test_file(bug_id, bug_record, repo_path, separate_envs)

    features = extract_buggy_function(bug_record, repo_path)
    features = extract_functions_and_variables_from_file(features)
    features = extract_dynamic_features(
        features, bug_record, repo_path, bug_id, separate_envs
    )
    return features
