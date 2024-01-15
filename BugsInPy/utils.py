from BugsInPy.exceptions import InvalidExecutionOrderError
from BugsInPy.bgp_config import BugRecord, BGPConfig
import logging
import subprocess
import os
import shutil

from pathlib import Path
from typing import Dict, Any

try:
    from git import (
        Repo,
        NoSuchPathError,
        InvalidGitRepositoryError,
        GitCommandError,
    )

    GIT_INSTALLED = True
except ImportError:
    GIT_INSTALLED = False


def is_git_repo(path: Path) -> bool:
    """Checks if the given path is a Git repository."""
    try:
        _ = Repo(path)
        return True
    except InvalidGitRepositoryError:
        return False


def is_subdirectory(path: Path) -> bool:
    """Check whether a path is a subdirectory with the BGP clone."""
    path = Path(path).resolve()
    return path.is_relative_to(BGPConfig.BIP_CLONED_REPOS.resolve())


def delete(repo_path: Path) -> None:
    """Remove a benchmark repo from BGP's output directory."""
    assert is_subdirectory(repo_path)
    if repo_path.exists():
        shutil.rmtree(repo_path)


def clone(bug_id: str, repo_path: Path, bug_record: BugRecord, restart=False) -> None:
    """Clones a BGP bug into repo_path."""

    github_url = bug_record["benchmark_url"]
    if not github_url:
        raise EnvironmentError("GitHub URL information not found or invalid.")
    if restart and os.path.exists(repo_path):
        delete(repo_path)
    if os.path.exists(repo_path):
        if is_git_repo(repo_path):
            logging.warning(f"Repository already exists and is a Git repo: {repo_path}")
        else:
            logging.warning(
                f"Path already exists but is not a Git repo, cloning into: {repo_path}"
            )
            git_clone(github_url, repo_path)
    else:
        logging.info(f"Cloning repository into: {repo_path}")
        git_clone(github_url, repo_path)


def git_clone(github_url: str, repo_path: Path) -> None:
    """Clone GitHub repository into repo_path."""

    try:
        subprocess.run(["git", "clone", github_url, repo_path], check=True)
    except subprocess.CalledProcessError as e:  # pylint: disable=invalid-name
        error_message = e.stderr.decode("utf-8") if e.stderr else "Unknown error"
        raise EnvironmentError(
            f"Error during git clone operation: {error_message}"
        ) from e
    logging.info(f"Cloned repository: {github_url}")


def checkout(
    bug_id: str, repo_path: Path, commit_id: str, separate_envs: bool = False
) -> None:
    """Checkout the specified buggy commit from the repo at repo_path."""
    try:
        repo = Repo(repo_path)
    except NoSuchPathError as e:
        msg = (
            f"Please clone {bug_id}'s repository before running"
            " either the `checkout` or `prep` commands."
        )
        raise InvalidExecutionOrderError(msg) from e

    # Reset all tracked files to match the index state
    repo.git.reset("--hard")

    if not separate_envs:
        # Only force clean if reusing environments
        # This removes untracked files and directories
        repo.git.clean("-f", "-d")

    try:
        repo.git.checkout(commit_id, force=True)
    except GitCommandError as e:
        logging.error(
            f"Failed to checkout commit {commit_id} even after resetting. Error: {e}"
        )
