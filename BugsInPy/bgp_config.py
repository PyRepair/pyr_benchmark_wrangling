from pathlib import Path
import os
from dataclasses import dataclass
from typing import TypedDict, List


@dataclass
class BGPConfig:
    """Configuration settings for this script."""

    HOME_DIR = Path(os.path.expanduser("~"))
    BIP_BENCHMARKS_DIR = HOME_DIR / ".abw" / "BugsInPy_Dir"
    BIP_ROOT = BIP_BENCHMARKS_DIR / "BugsInPy"
    BIP_CLONED_REPOS = BIP_BENCHMARKS_DIR / "BugsInPy_Cloned_Repos"
    BIP_PROJECTS_DIR = BIP_ROOT / "projects"
    BUG_RECORDS = BIP_BENCHMARKS_DIR / "bgp_bug_records.json"
    BIP_ENVIRONMENT_CLASSES = BIP_BENCHMARKS_DIR / "classes"
    BIP_ENVIRONMENT_DIR = BIP_BENCHMARKS_DIR / "envs"

    TEST_STATUS_RECORDS = BIP_BENCHMARKS_DIR / "bgp_test_status_records.json"
    BIP_GIT_URL = "https://github.com/soarsmu/BugsInPy.git"
    # Testing utilities, notably pytest, can, and often do, run multiple tests.
    # We would like to separately timeout each test, but subprocess does not
    # afford this granularity.  Our kludge is to multiply the base timeout on
    # these subprocess run calls, by this factor.
    TIMEOUT_MULTIPLIER = 100


class BugRecord(TypedDict):
    benchmark_url: str
    bip_python_version: str
    buggy_commit_id: str
    failing_test_command: str
    fixed_commit_id: str
    fixing_patch: List[str]
    test_file: str
