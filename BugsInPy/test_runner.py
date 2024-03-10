from cgi import test
import configparser
import json
import logging
import os
import subprocess
import shutil
import sys

from pathlib import Path
from os.path import commonpath
from random import randint
from typing import Dict, List, Optional, Tuple
from BugsInPy.bgp_config import BGPConfig, BugRecord
from BugsInPy.utils import checkout, clone
from enum import Enum
from BugsInPy.exceptions import InvalidExecutionOrderError

SUPPORTED_OS = ["darwin", "linux", "win32"]


class TestStatus(Enum):
    """Enum for test results."""

    PASS = 0
    FAIL = 1
    INTERRUPTED = 2
    INTERNAL_ERROR = 3
    USAGE_ERROR = 4
    NO_TESTS_COLLECTED = 5
    UNKNOWN_ERROR = 6


def get_test_status_info(return_code) -> Tuple[bool, TestStatus]:
    """Convert return code into TestStatus."""
    test_run_status_mapping = {
        0: (True, TestStatus.PASS),
        1: (True, TestStatus.FAIL),
        2: (False, TestStatus.INTERRUPTED),
        3: (False, TestStatus.INTERNAL_ERROR),
        4: (False, TestStatus.USAGE_ERROR),
        5: (False, TestStatus.NO_TESTS_COLLECTED),
    }

    return test_run_status_mapping.get(return_code, (False, TestStatus.UNKNOWN_ERROR))


def create_virtualenv(
    repo_path: Path, python_version: str, timeout: int
) -> Tuple[str, str]:
    """Create a virtual environment for the specified repo."""

    if len(python_version.split(".")) >= 3:
        python_version = ".".join(python_version.split(".")[:2])
    venv_path = repo_path
    if venv_path.exists():
        try:
            shutil.rmtree(venv_path)
        except:
            # Some files, such as links, cause exceptions
            pass

    python_binary = install_python(python_version, timeout)

    try:
        subprocess.run(
            [python_binary, "-m", "venv", venv_path],
            check=True,
            timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:  # pylint: disable=invalid-name
        logging.error(f"Error: {str(e)}")
    logging.info(f"Virtual environment created at {venv_path} using {python_binary}.")

    python_path = os.path.join("venv", "bin", python_binary)
    return str(venv_path), python_path


def install_python(bip_python_version: str, timeout: int) -> Optional[str]:
    """Install an older version of Python in order to build a buggy version of a project."""
    python_version = bip_python_version

    python_binary = None
    if float(bip_python_version) < 3.7:
        # Most package managers do not currently support python versions < 3.7,
        # so we choose the next nearest supported version
        # Python 3.6 build seems unstable and results in signal 11 during some runs on docker
        python_version = "3.7"
        logging.info(
            f"Installing Python {python_version}, not {bip_python_version},"
            " specified in BugsInPy."
        )

    try:
        subprocess.run(
            [f"python{python_version}", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        python_binary = f"python{python_version}"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        # Python version is not properly installed, or python version not found
        pass
    if python_binary:
        return python_binary

    if sys.platform not in SUPPORTED_OS:
        raise NotImplementedError(
            f"Your platform --- {sys.platform} --- is not supported."
        )
    pkg_commands = {
        "darwin": [
            ("brew", ["brew", "install", f"python@{python_version}"]),
            ("port", ["port", "install", f"python{python_version}"]),
        ],
        "linux": [
            (
                "apt",
                ["sudo", "apt", "install", f"python{python_version}", "-y"],
            ),
            (
                "yum",
                ["sudo", "yum", "install", f"python{python_version}", "-y"],
            ),
            (
                "dnf",
                ["sudo", "dnf", "install", f"python{python_version}", "-y"],
            ),
            (
                "zypper",
                ["sudo", "zypper", "install", f"python{python_version}", "-y"],
            ),  # openSUSE
            (
                "apk",
                ["sudo", "apk", "add", f"python{python_version}"],
            ),  # Alpine Linux
            (
                "pacman",
                [
                    "sudo",
                    "pacman",
                    "-S",
                    f"python{python_version}",
                    "--noconfirm",
                ],
            ),  # Arch Linux
        ],
        "win32": [
            (
                "choco",
                ["choco", "install", f"python{python_version}", "-y"],
            ),  # Windows (chocolatey)
        ],
    }
    pkg_manager_list = [
        pkg_manager[0]
        for pkg_managers in pkg_commands.values()
        for pkg_manager in pkg_managers
    ]

    has_pkg_manager = False
    for command in pkg_commands.get(sys.platform, []):
        pkg_manager_name = command[0]
        if not shutil.which(pkg_manager_name):
            logging.warning(
                f"The following package manager `{pkg_manager_name}` is not installed"
            )
            continue
        try:
            has_pkg_manager = True
            subprocess.run(command[1], check=True, timeout=timeout)
            subprocess.run(
                [f"python{python_version}", "--version"],
                check=True,
                timeout=timeout,
            )
            python_binary = f"python{python_version}"
            break
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            raise EnvironmentError(
                f"Unknown error encountered when installing python: {e}"
            ) from e

    if has_pkg_manager is False:
        supported_pkg_managers = ", ".join(pkg_manager_list)
        raise NotImplementedError(
            "Your environment does not have a supported package manager."
            f" Supported package managers: {supported_pkg_managers}"
        )

    if python_binary is None:
        raise EnvironmentError(f"Failed to install {bip_python_version}.")

    return python_binary


def convert_unittest_to_pytest(repo_path: Path, command: str) -> str:
    # Remove the initial part of the command
    parsed_specs = command.replace("unittest -q ", "").split()
    test_specs = []
    for spec in parsed_specs:
        test_specs.extend(spec.split(";"))

    pytest_commands = []

    for test_spec in test_specs:
        if test_spec.strip() == "":
            continue
        # Split the test specification into parts
        parts = test_spec.split(".")

        file_path = None
        test_components = None

        # Try to find the file path incrementally
        for i in range(len(parts), 0, -1):
            potential_file_path = repo_path / Path("/".join(parts[:i]) + ".py")
            if potential_file_path.exists():
                file_path = potential_file_path
                test_components = parts[i:]
                break

        # Skip if the file path is not found (or handle it as needed)
        if not file_path:
            print(f"File not found for test spec: {test_spec}")
            continue

        if "__init__.py" in str(file_path):
            # Can't run pytest on init files
            # Full path for the new file
            new_file_path = file_path.with_name("test_init.py")

            # Renaming the file
            try:
                file_path.rename(new_file_path)
                file_path.touch()
                file_path = new_file_path
            except OSError as error:
                print(f"Some Error renaming file: {error}")

        # Construct the test specifier (class and method)
        test_specifier = "::".join(test_components)

        # Construct the pytest command for this test spec
        pytest_command_for_spec = str(file_path)
        if test_specifier:
            pytest_command_for_spec += f"::{test_specifier}"

        pytest_commands.append(pytest_command_for_spec)

    # Join all the individual pytest commands
    return "pytest " + " ".join(pytest_commands)


def get_test_command(
    failing_test_commands, timeout, xml_output, python_path, repo_path
):
    failing_test_commands = failing_test_commands.replace("python -m", "")
    failing_test_commands = failing_test_commands.replace("python3 -m", "")
    if "tox" in failing_test_commands:
        failing_test_commands = failing_test_commands.replace("tox", "pytest")
    if "py.test" in failing_test_commands:
        failing_test_commands = failing_test_commands.replace("py.test", "pytest")
    if "unittest" in failing_test_commands:
        failing_test_commands = convert_unittest_to_pytest(
            repo_path, failing_test_commands
        )
    commands_list = failing_test_commands.split(";")
    commands_list = [
        command.strip() for command in commands_list if command.strip() != ""
    ]
    # Extract the test names from each command
    test_names = [command.split()[-1] for command in commands_list]

    # Join the test names into a single string separated by spaces
    test_names_str = " ".join(test_names)

    # Form the complete pytest command
    if xml_output:
        test_command = (
            f"{python_path} -m pytest {test_names_str} --timeout={timeout} -vv"
            f" --junit-xml={xml_output}"
        )
    else:
        test_command = (
            f"{python_path} -m pytest {test_names_str} --timeout={timeout} -vv"
        )
    return test_command


def run_test(
    bug_id: str,
    repo_path: Path,
    python_path: str,
    failing_test_commands: str,
    failing=True,
    timeout=None,
    xml_output=None,
    test_output_stdout=None,
) -> int:
    """Run the specified test on a checked out project."""

    # We change directories instead of using absolute paths b/c some tools,
    # notably tox, require relative paths.
    current_path = os.getcwd()
    os.chdir(repo_path)
    test_command = get_test_command(
        failing_test_commands, timeout, xml_output, python_path, repo_path
    )
    failed_as_expected = False
    returncode = 0
    try:
        result = subprocess.run(
            test_command,
            shell=True,
            check=True,
            timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
            stdout=test_output_stdout,
        )
        returncode = result.returncode
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:  # pylint: disable=invalid-name
        # We log, but tolerate, failures since it facilitates running on many tests,
        # which is especially useful when we are trying to get bugs running.
        if isinstance(e, subprocess.TimeoutExpired):
            # Return code for user interruption
            returncode = 2
        else:
            returncode = e.returncode
        if not failing:
            logging.error(f"Bug {bug_id} error: {str(e)}")
        else:
            failed_as_expected = True

    if failing:
        if failed_as_expected:
            logging.info(
                f"Bug {bug_id}:  Command '{test_command}' failed"
                f" as expected with result {returncode}."
            )
        else:
            logging.error(
                f"Bug {bug_id}:  Command '{test_command}'"
                f" unexpectedly passed with result {returncode}."
            )
    else:
        logging.info(
            f"Bug {bug_id}:  Command '{test_command}' ran"
            f" with the return code {returncode}."
        )

    os.chdir(current_path)
    return returncode


def install_envs(
    repo_path, bug_id, python_version, timeout, pip_output_redirection, restart
):
    class_json_path = (
        BGPConfig.BIP_ENVIRONMENT_CLASSES / "repo_bug_to_classname_mapping.json"
    )
    with class_json_path.open("r") as file:
        repo_bug_to_classname = json.load(file)

        # Look up the class index using the bug identifier.
        class_index = repo_bug_to_classname.get(bug_id)
        if class_index is None:
            # Empty requirements
            class_index = -1
    env_path = BGPConfig.BIP_ENVIRONMENT_DIR / bug_id.replace(":", "_")
    if restart and env_path.exists():
        try:
            shutil.rmtree(env_path)
        except:
            # Some files, such as links, cause exceptions
            pass
    python_path = env_path / "bin" / "python"
    if python_path.exists() is False:
        create_virtualenv(env_path, python_version, timeout)
        if class_index >= 0 and "ansible" not in repo_path.name:
            req_file_path = (
                BGPConfig.BIP_ENVIRONMENT_CLASSES / f"class_{class_index}.json"
            )
            with open(req_file_path, "r") as req_file:
                reqs = json.load(req_file)
                install_requirements(
                    python_path,
                    reqs["requirements"] + ["future"],
                    timeout,
                    pip_output_redirection,
                )
        install_test_dependencies(
            python_path, repo_path.name, timeout, pip_output_redirection
        )
        install_dependencies(
            bug_id, python_path, repo_path, timeout, pip_output_redirection
        )

    return env_path, python_path


def move_test_directory(repo_path):
    # Find the common parent directory for all test files
    common_dir = "tests"
    test_dir = repo_path / common_dir

    # Backup directory path
    test_dir_backup = BGPConfig.BIP_CLONED_REPOS / (
        common_dir + repo_path.name + str(randint(0, 100000))
    )
    if test_dir_backup.exists():
        shutil.rmtree(test_dir_backup)
    # Backup the test directory
    if test_dir.exists():
        shutil.move(test_dir, test_dir_backup)
    else:
        print(f"Test directory {test_dir} does not exist. Skipping backup.")
        return
    return test_dir, test_dir_backup


def move_test_file(bug_id, bug_record, repo_path, separate_envs):
    commit_id = bug_record["fixed_commit_id"]
    checkout(bug_id, repo_path, commit_id, separate_envs)
    tests = bug_record["test_file"].split(";")
    if "black" in bug_id or "scrapy" in bug_id:
        # Move the entire test directory
        test_dir, test_dir_backup = move_test_directory(repo_path)

        # Checkout the buggy version
        commit_id = bug_record["buggy_commit_id"]

        checkout(bug_id, repo_path, commit_id, separate_envs)
        # Restore the test directory in the buggy version
        if test_dir_backup and test_dir_backup.exists():
            if test_dir.exists():
                shutil.rmtree(test_dir)
            shutil.move(test_dir_backup, test_dir)
    else:
        # Copy the regression test that reveals the bug in the buggy version from the
        # fixed version into the buggy version.
        tests = [test for test in tests if test.strip() != ""]
        test_file_content = {}
        for test in tests:
            test_file = repo_path / Path(test)
            with open(test_file, "r", encoding="utf-8") as file:
                test_file_content[test] = file.read()

        commit_id = bug_record["buggy_commit_id"]
        checkout(bug_id, repo_path, commit_id, separate_envs)
        for test in tests:
            test_file = repo_path / Path(test)
            with open(test_file, "w", encoding="utf-8") as file:
                file.write(test_file_content[test])


def get_test_command_and_env(
    bug_id: str,
    repo_path: Path,
    bug_record: BugRecord,
    test_status_record: Dict[str, str] | None,
    timeout: int | None = None,
    pip_output_redirection=None,
    test_output_stdout=None,
    separate_envs=False,
) -> Tuple[Path, Path]:
    commit_id = bug_record["fixed_commit_id"]

    checkout(bug_id, repo_path, commit_id, separate_envs)

    test_status_change = False

    if not test_status_record:
        test_status_record = {}

    # Create virtualenv for dependencies, installing Python if necessary.
    if separate_envs:
        env_path, python_path = install_envs(
            repo_path,
            bug_id,
            bug_record["bip_python_version"],
            timeout,
            pip_output_redirection,
            False,
        )
    else:
        env_path, python_path = extract_and_install_class(
            repo_path.name,
            bug_id,
            bug_record["bip_python_version"],
            timeout,
            pip_output_redirection,
            False,
        )
        install_dependencies(
            bug_id, python_path, repo_path, timeout, pip_output_redirection
        )

    test_command = get_test_command(
        bug_record["failing_test_command"], timeout, None, python_path, repo_path
    )
    return python_path, test_command


def prep(
    bug_id: str,
    repo_path: Path,
    bug_record: BugRecord,
    test_status_record: Dict[str, str] | None,
    timeout: int | None = None,
    pip_output_redirection=None,
    test_output_stdout=None,
    restart=False,
    separate_envs=False,
) -> Tuple[bool, Dict[str, str]]:
    commit_id = bug_record["fixed_commit_id"]

    checkout(bug_id, repo_path, commit_id, separate_envs)

    test_status_change = False

    if not test_status_record:
        test_status_record = {}

    # Create virtualenv for dependencies, installing Python if necessary.
    if separate_envs:
        env_path, python_path = install_envs(
            repo_path,
            bug_id,
            bug_record["bip_python_version"],
            timeout,
            pip_output_redirection,
            restart,
        )
    else:
        env_path, python_path = extract_and_install_class(
            repo_path.name,
            bug_id,
            bug_record["bip_python_version"],
            timeout,
            pip_output_redirection,
            restart,
        )
        install_dependencies(
            bug_id, python_path, repo_path, timeout, pip_output_redirection
        )
    test_status_record["python_path"] = str(python_path)
    logging.info("Running test on the fixed commit")
    return_code = run_test(
        bug_id,
        repo_path,
        python_path,
        bug_record["failing_test_command"],
        False,
        timeout,
        xml_output=None,
        test_output_stdout=test_output_stdout,
    )
    does_test_run, test_status = get_test_status_info(return_code)
    if "does_test_run" not in test_status_record or test_status_record[
        "does_test_run"
    ] != str(does_test_run):
        test_status_record["does_test_run"] = str(does_test_run)
        test_status_change = True
    if "test_status" not in test_status_record or test_status_record[
        "test_status"
    ] != str(test_status):
        test_status_record["test_status"] = str(test_status)
        test_status_change = True
    if test_status == TestStatus.PASS:
        logging.info("Test passes on fixed commit!")
    else:
        logging.warning(
            f"Test encountered error code {test_status} during execution"
            " on fixed commit!"
        )

    # Copy the regression test that reveals the bug in the buggy version from the
    # fixed version into the buggy version.
    move_test_file(bug_id, bug_record, repo_path, separate_envs)

    logging.info(f"Checked out to buggy commit: {bug_record['buggy_commit_id']}")

    return test_status_change, test_status_record


def install_dependencies(
    bug_id: str,
    python_path: str,
    repo_path: Path,
    timeout: int,
    pip_output_redirection: int,
) -> None:
    """Install project dependencies."""

    # We change directories instead of using absolute paths b/c some tools,
    # notably tox, require relative paths.
    current_path = os.getcwd()
    os.chdir(repo_path)
    repo_name = repo_path.name
    if "_" in repo_name:
        repo_name = repo_name.split("_")[0]
    if repo_name in ["PySnooper"]:
        install_test_dependencies(
            python_path, repo_path.name, timeout, pip_output_redirection
        )

    if repo_name in [
        "ansible",
        "black",
        "pandas",
        "matplotlib",
        "sanic",
        "scrapy",
    ]:
        install_build_dependencies(
            python_path, repo_path, timeout, pip_output_redirection
        )
        if "matplotlib" in repo_name or "ansible" in repo_name:
            command = (
                f"{python_path} -m pip install --no-build-isolation --editable .[dev]"
            )
        elif "pandas" in repo_name:
            command = f"{python_path} setup.py build_ext --inplace --force"
        elif "spacy" in repo_name:
            command = f"python setup.py build_ext --inplace"
        else:
            command = f"{python_path} -m pip install ."
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            logging.error(f"Error: {str(e)}")

    os.chdir(current_path)


def install_test_dependencies(
    python_path: Path | str, repo: str, timeout: int, pip_output_redirection
) -> None:
    if os.path.exists("test_requirements.txt"):
        try:
            subprocess.run(
                f"{python_path} -m pip install -r test_requirements.txt",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            logging.error(f"Error: {str(e)}")
    if os.path.exists("requirements.txt"):
        try:
            subprocess.run(
                f"{python_path} -m pip install -r requirements.txt",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            logging.error(f"Error: {str(e)}")

    try:
        if "httpie" in repo:
            subprocess.run(
                f"{python_path} -m pip install --no-deps pytest-timeout==1.0.0",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        elif "sanic" in repo:
            subprocess.run(
                f"{python_path} -m pip install pytest==5.2.1"
                " pytest-sanic==1.6.1 pytest-sugar==0.9.3 pytest-timeout==2.2.0",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        elif "thefuck" in repo:
            subprocess.run(
                f"{python_path} -m pip install pytest==3.10.0 pytest-timeout==2.2.0"
                " pytest-mock pytest-cov",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        elif "PySnooper" in repo:
            try:
                subprocess.run(
                    f"{python_path} -m pip install six pytest-timeout pytest==2.2.0"
                    " python_toolbox",
                    shell=True,
                    check=True,
                    timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                    stdout=pip_output_redirection,
                    stderr=pip_output_redirection,
                )
            except subprocess.CalledProcessError as e:  # pylint: disable=invalid-name
                logging.error(f"Error: {str(e)}")

        else:
            subprocess.run(
                f"{python_path} -m pip install pytest pytest-timeout==2.2.0"
                " pytest-mock pytest-cov",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:  # pylint: disable=invalid-name
        logging.error(f"Error: {str(e)}")


def modify_setup_file_remove_werror(file_path):
    """Remove -Werror switch from setup.py to reproduce more bugs."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    modified_lines = [line.replace('["-Werror"]', "[]") for line in lines]

    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(modified_lines)


def install_build_dependencies(
    python_path: str, repo_path: Path, timeout: int, pip_output_redirection
) -> None:
    if "pandas" in repo_path.name:
        modify_setup_file_remove_werror("setup.py")
        try:
            subprocess.run(
                f"{python_path} -m pip install cython==0.29.19 numpy==1.18.4 "
                "python-dateutil hypothesis==5.15.1 pytz==2020.1",
                shell=True,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            logging.error(f"Error: {str(e)}")


def install_project_requirements(
    bug_id: str, venv_path: str, repo_path: Path, timeout: int, pip_output_redirection
) -> None:
    project_name = repo_path.name
    bgp_path = (
        BGPConfig.BIP_PROJECTS_DIR / project_name / "bugs" / bug_id / "requirements.txt"
    )
    if not bgp_path.exists():
        return

    # ignore non utf-8 characters
    nl = "\n"  # pylint: disable=invalid-name
    with open(bgp_path, "rb") as file:
        content_bytes = file.read()
        try:
            requirements = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            requirements = content_bytes.decode("cp1252")
            requirements = requirements.replace("\r\n", nl)
    requirement_lines = requirements.split("\n")
    install_requirements(venv_path, requirement_lines, timeout, pip_output_redirection)


def get_available_extras() -> List[str]:
    """Return list of extra dependencies needed by some projects."""
    return ["test", "tests", "d", "develop"]


def install_extras(extras: List[str], venv_path: str, timeout: int) -> None:
    """Install extra dependencies needed by some projects."""
    for extra in extras:
        pip_command = [venv_path, "-m", "pip", "install", f".[{extra}]"]
        try:
            subprocess.run(
                pip_command,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            # Expected to fail because the list of extras we try to install
            # is a superset of those needed by any given dependency.
            pass


def install_tox_deps(
    venv_path: str, tox_ini_path: str, timeout: int, pip_output_redirection
) -> Dict[str, List[str]]:
    """Install tox dependencies needed by some projects."""

    logging.info("Installing tox dependencies")
    config = configparser.ConfigParser()
    config.read(tox_ini_path)

    test_dependencies = {}
    for section in config.sections():
        if section.startswith("testenv:"):
            env_name = section[len("testenv:") :]
            dependencies = config.get(section, "deps", fallback="").split()
            if "deps" in dependencies:
                continue
            test_dependencies[env_name] = dependencies
        elif section.startswith("testenv"):
            env_name = "None"
            dependencies = config.get(section, "deps", fallback="").split()
            if "deps" in dependencies:
                continue
            test_dependencies[env_name] = dependencies

    for deps in test_dependencies.values():
        command = [venv_path, "-m", "pip", "install"] + deps
        try:
            subprocess.run(
                command,
                check=True,
                timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
                stdout=pip_output_redirection,
                stderr=pip_output_redirection,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:  # pylint: disable=invalid-name
            # Expected to fail in some circumstances
            logging.error(f"Error: {str(e)}")

    return test_dependencies


def ignore_venv(repo_path: Path) -> None:
    """Utility function that programmatically adds venv/ to project .gitignore files."""

    if repo_path.exists():
        gitignore_file = repo_path / ".gitignore"
        if gitignore_file.exists():
            with open(repo_path / ".gitignore", "r") as file:
                content = file.read()
                if "venv/\n" in content:
                    return
        with open(repo_path / ".gitignore", "a") as file:
            file.write("venv/\n")
    else:
        msg = (
            f"Please clone {repo_path.name}'s repository before attempting to"
            " modify it."
        )
        raise InvalidExecutionOrderError(msg)


def extract_and_install_class(
    repo, bug_id, python_version, timeout, pip_output_redirection, restart
):
    class_json_path = (
        BGPConfig.BIP_ENVIRONMENT_CLASSES / "repo_bug_to_classname_mapping.json"
    )
    with class_json_path.open("r") as file:
        repo_bug_to_classname = json.load(file)

        # Look up the class index using the bug identifier.
        class_index = repo_bug_to_classname.get(bug_id)
        if class_index is None:
            # Empty requirements
            class_index = -1
    env_path = BGPConfig.BIP_ENVIRONMENT_DIR / str(class_index)
    python_path = env_path / "bin" / "python"
    if restart and env_path.exists():
        try:
            shutil.rmtree(env_path)
        except:
            # Some files, such as links, cause exceptions
            pass
    if env_path.exists() is False:
        create_virtualenv(env_path, python_version, timeout)
        if class_index >= 0 and "ansible" not in repo:
            req_file_path = (
                BGPConfig.BIP_ENVIRONMENT_CLASSES / f"class_{class_index}.json"
            )
            with open(req_file_path, "r") as req_file:
                reqs = json.load(req_file)
                install_requirements(
                    python_path,
                    reqs["requirements"] + ["future"],
                    timeout,
                    pip_output_redirection,
                )
        install_test_dependencies(python_path, repo, timeout, pip_output_redirection)

    return env_path, python_path


def remove_non_ascii(input_str: str) -> str:
    """Remove non-ASCII characters from the input string."""
    return "".join(ch for ch in input_str if ord(ch) < 128 and ord(ch) > 0)


def filter_requirements(requirements: str) -> str:
    """
    Filter requirements
    :param nl: The new line
    :param requirements: The requirements
    """
    modified_requirements = requirements.replace("# Our libraries", "")
    modified_requirements = modified_requirements.replace("# Optional dependencies", "")
    modified_requirements = modified_requirements.replace(
        "# Development dependencies", ""
    )
    modified_requirements = (
        modified_requirements.replace("rc2", "").replace("rc1", "").replace(".dev0", "")
    )
    modified_requirements = modified_requirements.replace(
        f"pkg-resources==0.0.0\n", ""
    ).replace(f"Keras==2.2.2\n", "")
    modified_requirements = modified_requirements.replace(
        "pkg-resources==0.0.0", ""
    ).replace(f"Keras==2.2.2", "")
    modified_requirements = modified_requirements.replace(
        "ansible-base==2.10.1", "ansible-base==2.10.0"
    )
    modified_requirements = modified_requirements.replace(
        f"ansible==2.10.0\n", ""
    ).replace(f"ansible==2.10.0", "")
    modified_requirements = modified_requirements.replace(
        f"ansible==2.10.0.dev0\n", ""
    ).replace(f"ansible==2.10.0.dev0", "")
    modified_requirements = modified_requirements.replace(f"pywin32==227\n", "")
    modified_requirements = modified_requirements.replace(f"pywin32==227", "")
    modified_requirements = modified_requirements.replace(f"pywin32>=223\n", "")
    modified_requirements = modified_requirements.replace(f"pywin32>=223", "")
    modified_requirements = modified_requirements.replace(f"pypiwin32==223", "")
    modified_requirements = modified_requirements.replace(f"h11==0.9.0", "h11==0.8.1")
    if "luigi.git" in modified_requirements:
        modified_requirements = modified_requirements.replace(
            "tornado==5.0", "tornado==4.5.3"
        )
        modified_requirements = modified_requirements.replace(
            "python-dateutil==2.8.1", ""
        )

    if "Keras-Applications==1.0.7" in modified_requirements:
        modified_requirements = modified_requirements.replace(
            "tensorflow==1.15.0", "tensorflow==1.14.0"
        )
        modified_requirements = modified_requirements.replace(
            "tensorboard==1.15.0", "tensorboard==1.14.0"
        )
        modified_requirements = modified_requirements.replace(
            "tensorflow-estimator==1.15.1", "tensorflow-estimator==1.14.0"
        )
    modified_requirements = modified_requirements.replace(
        f"fastapi==0.55.1\n", ""
    ).replace(f"fastapi==0.55.1", "")
    return modified_requirements


def install_requirements(venv_path, requirements, timeout, pip_output_redirection):
    # Some lines have comments with '#', others have ';'.
    # Dropping these is better
    lines = [
        line.split(";")[0].strip()
        for line in requirements
        if line.strip() and not line.strip().startswith("#")
    ]
    modified_requirements = " ".join(lines)
    modified_requirements = remove_non_ascii(modified_requirements)
    modified_requirements = filter_requirements(modified_requirements)
    try:
        command = "{} -m pip install --no-cache-dir {}".format(
            venv_path, modified_requirements.replace("\n", " ")
        )
        # Too long to use shell=True, have to use split
        subprocess.run(
            command.split(),
            shell=False,
            check=True,
            timeout=BGPConfig.TIMEOUT_MULTIPLIER * timeout,
            stdout=pip_output_redirection,
            stderr=pip_output_redirection,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:  # pylint: disable=invalid-name
        # Some packages might not be available
        logging.error(f"Error: {str(e)}")
