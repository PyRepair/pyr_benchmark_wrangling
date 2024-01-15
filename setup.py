from setuptools import setup, find_packages

# Read requirements.txt and set it as the requirements for our package
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="pyr_benchmark_wrangling",
    version="0.0.1",
    # If they are structured as proper Python packages, this should find both:
    packages=find_packages(include=["BugsInPy*", "diff_utils*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lmeasures=BugsInPy.lmeasures:main",
            "bgp=BugsInPy.bgp:main",
            "sample_bip=BugsInPy.sample_bip:main",
            "run_custom_patch=BugsInPy.run_custom_patch:main",
            "diff_utils=diff_utils.diff_utils:main",
        ]
    },
    include_package_data=True,
)
