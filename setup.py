from setuptools import setup, find_packages


#location = os.path.abspath(os.path.dirname(__file__))
#with open(os.path.join(location, "requirements.txt"), "r") as f:
#    requirements = f.read().splitlines()

# For some reason tox fails during the above file read
REQUIREMENTS = ["absl-py>=1.0.0,<2.0.0", "GitPython>=3.0.0,<4.0.0"] 

setup(
    name="pyr_benchmark_wrangling",
    version="0.0.1",
    # If they are structured as proper Python packages, this should find both:
    packages=find_packages(include=["BugsInPy*", "diff_utils*"]),
    install_requires=REQUIREMENTS,
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
