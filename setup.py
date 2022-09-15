import json
import re
from pathlib import Path
from setuptools import setup, find_packages


MODULE_NAME = "spatial_effects"
DEPENDENCIES = [
    "numpy",
]


with open(Path.cwd() / "project_info.json", "r") as f:
    pkg_info = json.load(f)


def find_version(file: Path) -> str:
    """Searches file for a pattern like __version__ = "1.2.3" and returns the
    version string.
    """
    name = str(file)

    if not file.is_file():
        raise ValueError(f"Cannot open {name}")

    with open(name) as f:
        match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if match:
            return match.group(1)
        raise RuntimeError(f"No version found in {name}.")


setup(
    name=MODULE_NAME,
    version=find_version(Path.cwd() / MODULE_NAME / "__init__.py"),
    packages=find_packages(),
    url=pkg_info["url"],
    license=pkg_info["license"],
    author=pkg_info["author"],
    author_email=pkg_info["email"],
    description=pkg_info["description"],
    install_requires=DEPENDENCIES,
    extras_require={"full": []},
    package_data={},
    test_suite=f"{MODULE_NAME}.tests",
)
