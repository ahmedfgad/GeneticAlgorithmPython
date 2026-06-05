# This test checks the version of each submodule. Each submodule keeps its
# own __version__ in its __init__.py. When the code in a submodule changes,
# its version should change too. It is easy to forget this, so the version
# can stay the same as the last release by mistake.
#
# The top-level pygad version does not need this check. PyPI does not allow
# uploading the same version twice, so a wrong pygad version stops the upload.
# The submodule versions do not have this protection, so this test adds it.
#
# To find the last release, the test downloads pygad from PyPI with
# pip download. PyPI always has the last release, so this works without git
# tags or git history. The downloaded files are compared with the current
# pygad files. If a submodule changed but its version did not, the test fails.
#
# If PyPI cannot be reached, the test is skipped so it does not break an
# offline run. Set PYGAD_REQUIRE_RELEASE_CHECK=1 to make it fail instead of
# skip.

import os
import re
import sys
import zipfile
import tarfile
import subprocess
from pathlib import Path

import pytest

import pygad


SUBMODULES = [
    "utils",
    "helper",
    "visualize",
    "nn",
    "cnn",
    "gann",
    "gacnn",
    "kerasga",
    "torchga",
    "benchmarks",
]

# Path to the pygad package that is imported now. In CI this is the built
# wheel installed by pip. In a local checkout it is the repo pygad/ folder.
CURRENT_PKG = Path(pygad.__file__).resolve().parent

_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _parse_version(init_path):
    """Read __version__ from an __init__.py file by reading its text."""
    match = _VERSION_RE.search(init_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _source_map(package_dir):
    """Return a dict of file path to file text for every .py file in a folder.

    __pycache__ files are skipped. Trailing spaces and the last newline are
    removed so a small spacing difference is not counted as a code change.
    """
    sources = {}
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in text.splitlines())
        sources[str(path.relative_to(package_dir))] = normalized
    return sources


@pytest.fixture(scope="module")
def released_pkg(tmp_path_factory):
    """Download the last pygad release from PyPI and return its pygad/ folder."""
    dest = tmp_path_factory.mktemp("pygad_release_download")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "download", "pygad",
             "--no-deps", "--dest", str(dest)],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        _skip_or_fail(f"Could not reach PyPI to download pygad: {error}")
    else:
        if result.returncode != 0:
            _skip_or_fail(
                "`pip download pygad` failed (offline?):\n" + result.stderr
            )

    artifacts = list(dest.glob("pygad-*.whl")) + list(dest.glob("pygad-*.tar.gz"))
    if not artifacts:
        _skip_or_fail("No pygad artifact was downloaded from PyPI.")

    extract_dir = tmp_path_factory.mktemp("pygad_release_extracted")
    artifact = artifacts[0]
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            archive.extractall(extract_dir)
    else:
        with tarfile.open(artifact) as archive:
            archive.extractall(extract_dir)

    # The pygad folder is at <extract>/pygad for a wheel or at
    # <extract>/pygad-x.y.z/pygad for an sdist. Find it by its __init__.py.
    inits = [
        p for p in extract_dir.rglob("__init__.py")
        if p.parent.name == "pygad"
    ]
    if not inits:
        _skip_or_fail("Downloaded pygad artifact has no pygad/ package.")
    return min((p.parent for p in inits), key=lambda p: len(p.parts))


def _skip_or_fail(message):
    if os.environ.get("PYGAD_REQUIRE_RELEASE_CHECK") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_changed_submodule_is_version_bumped(submodule, released_pkg):
    current_dir = CURRENT_PKG / submodule
    assert current_dir.is_dir(), (
        f"submodule pygad/{submodule} is missing from the current package"
    )

    current_version = _parse_version(current_dir / "__init__.py")
    assert current_version, f"pygad/{submodule}/__init__.py has no __version__"

    released_dir = released_pkg / submodule
    if not released_dir.is_dir():
        # This submodule is new and was not in the last release, so there
        # is nothing to compare. Just make sure it has a version.
        return

    released_version = _parse_version(released_dir / "__init__.py")

    if _source_map(current_dir) != _source_map(released_dir):
        assert current_version != released_version, (
            f"pygad.{submodule} source changed since the latest PyPI release "
            f"({released_version}) but __version__ is still {current_version}. "
            f"Bump __version__ in pygad/{submodule}/__init__.py."
        )
