"""Prepare documented release notes and verified copies of PyPI distributions."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/ahmedfgad/GeneticAlgorithmPython"


def release_notes(version):
    """Extract one published release and make its documentation links portable."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Use a bare release version, such as 3.8.0.")
    source = (ROOT / "docs/source/releases.md").read_text(encoding="utf-8")
    heading = "## PyGAD " + version
    sections = re.split(r"(?m)^## ", source)
    matches = [section for section in sections if section.startswith("PyGAD " + version + "\n")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one release-notes section for " + version)
    notes = "## " + matches[0].strip()
    if not re.search(r"(?m)^Release Date: [A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}\.$", notes):
        raise ValueError("Set a publication date before creating release notes.")
    if "pending publication" in notes or "has not been published" in notes:
        raise ValueError("Remove pending-publication text before releasing.")
    base = REPOSITORY + "/blob/" + version + "/docs/source/"
    notes = re.sub(r"\]\(([^)]+)\)", lambda match: "](" + urljoin(base, match[1]) + ")", notes)
    return notes.replace(heading, "# PyGAD " + version, 1) + "\n"


def fetch_pypi(version, built_directory, output_directory):
    """Download the published wheel and sdist only when their build hashes match."""
    expected = {
        "pygad-" + version + "-py3-none-any.whl": "bdist_wheel",
        "pygad-" + version + ".tar.gz": "sdist",
    }
    built_hashes = {
        name: hashlib.sha256((built_directory / name).read_bytes()).hexdigest()
        for name in expected
    }
    for attempt in range(30):
        try:
            with urlopen("https://pypi.org/pypi/pygad/" + version + "/json", timeout=30) as response:
                metadata = json.load(response)
            if metadata["info"]["version"] != version:
                raise ValueError("PyPI returned a different package version.")
            files = {item["filename"]: item for item in metadata["urls"]}
            if set(files) == set(expected):
                break
        except (HTTPError, URLError) as error:
            if isinstance(error, HTTPError) and error.code != 404:
                raise
        if attempt == 29:
            raise RuntimeError("Both published distributions are not available on PyPI.")
        time.sleep(10)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, package_type in expected.items():
        item = files[name]
        if item["yanked"] or item["packagetype"] != package_type:
            raise ValueError("Unexpected published distribution: " + name)
        if item["digests"]["sha256"] != built_hashes[name]:
            raise ValueError("PyPI distribution differs from the checked build: " + name)
        if not item["url"].startswith("https://files.pythonhosted.org/"):
            raise ValueError("Unexpected PyPI download host.")
        with urlopen(item["url"], timeout=60) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != built_hashes[name]:
            raise ValueError("Downloaded PyPI distribution failed verification: " + name)
        (output_directory / name).write_bytes(contents)
        print(name + " SHA-256 " + built_hashes[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    notes = commands.add_parser("notes")
    notes.add_argument("version")
    notes.add_argument("output", type=Path)
    packages = commands.add_parser("fetch-pypi")
    packages.add_argument("version")
    packages.add_argument("built_directory", type=Path)
    packages.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "notes":
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(release_notes(arguments.version), encoding="utf-8")
    else:
        fetch_pypi(arguments.version, arguments.built_directory, arguments.output_directory)


if __name__ == "__main__":
    main()
