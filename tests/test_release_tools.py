"""Publication must use dated notes and exactly the checked PyPI distributions."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "release_tools", Path(__file__).resolve().parents[1] / "tools/release.py")
release_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_tools)


def test_release_notes_are_scoped_and_links_are_portable(tmp_path, monkeypatch):
    source = tmp_path / "docs/source"
    source.mkdir(parents=True)
    (source / "releases.md").write_text(
        "## Unreleased\n\nNo changes.\n\n## PyGAD 3.8.0\n\n"
        "Release Date: October 9, 2026.\n\n"
        "1. [Lifecycle](visualize.md#plot_lifecycle).\n"
        "2. [Issue](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/1).\n\n"
        "## PyGAD 3.7.0\n\nOld release.\n", encoding="utf-8")
    monkeypatch.setattr(release_tools, "ROOT", tmp_path)
    notes = release_tools.release_notes("3.8.0")
    assert notes.startswith("# PyGAD 3.8.0\n")
    assert "October 9, 2026" in notes
    assert "/blob/3.8.0/docs/source/visualize.md#plot_lifecycle" in notes
    assert "https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/1" in notes
    assert "Unreleased" not in notes and "Old release" not in notes


def test_release_notes_reject_pending_date(tmp_path, monkeypatch):
    source = tmp_path / "docs/source"
    source.mkdir(parents=True)
    (source / "releases.md").write_text(
        "## PyGAD 3.8.0\n\nRelease Date: pending publication.\n", encoding="utf-8")
    monkeypatch.setattr(release_tools, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="publication date"):
        release_tools.release_notes("3.8.0")


def published_files(tmp_path, monkeypatch, change=None):
    names = ["pygad-3.8.0-py3-none-any.whl", "pygad-3.8.0.tar.gz"]
    contents = [b"checked wheel", b"checked sdist"]
    files = []
    for name, data, kind in zip(names, contents, ["bdist_wheel", "sdist"]):
        (tmp_path / name).write_bytes(data)
        files.append({"filename": name, "packagetype": kind, "yanked": False,
                      "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                      "url": "https://files.pythonhosted.org/" + name})
    metadata = {"info": {"version": "3.8.0"}, "urls": files}
    if change:
        change(metadata)
    replies = iter([json.dumps(metadata).encode()] + contents)
    monkeypatch.setattr(release_tools, "urlopen", lambda *args, **kwargs: io.BytesIO(next(replies)))
    return names, contents


def test_downloads_both_verified_pypi_files(tmp_path, monkeypatch):
    names, contents = published_files(tmp_path, monkeypatch)
    destination = tmp_path / "published"
    release_tools.fetch_pypi("3.8.0", tmp_path, destination)
    assert [(destination / name).read_bytes() for name in names] == contents


@pytest.mark.parametrize("change, message", [
    (lambda data: data["urls"][0]["digests"].update(sha256="bad"), "differs"),
    (lambda data: data["urls"][0].update(yanked=True), "Unexpected published"),
    (lambda data: data["urls"][0].update(url="https://example.com/file"), "download host"),
    (lambda data: data["info"].update(version="3.7.0"), "different package version"),
])
def test_rejects_incorrect_pypi_files(tmp_path, monkeypatch, change, message):
    published_files(tmp_path, monkeypatch, change)
    with pytest.raises(ValueError, match=message):
        release_tools.fetch_pypi("3.8.0", tmp_path, tmp_path / "published")


def test_rejects_corrupted_download(tmp_path, monkeypatch):
    names, contents = published_files(tmp_path, monkeypatch)
    first_open = release_tools.urlopen
    metadata = first_open().read()
    replies = iter([metadata, b"corrupted download"])
    monkeypatch.setattr(release_tools, "urlopen", lambda *args, **kwargs: io.BytesIO(next(replies)))
    with pytest.raises(ValueError, match="failed verification"):
        release_tools.fetch_pypi("3.8.0", tmp_path, tmp_path / "published")
