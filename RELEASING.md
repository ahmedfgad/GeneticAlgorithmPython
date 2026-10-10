# Releasing

Releases are automated. Pushing a version tag builds the package, publishes it to
PyPI, and downloads the published wheel and source distribution for the GitHub
Release. Downloaded files must match the checked build's SHA-256 hashes.
The GitHub Release uses the release notes from `docs/source/releases.md`.

## Steps

1. Bump the version in `pygad/_version.py`. This is the only place the version
   lives.
2. Update the matching `PyGAD <version>` section in `docs/source/releases.md`.
   Set `Release Date: Month D, YYYY.` and remove pending-publication text.
3. Commit and push:
   ```bash
   git add pygad/_version.py docs/source/releases.md
   git commit -m "Prepare PyGAD 3.8.0"
   git push
   ```
   Stage any other intended release changes before committing. Preparation stays
   on `github-actions` until the maintainer chooses the final release commit;
   these steps do not require changes to `master`.
4. Wait for the test workflow (`main.yml`) to pass on that commit. Confirm the
   release notes describe the intended version and replace its pending release
   date with the actual publication date. Documentation reads the package version
   automatically. Build and check the distributions before tagging:
   ```bash
   python -m build
   python -m twine check dist/*
   ```
5. Create or update a pull request from `github-actions` to `master`, using the
   documented release notes as its description. Generate the description with:
   ```bash
   python tools/release.py notes 3.8.0 docs/build/release-notes-3.8.0.md
   ```
   Wait for its checks and merge it. Tag the merged `master` commit and push the tag:
   ```bash
   git switch master
   git pull --ff-only origin master
   git tag 3.8.0
   git push origin 3.8.0
   ```

The `release` workflow first runs the full Python 3.8 through 3.14 test matrix,
then builds the wheel and sdist, checks documentation and release notes, and
publishes the packages to PyPI. It downloads both published files, verifies
their SHA-256 hashes against the build, and creates a GitHub Release with those
files and the documented notes. Documentation links in the PR and release notes
point to the tagged source. Follow the workflow with `gh run watch` or the Actions tab.
Verify the PyPI version and GitHub assets after it succeeds.

## Rules

- The tag must match `pygad/_version.py` and is the bare version number with no
  `v` prefix, for example `3.8.0`. The tag is what triggers the release.
- Every release needs a new version number. PyPI does not allow re-uploading or
  overwriting a version that already exists.
- Do not run `twine upload` or upload files to the GitHub Release by hand. The
  tag does both for you.

## Manual fallback

`publish.sh` can build and upload to PyPI from your machine if you ever need it.

## One-time setup (maintainers)

Done once per project. No API token is involved, because PyPI trusted publishing
is tokenless.

- On the PyPI `pygad` project, open Settings, then Publishing, and add a GitHub
  publisher: owner `ahmedfgad`, repository `GeneticAlgorithmPython`, workflow
  `release.yml`, environment `pypi`.
- In the GitHub repo, open Settings, then Environments, and create an environment
  named `pypi`.
