# Releasing

Releases are automated. Pushing a version tag builds the package, publishes it to
PyPI, and attaches the built files to a GitHub Release. Nothing is uploaded by
hand.

## Steps

1. Bump the version in `pygad/_version.py`. This is the only place the version
   lives.
2. Update the release notes in the docs if you keep them there.
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
5. Tag the release and push the tag:
   ```bash
   git tag 3.8.0
   git push origin 3.8.0
   ```

The `release` workflow first runs the full Python 3.8 through 3.14 test matrix,
then builds the wheel and sdist, publishes them to PyPI, and creates a GitHub
Release with both files attached after publication succeeds. Follow it
with `gh run watch` or the Actions tab.

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
