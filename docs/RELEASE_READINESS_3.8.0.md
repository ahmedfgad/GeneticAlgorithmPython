# PyGAD 3.8.0 release readiness

Reviewed October 9, 2026. All library preparation is pushed to the
`github-actions` branch. Nothing has been tagged, merged or published.
The `master` branch remains at `92e7c7f`.

## Version

The latest public release is **3.7.0**, published June 5, 2026. PyPI has no
3.8.0 distribution as of this review. The next release is **3.8.0** because
it adds public features, including `plot_lifecycle()` and generation metadata,
alongside fixes and internal refactoring. A patch release would understate
those additions. The main `GA` constructor signature is unchanged from 3.7.0;
the reviewed changes do not require a new major version.

The package version is prepared in `pygad/_version.py`. Sphinx reads that file
directly so documentation and package versions cannot drift.

## Validation

- Source checkout: 1,871 tests passed, one skipped, on Python 3.11.13 with
  NumPy 2.4.4. The skip is the unavailable Windows `fork` process start method.
- TensorFlow/Keras and PyTorch tests were excluded locally because those
  optional frameworks are absent. They are covered by the existing remote
  matrix on its supported framework versions.
- Release preparation matrix: all Python 3.8 through 3.14 jobs passed at commit
  `6f31df251d4faca89571573ffac5a8e53f3ffe1d`, using the changed workflow and
  installed 3.8.0 wheels:
  https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/37972271705
  The following commit only updates this readiness report.
- SDK compatibility: all four minimum/latest SDK jobs on Python 3.11 and 3.12
  passed for that same release preparation commit:
  https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/37972271741
- Wheel and source distribution built as 3.8.0 and both passed `twine check`.
  The wheel contains the logo needed by PDF reports and declares its extras.
- Installed wheel: 1,871 tests passed, one skipped, in a separate virtual
  environment from outside the repository. Imports resolved to the installed
  3.8.0 wheel, not the source checkout. The PyPI submodule-version check was
  required to pass rather than silently skip if unavailable.
- HTML documentation built with `-W --keep-going` and no warnings.
- Generated example Markdown is current; its catalog covers 81 Python scripts
  and one notebook.
- Changed workflows passed actionlint 1.7.12. Whitespace checks passed.

## Publishing safeguards prepared

The release workflow now runs the same full Python matrix before building
and publishing. Matrix tests run outside the checkout to import the installed
wheel. The workflow rejects mismatches between the tag and package version,
checks distributions, and requires a clean documentation build. The GitHub
Release job waits for successful PyPI publication.

The GitHub `pypi` environment exists. The prior 3.7.0 release completed through
the trusted-publishing workflow:
https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/27043771194
Its previous success does not establish that account permissions can never
change; the next release run remains the verification of live publishing.

## Compatibility notes to retain

- Seeded results can differ from earlier versions. Reproducibility is within
  the same version and environment, with independent generators per instance.
- Invalid fitness values and malformed constructor settings are rejected
  earlier and consistently. These corrections are detailed in the release notes.
- Saved histories retain both snapshots at repeated-run boundaries and expose
  actual generation numbers.
- Metadata now declares Python 3.8 or newer, matching the minimum tested version.

## Remaining publication steps

The reviewed tree passed the local prepublication checks. The changed test
workflow and SDK compatibility workflow also passed on GitHub for the release
preparation commit. The tag-triggered publishing workflow has been checked
locally; its publishing steps have not been executed for 3.8.0.

Review the prepared diff.
Set the actual publication date in `docs/source/releases.md` when releasing.
Choose the final release commit, then create the matching 3.8.0 tag only when
publication is authorized. Pushing that tag publishes automatically.

## Announcement video

The previous source project was found at
https://github.com/ahmedfgad/PyGADReleaseVideo and cloned as a sibling at
`D:/Projects/PyGADReleaseVideo`. New work is on `codex/pygad-3.8.0-video`.
It reuses the original animation kit, fonts, logo, background music, typing
sounds and transition effects. Only the bell cues are replaced with the
user-approved soft wooden tap.
Runnable on-screen examples, real output, chapters and draft social posts
are in `content/3.8.0/`. See that repository's `RELEASE_3.8.0.md` for builds.

The landscape (3840x2160), vertical (2160x3840), and square (2160x2160)
videos are complete. The announcement kit was uploaded to `PyGAD_3.8.0/`
on that repository's video branch, with MP4 files stored through Git LFS:
https://github.com/ahmedfgad/PyGADReleaseVideo/tree/codex/pygad-3.8.0-video/PyGAD_3.8.0
All four MP4 objects uploaded successfully and local Git LFS integrity checks
passed. A fresh authenticated download of the preview matched its SHA-256 hash.
The three full videos are 127 seconds each, and the preview is 21.5 seconds.
The opening shows the logo and version for 5.5 seconds, down from 9.5 seconds.
All four run at 60 fps,
with H.264 video and AAC stereo audio. Every encoded frame decoded successfully;
contact sheets from all nine segments were visually checked in all orientations.
Thumbnails, a Reel cover, chapter timestamps, title captions, draft social posts,
and machine-readable media validation are included. Social posts remain drafts;
no videos have been published to social platforms.

The October 9 revisions move section takeaways into larger, high-contrast
callouts above the output and enlarge the history chart in every orientation.
Plots come from 360 dpi exports. Audio checks verify that the original music
and non-bell effects are preserved, including all 417 typing clicks. Encoded
music, typing transients, tap levels, loudness and peaks are checked in all four
videos. Layout and audio measurements are included in the
video repository's `PyGAD_3.8.0/layout_review.json` and `audio_validation.json`.
