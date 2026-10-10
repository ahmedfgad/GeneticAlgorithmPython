# PyGAD 3.8.0 release readiness

Prepublication review completed October 9, 2026. This document records the
preparation and validation performed on `github-actions` before its release
pull request to `master`. The publication date in the release notes is
October 9, 2026.

## Published release

PyGAD 3.8.0 is published on [PyPI](https://pypi.org/project/pygad/3.8.0/)
and [GitHub](https://github.com/ahmedfgad/GeneticAlgorithmPython/releases/tag/3.8.0).
[PR #378](https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/378) merged
`github-actions` into `master`. The release tag points to commit
`c8752b6a901a078299998f137453f7b650046c94`.

The [publishing workflow](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/38013833060)
passed all seven Python jobs, package checks, documentation checks, PyPI
publication, and GitHub Release creation. Independent downloads confirmed that
both GitHub package files match PyPI by SHA-256. Fresh installs of the wheel and
source distribution passed optimization, repeated-history, transparent-chart,
and PDF-report checks. Another 151 regression tests passed against the published
wheel. The public `latest` and `stable` documentation builds succeeded.

Watch the [published announcement video on YouTube](https://youtu.be/8pdIiMAMLUM).

## Version

At the start of this review, the latest public release was **3.7.0**, published
June 5, 2026. The prepared release is **3.8.0** because
it adds public features, including `plot_lifecycle()` and generation metadata,
alongside fixes and internal refactoring. A patch release would understate
those additions. The main `GA` constructor signature is unchanged from 3.7.0;
the reviewed changes do not require a new major version.

The package version is prepared in `pygad/_version.py`. Sphinx reads that file
directly so documentation and package versions cannot drift.

## Validation

- Release preparation baseline: 1,871 tests passed, one skipped, on Python 3.11.13 with
  NumPy 2.4.4. The skip is the unavailable Windows `fork` process start method.
- TensorFlow/Keras and PyTorch tests were excluded locally because those
  optional frameworks are absent. They are covered by the existing remote
  matrix on its supported framework versions.
- Latest matrix: all seven Python 3.8 through 3.14 jobs passed at commit
  `adde6bfe83f1e8b2ba8a3c8f73205f2dbcab8951`, including the lifecycle export fix,
  using the changed workflow and
  installed 3.8.0 wheels:
  https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/37995352872
- SDK compatibility: all six minimum/latest/development SDK jobs on Python 3.9
  and 3.12 passed for that same commit:
  https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/runs/37995352864
- The lifecycle fix passed all 41 focused lifecycle tests locally, related
  report tests, and a strict Sphinx build with `-n -W --keep-going`.
  Transparent PNG tests verify alpha, small margins and unclipped chart text.
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
Release job waits for successful PyPI publication, then downloads the published
wheel and source distribution and verifies their SHA-256 hashes against the
build. The documented release notes are used for both the PR and GitHub Release.

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

## Publication procedure

The reviewed tree passed the local prepublication checks. The changed test
workflow and SDK compatibility workflow also passed on GitHub for the release
preparation commit. Publication is authorized by the maintainer. The release
notes have the requested October 9, 2026 date. The release PR uses those notes,
and the merged `master` commit receives the matching 3.8.0 tag. Pushing that tag
publishes automatically. The final verification checks PyPI installation and
confirms that both GitHub assets match the published PyPI packages.

## Announcement video

The previous source project was found at
https://github.com/ahmedfgad/PyGADReleaseVideo and cloned as a sibling at
`D:/Projects/PyGADReleaseVideo`. All video work is consolidated on `main`.
It reuses the original animation kit, fonts, logo, typing sounds and transition
effects. Bell cues use the user-approved soft wooden tap. The revised background
music is the user-approved A: Warm keys. C: Floating ambient and the original
release music are retained as separate three-minute WAV and MP3 tracks, with
their synthesis code, in `Music_Library_3.8.0/`.
https://github.com/ahmedfgad/PyGADReleaseVideo/tree/main/Music_Library_3.8.0
Runnable on-screen examples, real output, chapters and draft social posts
are in `content/3.8.0/`. See that repository's `RELEASE_3.8.0.md` for builds.

The landscape (3840x2160), vertical (2160x3840), and square (2160x2160)
videos are complete. The announcement kit was uploaded to `PyGAD_3.8.0/`
on that repository's `main` branch, with MP4 files stored through Git LFS:
https://github.com/ahmedfgad/PyGADReleaseVideo/tree/main/PyGAD_3.8.0
The revised video commit is `f31aaad8ff7c9bc7b02fa956b6a010848fc9ee16`,
using the library implementation at `adde6bfe`.
All four MP4 objects uploaded successfully and local Git LFS integrity checks
passed. A fresh authenticated download of the preview matched its SHA-256 hash.
The three full videos are 134.13 seconds each, and the preview is 22.5 seconds.
The opening shows the logo and version for 5.5 seconds, down from 9.5 seconds.
All four run at 60 fps,
with H.264 video and 48 kHz AAC stereo audio. Every encoded frame decoded successfully;
contact sheets from all nine segments were visually checked in all orientations.
Thumbnails, a Reel cover, chapter timestamps, title captions, draft social posts,
and machine-readable media validation are included. The release video is now
published on [YouTube](https://youtu.be/8pdIiMAMLUM). The other platform posts
remain drafts.

The October 9 revisions move section takeaways into larger, high-contrast
callouts above the output and enlarge the history chart in every orientation.
Plots come from 360 dpi exports. All section titles are uppercase. Code typing
is reduced from 60 to 44 characters per second, and the documentation command
from 34 to 30. The permutation demo highlights and prints its initial row.
The lifecycle call is fully highlighted, and the library now exports its chart
with small margins and optional transparency. The video uses that actual export.
The worker demo explains a target sum of 20 and illustrates two real process
workers reused over four generations; a pool/PID audit is included. The
documentation demo uses an actual screenshot and its complete example command.

Audio checks verify the approved music and preserved non-bell effects,
including all 469 typing clicks. Encoded
music, typing transients, tap levels, loudness and peaks are checked in all four
videos. Layout and audio measurements are included in the
video repository's `PyGAD_3.8.0/layout_review.json` and `audio_validation.json`.
