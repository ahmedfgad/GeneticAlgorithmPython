# PyGAD and Vilvik SDK compatibility

`test_vilvik_compatibility.py` runs real PyGAD optimizations and calls the real
SDK through `GA.push_to_vilvik()`. Only HTTP is intercepted; no API key, account,
VPS, billing action or network request to Vilvik is needed.

The tests verify result/population fidelity, scalar and NumPy gene types,
precision and mixed types, single/multiple objectives, stop criteria, mutation
control precedence, callback source, dry runs, explicit capture overrides,
population opt-out and SDK error propagation. Exported source and constructor
parameters are used to reconstruct and run another GA. This catches payloads
that serialize successfully but cannot be continued.

Run locally:

```sh
python -m pip install -e . 'vilvik>=0.5.3' pytest responses
python -m pytest tests/test_push_to_vilvik.py tests/test_vilvik_compatibility.py -q
```

The dedicated GitHub workflow runs on relevant pushes and pull requests, every
Monday, and manually. Python 3.9/3.12 each test SDK 0.5.3, the latest PyPI SDK,
and the SDK's `main` branch. Weekly checks catch SDK releases even when this
repository has no new commits. The optional tests skip when the SDK or HTTP
test dependency is absent in ordinary PyGAD-only environments; the dedicated
workflow explicitly installs and imports both, so missing dependencies fail CI.
The release workflow also runs these tests with the latest released SDK before
building/publishing PyGAD.

The SDK repository has the reciprocal suite: its current source is tested
against released and development PyGAD versions. A failing compatibility job
must be resolved before releasing the affected package. Neither suite replaces
the Vilvik server contract tests or live beta tests; those cover server
validation, execution, authentication, billing and webhooks.
