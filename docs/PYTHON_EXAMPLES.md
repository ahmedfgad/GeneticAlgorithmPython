# Connecting Python Examples to the Documentation

`python_examples.json` is the shared catalog for the Examples index and the Python example cards in the guides. Keep the descriptions and requirements here rather than copying them into each guide. Paths are relative to the repository's `examples/` directory.

To show one or more examples beside a relevant explanation, use this template in a documentation page:

```markdown
:::{python-examples}
example_initial_population.py
example_gene_type_conversion.py
:::
```

For larger groups, the same directive uses a compact table inside the card. Run instructions remain in a dropdown. The Examples index uses `python-examples-index` to list every entry by topic, with links back to its guide.

Each catalog entry has these fields:

- `path`: Existing Python script or notebook under `examples/`.
- `title`: Short descriptive name for the example.
- `description`: What readers will learn from the script.
- `category`: Topic heading in the Examples index. Categories follow their first appearance in the catalog.
- `guide`: Existing Markdown guide, relative to `docs/source/`.
- `requirements`: Libraries or optional extras needed in addition to a matching version of PyGAD.
- `run`: Command to run from the repository root, or an empty string for a notebook.
- `run_note` (optional): Working-directory instructions when the root cannot be used directly.
- `data` (optional): Dataset filenames, expected layout, and any setup limitations.
- `download` (optional, default `true`): Set to `false` when downloading a script alone would omit required data or companion files. Readers receive a folder link instead.

The shared Markdown templates are in `python_example_templates/`. They use the existing Sphinx Design cards and dropdowns and Sphinx's native download links. The small `python_examples.py` extension resolves catalog paths and renders those templates; it does not execute example scripts.

The documentation build checks that every Python script appears in the catalog, all catalog paths stay inside `examples/`, and the linked guides exist. Missing entries fail the build so new examples are not silently left out. Unknown paths in a guide also fail the build. Sphinx copies downloadable scripts from the repository into the built documentation; no second source copy needs to be maintained.

GitHub links use the commit checked out for the documentation build. Without Git, the configured Read the Docs identifier is used, falling back to `master`. This keeps source links aligned with versioned documentation.
