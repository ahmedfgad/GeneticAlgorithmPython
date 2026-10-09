# Connecting Python Examples to the Documentation

`python_examples.json` is the shared catalog for the Examples index and the Python example cards in the guides. Keep the descriptions and requirements here rather than copying them into each guide. Paths are relative to the repository's `examples/` directory.

For the general documentation conventions, see [Writing Documentation in Markdown](MARKDOWN.md).

To show one or more examples beside a relevant explanation, use this template in a documentation page:

```markdown
<!-- python-examples
example_initial_population.py
example_gene_type_conversion.py
-->

<!-- /python-examples -->
```

After changing the catalog or adding a section, update the checked-in Markdown from the repository root:

```console
python docs/markdown_compatibility.py --update-examples
```

The generated section contains ordinary Markdown links, descriptions, and expandable run instructions. It is readable on GitHub and in Markdown previews before any build. Do not edit the generated text directly; edit the catalog or shared templates, then regenerate it. To check that the generated sections are current, run `python docs/markdown_compatibility.py`.

For larger groups, the section uses a compact table with expandable run instructions. The Examples index uses `python-examples-index` comments to list every entry by topic, with links back to its guide. Sphinx presents these sections using the existing example cards, tables, dropdowns, and downloads.

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

The shared templates are in `python_example_templates/`. The `*-source.md.template` files use standard Markdown and generate the checked-in sections. The `.md.jinja` files use Sphinx Design cards and dropdowns and Sphinx's native download links. Both presentations use the same catalog. The `markdown_compatibility.py` extension checks the source sections and passes them to `python_examples.py` for the built presentation; neither executes example scripts.

The documentation build checks that every Python script appears in the catalog, all catalog paths stay inside `examples/`, and the linked guides exist. Missing entries, unknown paths, incomplete section comments, and stale generated Markdown fail the build so examples are not silently left out. Sphinx copies downloadable scripts from the repository into the built documentation; no second script copy needs to be maintained.

GitHub links use the commit checked out for the documentation build. Without Git, the configured Read the Docs identifier is used, falling back to `master`. This keeps source links aligned with versioned documentation.
