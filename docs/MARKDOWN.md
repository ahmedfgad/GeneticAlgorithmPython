# Writing Documentation in Markdown

Documentation pages should be readable on GitHub and in Markdown previews as well as on Read the Docs. Use ordinary Markdown for headings, links, images, lists, tables, and code blocks.

For links within the documentation, use a relative `.md` path and the GitHub-style heading anchor:

```markdown
[Plot Lifecycle](visualize.md#plot_lifecycle)
```

MyST resolves these links to the appropriate pages and section IDs during a Sphinx build. Existing published anchors remain available. Builds report missing pages and heading anchors.

Use HTML `<details>` and `<summary>` for collapsible descriptions. Leave blank lines around their Markdown content. Use `<code>` for code in a summary because Markdown formatting is not processed inside the summary itself:

```markdown
<details>
<summary><code>gene_type=float</code>: Data type of the genes.</summary>

The type used to store each gene value.

</details>
```

The `markdown_compatibility.py` extension converts these blocks into the existing Sphinx Design dropdowns for HTML and other documentation formats.

Keep Sphinx-only metadata, such as explicit labels and toctrees, inside `<!-- sphinx ... -->` comments. The extension restores this metadata during a build; Markdown previews hide it. A toctree should have visible Markdown navigation beside it, either a list on the home page or a navigation group:

```markdown
<!-- navigation-grid: 1 2 2 3 -->

- [Controlling Gene Values](gene_values.md) — Set ranges, types, constraints, and duplicate prevention.
- [Controlling Generations](generations.md) — Configure stopping, elitism, and continuation.

<!-- /navigation-grid -->
```

The build presents these lists as the existing navigation cards. Their titles, destinations, and descriptions are written only once.

For diagrams with a preferred display width, put a normal PNG image and its caption between `documentation-figure` comments, following the existing pages. The image works directly in Markdown; the build restores the centered figure, caption, and width and selects the appropriate image format. Embedded videos stay in Sphinx comments with a visible YouTube link beside them.

Python example sections are generated from the catalog and shared templates. See [Connecting Python Examples to the Documentation](PYTHON_EXAMPLES.md) for the editing and checking commands. Regeneration needs only Python; it does not require Sphinx or execute the examples.
