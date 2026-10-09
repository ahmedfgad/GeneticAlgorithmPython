# Examples

Find complete Python scripts by topic, open their source on GitHub, or download examples that need no companion files. Each entry links back to its documentation guide. The same examples appear in **Python example** cards beside the relevant explanations in those guides.

## Running the Examples

For examples from this documentation revision, use the matching repository version of PyGAD. This is especially important for features in the Unreleased notes. Clone or download that repository revision, then install it from the repository root:

```console
python -m pip install -e ".[visualize]"
```

Run a script using its path, for example:

```console
python examples/plots/example_plot_lifecycle.py
```

For a script downloaded separately, run `python` with the path where you saved it. It still requires a compatible installed version of PyGAD. Plotting examples need Matplotlib; PDF reports need `pygad[report]`. Keras and PyTorch examples need their respective frameworks. The **Run this example** dropdowns in the guides give additional requirements and working directories.

Examples requiring datasets link to their folders instead of offering a standalone download. Their datasets are not bundled with PyGAD or the repository. Follow the linked data setup instructions and keep the repository directory layout. The TSP notebook is listed alongside the Python scripts; it uses Google Colab and a user-supplied CSV. Adapt its Colab-specific imports and CSV path before running it locally with Jupyter.

Use documentation search or your browser's find command to locate a topic or filename on this page.

:::{python-examples-index}
:::
