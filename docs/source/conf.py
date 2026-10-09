# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'PyGAD'
copyright = '2026, Ahmed Fawzy Gad'
author = 'Ahmed Fawzy Gad'

# The full version, including alpha/beta/rc tags.
release = '3.7.0'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pin example links to the checkout used to build the documentation. This also
# works for release branches and older documentation versions on Read the Docs.
try:
    python_examples_source_revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
        text=True, stderr=subprocess.DEVNULL).strip()
except (OSError, subprocess.CalledProcessError):
    python_examples_source_revision = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', 'master')

# The documentation is written in Markdown and read directly by Sphinx
# through the MyST parser. There is no Markdown-to-reStructuredText step.
extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx_copybutton',
    'python_examples',
    'markdown_compatibility',
]

# Read both Markdown and reStructuredText. Markdown is the source of truth.
# The .rst mapping lets pages that are not migrated yet keep building.
source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

# _templates is not used.
templates_path = []

# Files and directories to skip when looking for source files.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'linkify',
    'substitution',
    'tasklist',
    'dollarmath',
]

# Resolve Markdown links such as [Plot Lifecycle](visualize.md#plot_lifecycle)
# using GitHub-style heading anchors at every heading level. MyST maps these
# anchors to the existing Sphinx section IDs, preserving published links.
myst_heading_anchors = 6

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'PyGAD'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['scroll-sidebar.js', 'release-history.js']

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#0b6e4f',
        'color-brand-content': '#0b6e4f',
    },
    'dark_css_variables': {
        'color-brand-primary': '#27ae60',
        'color-brand-content': '#27ae60',
    },
}

# -- Options for LaTeX / PDF output (xelatex) --------------------------------

latex_engine = 'xelatex'
latex_elements = {
    'inputenc': '',
    'utf8extra': '',
    'preamble': r'''
\usepackage{kotex}
\usepackage{fontspec}
\setsansfont{Arial}
\setromanfont{Arial}
''',
}
