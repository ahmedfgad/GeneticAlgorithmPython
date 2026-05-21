# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'PyGAD'
copyright = '2026, Ahmed Fawzy Gad'
author = 'Ahmed Fawzy Gad'

# The full version, including alpha/beta/rc tags.
release = '3.6.0'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# The documentation is written in Markdown and read directly by Sphinx
# through the MyST parser. There is no Markdown-to-reStructuredText step.
extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx_copybutton',
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

# Do NOT set myst_heading_anchors. Leaving it unset keeps Sphinx using the
# docutils section IDs (for example "PyGAD 2.18.0" -> "pygad-2-18-0"), which
# are the anchors the live site already links to. Turning it on would switch
# to GitHub-style slugs and break those links.

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'PyGAD'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['scroll-sidebar.js']

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
