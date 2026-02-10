"""Sphinx configuration for motac documentation."""

import os
import sys
import warnings

# Add package source directory to path
sys.path.insert(0, os.path.abspath("../src"))

project = "motac"
author = "OJ Watson"
copyright = "2026, OJ Watson"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

# Keep docs builds fast and deterministic; tutorials are rendered but not executed.
nbsphinx_execute = "never"

# Suppress nbformat warnings about missing cell IDs in notebooks.
warnings.filterwarnings("ignore", category=Warning, module="nbformat")

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

suppress_warnings = [
    "bibtex.duplicate_label",
    "bibtex.duplicate_citation",
    "bibtex.missing_field",
]
