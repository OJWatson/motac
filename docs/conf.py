project = "motac"
author = "motac3"
extensions = [
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 3,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# Keep docs as local-build artifacts; do not execute notebooks during docs build.
nbsphinx_execute = "never"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
