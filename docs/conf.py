import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "motac"
author = "OJ Watson"
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
    "tutorials/06_acled_hawkes_deep_dive_and_improvements.ipynb",
    "tutorials/08_acled_nuts_truncated.ipynb",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 3,
    "secondary_sidebar_items": [],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["mathjax-fix.js"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], [r"\\(", r"\\)"]],
        "displayMath": [["$$", "$$"], [r"\\[", r"\\]"]],
        "processEscapes": True,
    }
}

# Keep docs as local-build artifacts; do not execute notebooks during docs build.
nbsphinx_execute = "never"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
myst_update_mathjax = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
