# Installation

## Requirements

- Python 3.11+
- (Optional) `pandoc` if you want to render certain notebook formats in Sphinx.

## Install

Using `uv` (recommended):

```bash
uv sync
```

Editable install (useful for development):

```bash
uv pip install -e .
```

Docs dependencies:

```bash
uv sync --group docs
```
