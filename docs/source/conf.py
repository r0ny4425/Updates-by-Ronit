"""Sphinx configuration for the SimYuj documentation rewrite."""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"

sys.path.insert(0, str(_SRC_ROOT))

project = "SimYuj"
copyright = "2026, SimYuj"
author = "SimYuj"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["simyuj_docs.css", "simyuj_code_theme.css"]
html_js_files = ["simyuj_docs.js"]

mermaid_version = "11.12.1"
mermaid_output_format = "raw"
mermaid_init_config = {
    "startOnLoad": False,
    "theme": "default",
}
mermaid_height = "auto"
mermaid_fullscreen = False

html_theme_options = {
    "logo": {
        "image_light": "_static/simyuj-mark.svg",
        "image_dark": "_static/simyuj-mark.svg",
        "text": "SimYuj — Quantum Network Simulator",
    },
    "show_nav_level": 0,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "show_prev_next": True,
    "secondary_sidebar_items": ["page-toc"],
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["search-button-field"],
    "navbar_persistent": [],
    "primary_sidebar_end": [],
}

html_sidebars = {"**": ["sidebar-nav-bs"]}


_PY_LITERAL_RE = re.compile(
    r"``([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\([^)]*\))?)``"
)


def _python_object_aliases() -> dict[str, str]:
    aliases: defaultdict[str, set[str]] = defaultdict(set)

    for path in (_SRC_ROOT / "simyuj").rglob("*.py"):
        if path.name == "__init__.py":
            continue

        module = ".".join(path.relative_to(_SRC_ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                class_target = f"{module}.{node.name}"
                aliases[node.name].add(class_target)

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name.startswith("_"):
                            continue
                        aliases[f"{node.name}.{child.name}"].add(
                            f"{class_target}.{child.name}"
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    aliases[node.name].add(f"{module}.{node.name}")

    return {
        alias: next(iter(targets))
        for alias, targets in aliases.items()
        if len(targets) == 1
    }


_PY_OBJECT_ALIASES = _python_object_aliases()


def _xref_autodoc_literals(app, what, name, obj, options, lines) -> None:
    """Rewrite simple autodoc code literals as Python object references."""

    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        target = label.split("(", 1)[0]
        resolved = _PY_OBJECT_ALIASES.get(target)
        if resolved is None:
            return match.group(0)
        return f":py:obj:`{label} <{resolved}>`"

    for index, line in enumerate(lines):
        lines[index] = _PY_LITERAL_RE.sub(replace, line)


def _xref_source_literals(app, docname, source) -> None:
    """Rewrite simple hand-written RST code literals as Python object references."""

    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        target = label.split("(", 1)[0]
        resolved = _PY_OBJECT_ALIASES.get(target)
        if resolved is None:
            return match.group(0)
        return f":py:obj:`{label} <{resolved}>`"

    source[0] = _PY_LITERAL_RE.sub(replace, source[0])


def setup(app):
    app.connect("autodoc-process-docstring", _xref_autodoc_literals)
    app.connect("source-read", _xref_source_literals)
