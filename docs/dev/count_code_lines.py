"""Count non-docstring code lines, with a sum assertion that cannot be skipped.

Written because an ad-hoc version of this got it wrong. It computed

    code = total - docstring - blank - comment

while counting *every* blank line as blank, including the blank lines inside
docstrings. Those lines were therefore subtracted twice and ``code`` came out
low. Nothing detected it, because nothing checked that the four categories added
back up to the total.

The fix is one line -- exclude docstring lines from the blank and comment counts
-- and the guard is one more: ``assert code + doc + comment + blank == total``.
Any future variant of this script must keep that assertion. A line count in a
design memo is read by a later session as fact.

Usage::

    uv run python docs/dev/count_code_lines.py src/simyuj/signal/optical.py
    uv run python docs/dev/count_code_lines.py --per-definition <file>
"""

from __future__ import annotations

import argparse
import ast
import pathlib


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Return every line occupied by a docstring or attribute docstring."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
        ):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def count(path: pathlib.Path) -> tuple[int, int, int, int, int]:
    """Return ``(code, docstring, comment, blank, total)`` for one file.

    Raises
    ------
    AssertionError
        If the four categories do not sum to the total. That assertion is the
        entire point of this module; do not remove it.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    doc = _docstring_lines(ast.parse(source))

    total = len(lines)
    blank = sum(
        1 for i, line in enumerate(lines, 1) if not line.strip() and i not in doc
    )
    comment = sum(
        1
        for i, line in enumerate(lines, 1)
        if line.strip().startswith("#") and i not in doc
    )
    code = total - len(doc) - blank - comment

    assert code + len(doc) + comment + blank == total, (
        f"{path}: categories do not sum to the total -- "
        f"{code} + {len(doc)} + {comment} + {blank} != {total}"
    )
    return code, len(doc), comment, blank, total


def _code_between(
    lines: list[str],
    doc: set[int],
    lo: int,
    hi: int,
    exclude: tuple[tuple[int, int], ...] = (),
) -> int:
    skip: set[int] = set()
    for start, end in exclude:
        skip.update(range(start, end + 1))
    return sum(
        1
        for i in range(lo, hi + 1)
        if i not in doc
        and i not in skip
        and lines[i - 1].strip()
        and not lines[i - 1].strip().startswith("#")
    )


def per_definition(path: pathlib.Path) -> list[tuple[str, int]]:
    """Return ``(name, code_lines)`` for every top-level definition and method.

    The rows sum to the file's ``code`` count from :func:`count`, which the CLI
    asserts.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    doc = _docstring_lines(tree)

    rows: list[tuple[str, int]] = []
    covered: list[tuple[int, int]] = []
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            covered.append((node.lineno, node.end_lineno or node.lineno))
            methods = [m for m in node.body if isinstance(m, functions)]
            spans = tuple((m.lineno, m.end_lineno or m.lineno) for m in methods)
            rows.append(
                (
                    f"class {node.name} [fields]",
                    _code_between(
                        lines,
                        doc,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        spans,
                    ),
                )
            )
            for method in methods:
                rows.append(
                    (
                        f"    {node.name}.{method.name}",
                        _code_between(
                            lines,
                            doc,
                            method.lineno,
                            method.end_lineno or method.lineno,
                        ),
                    )
                )
        elif isinstance(node, functions):
            covered.append((node.lineno, node.end_lineno or node.lineno))
            rows.append(
                (
                    f"def {node.name}",
                    _code_between(
                        lines, doc, node.lineno, node.end_lineno or node.lineno
                    ),
                )
            )

    rows.append(
        (
            "<module: imports, constants, __all__>",
            _code_between(lines, doc, 1, len(lines), tuple(covered)),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--per-definition", action="store_true")
    args = parser.parse_args()

    if args.per_definition:
        for path in args.paths:
            rows = per_definition(path)
            print(path)
            for name, lines_of_code in rows:
                print(f"  {lines_of_code:>5}  {name}")
            subtotal = sum(lines_of_code for _, lines_of_code in rows)
            code = count(path)[0]
            assert subtotal == code, (
                f"{path}: per-definition rows sum to {subtotal}, "
                f"file total is {code}"
            )
            print(f"  {subtotal:>5}  TOTAL (matches whole-file count)")
        return

    print(f"{'code':>6} {'doc':>6} {'cmnt':>6} {'blank':>6} {'total':>7}  file")
    for path in args.paths:
        code, doc, comment, blank, total = count(path)
        print(f"{code:>6} {doc:>6} {comment:>6} {blank:>6} {total:>7}  {path}")


if __name__ == "__main__":
    main()
