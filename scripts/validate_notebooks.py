"""Validate notebook structure and optionally execute notebooks without saving outputs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def notebook_paths() -> list[Path]:
    return sorted(
        [
            *ROOT.glob("*.ipynb"),
            *ROOT.glob("modules/**/*.ipynb"),
            *ROOT.glob("exercises/**/*.ipynb"),
            *ROOT.glob("projects/**/*.ipynb"),
        ]
    )


def validate(execute: bool, selected: list[str] | None = None) -> None:
    paths = notebook_paths()
    if selected:
        selected_paths = {(ROOT / value).resolve() for value in selected}
        unknown = selected_paths - {path.resolve() for path in paths}
        if unknown:
            raise SystemExit(f"Unknown notebook path(s): {sorted(map(str, unknown))}")
        paths = [path for path in paths if path.resolve() in selected_paths]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as error:
                    raise SyntaxError(f"{path.relative_to(ROOT)} cell {index}: {error}") from error
        if execute:
            from nbclient import NotebookClient

            NotebookClient(notebook, timeout=300, kernel_name="python3").execute(cwd=ROOT)
        print(f"{'executed' if execute else 'validated'} {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Run code cells in a temporary kernel"
    )
    parser.add_argument(
        "--path", action="append", help="Validate only this repo-relative notebook (repeatable)"
    )
    args = parser.parse_args()
    validate(args.execute, args.path)
