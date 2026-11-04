"""Check local Markdown links in documentation and notebook Markdown cells."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def sources() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*.md"):
        if ".git" not in path.parts and ".venv" not in path.parts:
            entries.append((path, path.read_text(encoding="utf-8")))
    for path in ROOT.rglob("*.ipynb"):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        entries.append((path, markdown))
    return entries


def main() -> None:
    broken: list[str] = []
    checked = 0
    for source, text in sources():
        for match in LINK_PATTERN.finditer(text):
            target = match.group(1).strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            checked += 1
            relative = unquote(parsed.path)
            if not relative:
                continue
            destination = (source.parent / relative).resolve()
            if not destination.exists():
                broken.append(f"{source.relative_to(ROOT)} -> {target}")
    if broken:
        raise SystemExit("Broken local links:\n" + "\n".join(broken))
    print(f"Checked {checked} local Markdown links; none are broken.")


if __name__ == "__main__":
    main()
