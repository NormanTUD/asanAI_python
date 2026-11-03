"""Render the uncovered lines from a coverage.json file as a Markdown table.

Used by the CI workflow to put per-worker coverage details into the GitHub
step summary. Kept as a tiny standalone script so the YAML stays readable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render(path: Path) -> str:
    if not path.exists():
        return "Coverage JSON not found."

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for filename, info in data.get("files", {}).items():
        missing = info.get("missing_lines", [])
        if not missing:
            continue

        shown = ", ".join(str(line) for line in missing[:50])
        suffix = " …" if len(missing) > 50 else ""
        percent = info.get("summary", {}).get("percent_covered", 0.0)
        rows.append(f"| `{filename}` | {percent:.2f}% | {shown}{suffix} |")

    if not rows:
        return "No uncovered lines."

    header = "| File | Coverage | Missing lines |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "coverage.json")
    print(render(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
