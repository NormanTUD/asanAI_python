"""Tests for the CI coverage summary renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).with_name("coverage_summary.py")
_SPEC = importlib.util.spec_from_file_location("coverage_summary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
coverage_summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage_summary)


def test_render_missing_file(tmp_path):
    assert coverage_summary.render(tmp_path / "missing.json") == "Coverage JSON not found."


def test_render_no_uncovered_lines(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(
            {
                "files": {
                    "clean.py": {
                        "missing_lines": [],
                        "summary": {"percent_covered": 100.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert coverage_summary.render(path) == "No uncovered lines."


def test_render_uncovered_table(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(
            {
                "files": {
                    "short.py": {
                        "missing_lines": [1, 2],
                        "summary": {"percent_covered": 90.5},
                    },
                    "long.py": {
                        "missing_lines": list(range(1, 61)),
                        "summary": {"percent_covered": 10.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rendered = coverage_summary.render(path)
    assert "| File | Coverage | Missing lines |" in rendered
    assert "`short.py`" in rendered
    assert "90.50%" in rendered
    assert "1, 2" in rendered
    assert "`long.py`" in rendered
    assert "…" in rendered


def test_main(tmp_path, capsys):
    path = tmp_path / "coverage.json"
    path.write_text('{"files": {}}', encoding="utf-8")
    assert coverage_summary.main(["coverage_summary.py", str(path)]) == 0
    assert "No uncovered lines." in capsys.readouterr().out
