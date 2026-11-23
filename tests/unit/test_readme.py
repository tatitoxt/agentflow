from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_library_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert blocks, "README should contain a python example"
    exec(compile(blocks[0], "README.md", "exec"), {})
    assert capsys.readouterr().out == "It is 18C, cloudy in Lima. final_answer\n"
