import json
from pathlib import Path

import pytest


@pytest.fixture
def notebook_path():
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "examples" / "diagnose_intercept_and_svd_inaccuracy.ipynb"


def _source_text(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


@pytest.fixture
def notebook_code_cells(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        _source_text(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_diagnostic_notebook_executes(
    notebook_path, notebook_code_cells, monkeypatch
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "__main__"}

    assert notebook_code_cells
    for index, source in enumerate(notebook_code_cells, start=1):
        cell_name = f"{notebook_path}#code-cell-{index}"
        exec(compile(source, cell_name, "exec"), namespace)
