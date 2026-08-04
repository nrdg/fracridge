import hashlib
import json
from pathlib import Path

import pytest

EXPECTED_CODE_SHA256 = (
    "443bc52a59fd8544bba30e5387de778045e3d5d07e02359b0522480b911f1e6e"
)
REQUIRED_MAINTAINER_SECTIONS = (
    "## Executive summary",
    "### Expected invariant: affine OLS behavior",
    "### Current code path: preprocessing in the wrong coordinate system",
    "### Interpreting the current estimator",
    "### Why the raw-data reference isolates the fix",
    "### Problem 1 acceptance criteria",
    "### Expected invariant: stable OLS at fraction 1.0",
    "### Current code path: normal equations square conditioning",
    "### Controlled experiment and reported metrics",
    "### Interpreting coefficient error and residual error",
    "### Why the direct-SVD reference solves the root cause",
    "### Problem 2 acceptance criteria",
    "## Pull request acceptance checklist",
)


@pytest.fixture
def notebook_path():
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "examples" / "diagnose_intercept_and_svd_inaccuracy.ipynb"


def _source_text(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


@pytest.fixture
def notebook_document(notebook_path):
    return json.loads(notebook_path.read_text(encoding="utf-8"))


@pytest.fixture
def notebook_code_cells(notebook_document):
    return [
        _source_text(cell)
        for cell in notebook_document["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture
def notebook_markdown(notebook_document):
    return "\n".join(
        _source_text(cell)
        for cell in notebook_document["cells"]
        if cell["cell_type"] == "markdown"
    )


def _code_digest(code_cells):
    digest = hashlib.sha256()
    for source in code_cells:
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def test_diagnostic_notebook_executes(
    notebook_path, notebook_code_cells, monkeypatch
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "__main__"}

    assert notebook_code_cells
    for index, source in enumerate(notebook_code_cells, start=1):
        cell_name = f"{notebook_path}#code-cell-{index}"
        exec(compile(source, cell_name, "exec"), namespace)


def test_notebook_contains_maintainer_narrative(notebook_markdown):
    missing = [
        heading
        for heading in REQUIRED_MAINTAINER_SECTIONS
        if heading not in notebook_markdown
    ]
    assert not missing, f"Missing maintainer sections: {missing}"


def test_notebook_preserves_diagnostic_code(notebook_code_cells):
    assert len(notebook_code_cells) == 8
    assert _code_digest(notebook_code_cells) == EXPECTED_CODE_SHA256


def test_notebook_is_output_free(notebook_document):
    code_cells = [
        cell
        for cell in notebook_document["cells"]
        if cell["cell_type"] == "code"
    ]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(not cell["outputs"] for cell in code_cells)
