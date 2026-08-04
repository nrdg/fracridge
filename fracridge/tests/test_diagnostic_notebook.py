import json
from pathlib import Path

import pytest

REQUIRED_MAINTAINER_SECTIONS = (
    "## Executive summary",
    "### Legacy reproductions are local, not monkeypatches",
    "### Expected invariant: affine OLS behavior",
    "### Legacy code path: preprocessing in the wrong coordinate system",
    "### Why the raw-data control isolates the fix",
    "### Interpreting the legacy and corrected estimators",
    "### Problem 1 resolution",
    "### Expected invariant: stable OLS at fraction 1.0",
    "### Legacy code path: normal equations square conditioning",
    "### Why the direct-SVD path solves the root cause",
    "### Controlled experiment and reported metrics",
    "### Interpreting coefficient error and residual error",
    "### Problem 2 resolution",
    "## Pull request verification checklist",
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


def test_diagnostic_notebook_executes(
    notebook_path, notebook_code_cells, monkeypatch
):
    """Runs every code cell in order in one shared namespace, so the notebook's
    own assertions become the regression test for its legacy-versus-corrected
    evidence.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "__main__"}

    assert notebook_code_cells
    for index, source in enumerate(notebook_code_cells, start=1):
        cell_name = f"{notebook_path}#code-cell-{index}"
        exec(compile(source, cell_name, "exec"), namespace)


def test_notebook_contains_maintainer_narrative(notebook_markdown):
    """The notebook is the maintainer-facing record of both defects, so the
    sections carrying that explanation must survive later edits to the code
    cells around them.
    """
    missing = [
        heading
        for heading in REQUIRED_MAINTAINER_SECTIONS
        if heading not in notebook_markdown
    ]
    assert not missing, f"Missing maintainer sections: {missing}"

