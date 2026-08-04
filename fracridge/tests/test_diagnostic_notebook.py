import hashlib
import json
from pathlib import Path

import pytest

EXPECTED_CODE_SHA256 = (
    "9d1c13371baab219dc08e01da9bdd9c288e1041b39a78352a7f248eb89feaac4"
)
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


