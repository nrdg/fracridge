# Developer entry points for fracridge.

NOTEBOOK := examples/diagnose_intercept_and_svd_inaccuracy.ipynb

# nbconvert and a kernel are only needed to execute the notebook, so they are
# layered in for that one command rather than added to the package dependencies.
JUPYTER_DEPS := --with nbconvert --with ipykernel

# Keep matplotlib headless so the plotting cells never try to open a window.
export MPLBACKEND := Agg

.DEFAULT_GOAL := help
.PHONY: help run-notebook check-notebook test lint

help:
	@echo "run-notebook    Execute $(NOTEBOOK) in place, storing its outputs"
	@echo "check-notebook  Run every notebook code cell in-process, via pytest"
	@echo "test            Run the full test suite with doctests"
	@echo "lint            Run ruff"

# Executes with a real Jupyter kernel and writes the outputs back into the
# notebook, so the stored copy shows the legacy-versus-corrected evidence.
run-notebook:
	uv run $(JUPYTER_DEPS) jupyter nbconvert \
		--to notebook \
		--execute \
		--inplace \
		$(NOTEBOOK)

# Faster check with no kernel and no writes: asserts the cells still run and
# that the maintainer narrative is intact.
check-notebook:
	uv run pytest fracridge/tests/test_diagnostic_notebook.py -q

test:
	uv run pytest --pyargs fracridge --doctest-modules -q

lint:
	uv run ruff check .
