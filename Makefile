.PHONY: clean deepclean dev pre-commit lint black mypy flake8 pylint pip-check-reqs toml-sort test build upload docs devdocs

# Use pipenv when not in CI environment and pipenv command exists.
CI=false
PIPRUN := pipenv run
PKGDIR := unirobot


# Remove common intermediate files.
clean:
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -rf
	find . -name '__pycache__' -print0 | xargs -0 rm -rf
	-rm -rf \
		*.egg-info \
		.coverage \
		.eggs \
		.mypy_cache \
		.pytest_cache \
		Pipfile* \
		build \
		dist \
		output \
		public

# Remove common intermediate files alongside with `pre-commit` hook and virtualenv created by `pipenv`.
deepclean: clean
	-pre-commit uninstall --hook-type pre-push
	-pipenv --venv >/dev/null 2>&1 && pipenv --rm

# Prepare virtualenv.
# - Create virtual environment with pipenv and conda python when
#   - Not in CI environment.
#   - No existing venv.
venv:
	-[ "${CI}" != "true" ] && ! pipenv --venv >/dev/null 2>&1 && pipenv --site-packages

# Install package in editable mode without dev packages.
install: venv
	${PIPRUN} pip install --no-build-isolation  -e .

# Prepare dev environments:
# - Install package in editable mode alongside with dev requirements.
# - Install pre-commit hoook when not in CI environment.
dev: venv
	${PIPRUN} pip install --no-build-isolation  -e . -r requirements-dev.txt -r requirements.txt
	-[ "${CI}" != "true" ] && pre-commit install --hook-type pre-push

# Run pre-commit for all files.
pre-commit:
	pre-commit run --all-files

# Lint with all tools: black, mypy, flake8, pylint and toml-sort.
lint: mypy flake8 pylint toml-sort

# Code formatter.
black:
	${PIPRUN} python -m black setup.py tests ${PKGDIR}

# Static typing checker.
mypy:
	${PIPRUN} python -m mypy setup.py tests ${PKGDIR}

# Style checker with various of plugins.
flake8:
	${PIPRUN} python -m flake8

# Static code analysis.
pylint:
	${PIPRUN} python -m pylint .

# [Experimental] Check missing/redundant requirements.
pip-check-reqs:
	${PIPRUN} pip-missing-reqs ${PKGDIR}
	${PIPRUN} pip-extra-reqs ${PKGDIR}

# Sort and format toml files (especially for pyproject.toml).
toml-sort:
	${PIPRUN} toml-sort -a -i pyproject.toml

# Trigger tests.
test:
	${PIPRUN} python -m pytest --cov=${PKGDIR}  .

# Build package.
build:
	${PIPRUN} python -m build

# Upload package.
upload:
	${PIPRUN} python -m twine upload dist/*

# Generate docs.
docs:
	${PIPRUN} python -m sphinx.cmd.build docs public

# Auto build docs.
devdocs:
	${PIPRUN} python -m sphinx_autobuild docs public
