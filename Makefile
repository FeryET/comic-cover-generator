.PHONY: test test_full_err
.PHONY: create_conda_env

## test-section

test:
	PYTHONPATH=. python -m pytest tests/  -q  --no-header --tb=line --cov=comic_cover_generator/

test_full_err:
	PYTHONPATH=. python -m pytest tests/ --tb=long --cov=comic_cover_generator/

.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

create_conda_env:
	conda env create -f environment.yml
	$(CONDA_ACTIVATE) comic_cover_generator
	pip install --upgrade pip
	pip install -e '.[dev]'
