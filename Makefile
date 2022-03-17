.PHONY: test test_full_err


## test-section

test:
	PYTHONPATH=. python -m pytest tests/  -q  --no-header --tb=line --cov=comic_cover_generator/

test_full_err:
	PYTHONPATH=. python -m pytest tests/ --tb=long --cov=comic_cover_generator/
