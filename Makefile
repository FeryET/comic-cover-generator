.PHONY: test test_full_err


## test-section

test:
	PYTHONPATH=. pytest tests/  -q  --no-header --tb=line

test_full_err:
	PYTHONPATH=. pytest tests/ --tb=long
