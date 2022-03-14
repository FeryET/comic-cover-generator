.PHONY: test test_full_err


## test-section

test:
	PYTHONPATH=. py.test tests/  -q --no-summary --no-header --tb=line

test_full_err:
	PYTHONPATH=. py.test tests/ --tb=long
