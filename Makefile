.PHONY: test test_full_err


## test-section

test:
	py.test tests/  -q --no-summary --no-header --tb=line

test_full_err:
	py.test tests/ --tb=long
