install:
	python setup.py install --prefix=~

test: install
	pytest --verbose --cov --cov-report=term-missing
