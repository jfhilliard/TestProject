PYTHON = /opt/anaconda3/bin/python
PREFIX = $(PYTHONPREFIX)

install:
	python setup.py install --prefix=$(PREFIX)

test: install
	pytest --verbose --cov --cov-report=term-missing
