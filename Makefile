PYTHON = /opt/anaconda3/bin/python
PYTEST = /opt/anaconda3/bin/pytest
PREFIX = $(PYTHONPREFIX)

install:
	$(PYTHON) setup.py install --prefix=$(PREFIX)

test: install
	$(PYTEST) --verbose --cov --cov-report=term-missing

clean:
	-rm -rf build/*
