PYTHON      ?= ./venv/bin/python
PIP         ?= ./venv/bin/pip
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
COV_TARGETS  = --cov=agents --cov=utils --cov=formats --cov=config --cov=main_test

.PHONY: install test test-fast coverage lint run

install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	# requirements.txt pins torch==2.8.0+cu128 (CUDA build, PyTorch index only);
	# install the CPU build of the same version instead, then everything else.
	$(PIP) install torch==2.8.0 torchvision==0.23.0 --index-url $(TORCH_INDEX)
	grep -vE '^(torch|torchvision)==' requirements.txt > .requirements.notorch.txt
	$(PIP) install -r .requirements.notorch.txt -r requirements-dev.txt
	rm -f .requirements.notorch.txt

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

coverage:
	$(PYTHON) -m pytest $(COV_TARGETS) --cov-report=term-missing --cov-report=html

lint:
	$(PYTHON) -m pylint --disable=all --enable=E,F agents utils formats config.py main_test.py

run:
	$(PYTHON) main_test.py
