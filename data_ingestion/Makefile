# Makefile for Capstone Data Ingestion Project

# Conda environment name
ENV_NAME=capstone_mineru

# Paths
REQ_FILE=./requirements.txt
PIPELINE=src/ingest.py

.PHONY: create_env remove_env recreate_env install_requirements install_mineru run_pipeline test activate clean

create_env:
	conda create -y -n $(ENV_NAME) python=3.10

remove_env:
	conda env remove -y -n $(ENV_NAME)

recreate_env:
	conda env remove -y -n $(ENV_NAME)
	conda create -y -n $(ENV_NAME) python=3.10
	conda run -n $(ENV_NAME) pip install --upgrade pip
	conda run -n $(ENV_NAME) pip install uv
	conda run -n $(ENV_NAME) uv pip install -U "mineru[core]"
	conda run -n $(ENV_NAME) pip install -r $(REQ_FILE)

install_requirements:
	conda run -n $(ENV_NAME) pip install -r $(REQ_FILE)


install_mineru:
	pip install --upgrade pip
	pip install uv
	uv pip install -U "mineru[core]"

run_pipeline:
	conda run -n $(ENV_NAME) python $(PIPELINE)

test:
	conda run -n $(ENV_NAME) pytest

activate:
	@echo To activate the conda environment, run:
	@echo activate $(ENV_NAME)

clean:
	@echo Cleaning __pycache__ and .pyc files...
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /r . %%f in (*.pyc) do del /f /q "%%f"
