SHELL=/bin/bash

# TODO produce the zip. includ full code. 
material.zip:
	zip -r material.zip code data readme.md requirements.txt Makefile LICENSE .gitignore

# Install the venv. Depending on the mashine, one my need to adjust the requirements.txt file.
venv:
	rm -rf .venv
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train:
	.venv/bin/python code/training.py

apply:
	.venv/bin/python code/apply.py
