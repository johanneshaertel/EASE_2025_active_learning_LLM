SHELL=/bin/bash

# Produce the zip uploade on DOI: 
material.zip:
	zip -r material.zip code data readme.md requirements.txt Makefile LICENSE .gitignore

# Install the venv. Depending on the mashine, one my need to adjust the requirements.txt file.
venv:
	rm -rf .venv
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Trains the model to produce data/model.h5
train:
	.venv/bin/python code/training.py

# Apply the model to the data/reviews.zip.
apply:
	.venv/bin/python code/apply.py

# Runs a single simulation.
simulations:
	.venv/bin/python code/simulations.py

