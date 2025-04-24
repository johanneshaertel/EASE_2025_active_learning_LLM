# Zip Overview

This material is also be hosted on [https://github.com/johanneshaertel/EASE_2025_active_learning_LLM].

Welcome to the repository hosting the material for the paper **Improved Labeling of Security Defects in Code Review by Active Learning with LLMs**.

# Core commands to run this.

The relevant commands are saved in a Makefile. They have been tested on WSL2 Ubuntu. We provide code for training the LLM, application and simulation.

1. `make venv` Install a python virtual environment in the root under .venv used for the code.
2. `make train` Trains a fresh LLM on the most recent data from data/final.json. The model is saved in data/model.h5
3. `make apply` Run the trained data/model.h5 (potentially not uploaded since around 500 MB) to the data/reviews.zip (potentially not uploaded since around 500 MB). Candidate selection can be hardcoded into the apply.py easily.
