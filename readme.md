# Zip Overview

This material is also be hosted on [https://github.com/johanneshaertel/EASE_2025_active_learning_LLM].

Welcome to the repository hosting the material for the paper **Improved Labeling of Security Defects in Code Review by Active Learning with LLMs**.

# Core commands to run this.

The relevant commands are saved in a Makefile. They have been tested on WSL2 Ubuntu. We provide code for training the LLM, application and simulation.

1. `make venv` Install a python virtual environment in the root under .venv used for the code.
2. `make train` Trains a fresh LLM on the most recent data from data/final.json. The model is saved in data/model.h5
3. `make apply` Run the trained data/model.h5 (potentially not uploaded since around 500 MB) to the data/reviews.zip (potentially not uploaded since around 500 MB). Candidate selection can be hardcoded into the apply.py easily.

# The raw reviews

We deployed the full reviews dataset as part of the file shared on figshare. We do not include the associated code, but the commit SHA and all information on lines that are accessible over the GitHub API. Following is the metadata on the records.

| **Field**             | **Number of reviews with this field**  |
|------------------------|------------|
| `url`                 | 6922420    |
| `path`                | 6922420    |
| `number`              | 6922420    |
| `repo`                | 6922420    |
| `body`                | 6922420    |
| `originalCommit`      | 6374529    |
| `commit`              | 6921490    |
| `createdAt`           | 6922420    |
| `line`                | 2494771    |
| `originalLine`        | 5943671    |
| `originalStartLine`   | 479525     |
| `startLine`           | 127706     |
