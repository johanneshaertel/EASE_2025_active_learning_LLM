# Overview

This material is deployed via [GitHub](https://github.com/johanneshaertel/EASE_2025_active_learning_LLM)
and [DOI](https://doi.org/10.6084/m9.figshare.28303904)

Welcome to the repository hosting the material for the paper: **Improved Labeling of Security Defects in Code Review by Active Learning with LLMs**.

> **IMPORTANT:** The GitHub version of this material misses the big files **data/reviews.zip** and **data/model.h5/**. The model can be retrained in around 30 minutes. However, the [DOI](https://doi.org/10.6084/m9.figshare.28303904) hosts the full material, including data/reviews.zip and data/model.h5.

# Structure.
The repository is separated into:
1. **Code** includes all the main scripts used for processing. The most relevant commands are also saved in a Makefile.
2. **Crawler** includes the artifacts that are used for crawling the raw data from GitHub. The code might lose functionality since the GitHub API is a move target. The DOI version includes the raw dataset.
3. **Data** includes (all) data.
4. **Paper** includes the plots integrated into the paper. They can be re-derived from data using the code `generate_empirical.py` and `generate_simulation.py`.

# Core commands.

They have been tested on WSL2 Ubuntu. We provide code for training the LLM, application and simulation.

1. `make venv` Install a python virtual environment in the root under .venv used for the code.
2. `make train` Trains a fresh LLM on the most recent data from data/final.json. The model is saved in data/model.h5
3. `make apply` Applies the trained data/model.h5 (if it exists) to the data/reviews.zip (if it exists). Candidate selection can be hardcoded into the apply.py easily.
4. `make simulations` runs a single simulation.

# The raw reviews (data/reviews.zip)

Following is the metadata for the pull request reviews we use as our unlabeled dataset R.

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


# The simulation (data/simulation/)
The simulation data can also be derived by running `code/simulaitons.py` multiple times. This can be interesting
to experiment with alternatives.
The raw data is stored in `data/simulation/` and
analysis is executed using `code/generate_simulation.py`.

A prototypical record of the simulation is the following:

| **Field**              | **Description**                                                                 | **Value**                  |
|-------------------------|---------------------------------------------------------------------------------|----------------------------|
| `id`                   | Unique identifier for the simulation run.                                       | 0a04c8a9-a48a-4ad7-8cfb-7cc8c746b981 |
| `vars`                 | Number of variables used in the simulation.                                     | 20                         |
| `n`                    | Total number of samples in the dataset.                                         | 1000000                    |
| `n_pos`                | Number of positive samples in the dataset.                                      | 13807                      |
| `n_neg`                | Number of negative samples in the dataset.                                      | 986193                     |
| `balance`             | Ratio shift parameter (not equal to exact balance)                                          | 0.99                       |
| `sampling`             | Sampling method used (e.g., random).                                            | random                     |
| `increment`            | Incremental step size for active learning.                                             | 200                        |
| `iteration`            | Current iteration of the simulation.                                            | 0                          |
| `correlation`          | Whether correlation is considered in the simulation (`True` or `False`).        | True                       |
| `logit_stdev`          | Standard deviation of the logits (label predictability).                            | 0.7                        |
| `epochs`               | Number of epochs used for training.                                             | 5                          |
| `loss_full_dataset`    | Loss value calculated on the full dataset.                                      | 0.19277027249336243        |
| `n_obs`                | Number of observations in the current sample.                                   | 200                        |
| `n_pos_obs`            | Number of positive observations in the current sample.                          | 2                          |
| `n_neg_obs`            | Number of negative observations in the current sample.                          | 198                        |
| `loss_fit_fst`         | Initial loss value during model fitting.                                        | 15.333786964416504         |
| `loss_fit_lst`         | Final loss value after model fitting.                                           | 0.11693047732114792        |