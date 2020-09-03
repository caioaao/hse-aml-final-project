# competitive-data-science-final-project

## Environment setup

This submission makes use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/), so to install the dependencies, just run `pipenv install` inside the project directory. After that, run `pipenv shell` and you should be all set.

We also use [GNU Make](https://www.gnu.org/software/make/) to process the data and generate the final solution.

Also make sure the kaggle CLI is configured as it is used to fetch the datasets.

## Documentation

The whole process is documented using jupyter notebooks inside [./notebooks](./notebooks). Each notebook is prefixed with a number: the order in which they should be read. To view all, run `jupyter lab` under `<project-root>/notebooks`. The title also contains the analysis step I was in when I wrote the notebook (EDA, feature engineering, hyperparameter tuning).

## Models

After experimenting with the notebooks, I wrote the scripts inside [./src](./src). Then the dependency graph is declared using the [Makefile](./Makefile). After setting up the environment, all you need to do is run `make final-solution` and it will download the datasets, process them, train the models and output the submission.

### Feature sets

Each feature set builds on top of the last

| ID | Description |
| -- | -- |
| 000 | baseline: item ids, shop ids, date block num |
| 001 | month number, year number, lagged item counts |
| 002 | item categories metadata |
| 003 | encoding for shops using previous month's item category mean sales |
| 004 | external indicators: lagged RUB to USD/CNY/EUR conversions, lagged MOEX (Moscow Exchange) index |
| 005 | |

### Experiments

We define an experiment as a feature-set + an algorithm + tuned hyperparameters. Their evaluation are the validation score and the public LB score (also our generalization score). Since our amount of submissions is limited I didn't send every result to the competition and used my validation score to choose the ones I submitted.

All hyperparameters were tuned using optuna, and all validation was made by the same script.

| ID | Algorithm | Feature set | Validation Score | Public LB score |
| -- | -- | -- | -- | -- |
| 000 | XGB | 000 | 1.04507 | 1.18114 |
| 001 | XGB | 001 | 0.86231 | 1.01806 |
