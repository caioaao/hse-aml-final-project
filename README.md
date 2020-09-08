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

| ID | Description |
| -- | -- |
| 000 | baseline: item ids, shop ids, date block num |
| 001 | month number, year number, lagged item counts |
| 002 | item categories metadata |
| 003 | target encoding using `item_id`, `shop_id`, `category_name`, and `subcategory_name`  |
| 004 | external indicators: RUB to USD/CNY/EUR conversions, MOEX (Moscow Exchange) index (lagged and same month values) |
| 005 | 000 + 001 + 002 + 003 + 004 |
| 006 | 000 + 002 + 003 |
| 007 | 000 + 001 + 002 + 003 |
| 008 | 000 + 003 + 004 |

### Experiments

We define an experiment as a feature-set + an algorithm + tuned hyperparameters. Their evaluation are the validation score and the public LB score (also our generalization score). Since our amount of submissions is limited I didn't send every result to the competition and used my validation score to choose the ones I submitted.

All HPO were done using [optuna](https://optuna.readthedocs.io/en/stable/), and all validation was made by the same script.

| ID | Algorithm | Feature set | Validation Score | Public LB score |
| -- | -- | -- | -- | -- |
| 000 | XGB | 000 | 1.04507 | 1.18114 |
| 001 | XGB | 001 | 0.85824 | 1.01800 |
| 002 | XGB | 002 | 0.93756 | |
| 003 | XGB | 003 | 0.84287 | 0.98811 |
| 004 | XGB | 004 | 1.04352 | |
| 005 | XGB | 005 | 0.85924 | |
| 006 | XGB | 006 | 0.84128 | |
| 007 | XGB | 007 | 0.86218 | 1.02204 |
| 008 | XGB | 008 | 0.83944 | 1.00284 |
