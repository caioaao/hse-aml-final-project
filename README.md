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

Each feature set is generated by a script and the baseline validation score is generated by using an XGB with mostly default parameters and using a time series split with 3 months (31, 32, and 33).

| ID | Description | Baseline validation score |
| -- | -- | -- |
| 000 | baseline: item ids, shop ids, date block num | 1.04490 |
| 001 | month number, year number | 1.07931 |
| 002 | lagged item counts | 0.86927 |
| 003 | item categories metadata | 1.01996 |
| 004 | target encoding using `item_id`, `shop_id`, `category_name`, and `subcategory_name`  | 0.85031 |
| 005 | external indicators: RUB to USD/CNY/EUR conversions, MOEX (Moscow Exchange) index (lagged and same month values) | 1.07913 |
| 006 | 000 + 001 + 002 + 003 + 004 + 005 | 0.85090 |
| 007 | 002 after feature selection | 0.86913 |
| 008 | 000 + 001 + 003 + 004 + 005 + 007 | 0.85346 |
| 009 | Median prices for item, item+shop, category, shop+category | 1.03268 |
| 010 | Lagged ranks for item price and item cnt over item, category, shop+category | 0.87124 |
| 011 | 000 + 001 + 003 + 004 + 009 + 010 | 0.84906 |
| 012 | 004 + 007 + 009 + 010 | 0.85011 |
| 013 | 008 + 009 + 010 | 0.84963 |
| 014 | 008 + 009 | 0.85240 |
| 015 | 008 + 010 | 0.84892 |
| 016 | deltas for item sales | 1.07948 |
| 017 | 015 + 016 | __0.84365__ |

OBS: Best score is in bold

### Experiments

We define an experiment as a feature-set + an algorithm + tuned hyperparameters. Their evaluation are the validation score and the public LB score (also our generalization score). Since our amount of submissions is limited I didn't send every result to the competition and used my validation score to choose the ones I submitted.

All HPO were done using [optuna](https://optuna.readthedocs.io/en/stable/), and all validation was made by the same script.

| ID | Algorithm | Feature set | Validation Score | Public LB score |
| -- | -- | -- | -- | -- |
| 000 | XGB | 000 | 1.04383 | 1.18194 |
| 001 | XGB | 008 | __0.83436__ | __0.97848__ |
| 003 | XGB | 015 | 0.85073 | 0.98583 |
