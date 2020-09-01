# competitive-data-science-final-project

## Environment setup

This submission makes use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/), so to install the dependencies, just run `pipenv install` inside the project directory. After that, run `pipenv shell` and you should be all set.

We also use [GNU Make](https://www.gnu.org/software/make/) to process the data and generate the final solution.

Also make sure the kaggle CLI is configured as it is used to fetch the datasets.

## Documentation

The whole process is documented using jupyter notebooks inside [./notebooks](./notebooks). Each notebook is prefixed with a number: the order in which they should be read. To view all, run `jupyter lab` under `<project-root>/notebooks`. The title also contains the analysis step I was in when I wrote the notebook (EDA, feature engineering, hyperparameter tuning).

## Solution

After experimenting with the notebooks, I wrote the scripts inside [./src](./src). Then the dependency graph is declared using the [Makefile](./Makefile). After setting up the environment, all you need to do is run `make final-solution` and it will download the datasets, process them, train the models and output the submission.

### Feature sets

Each feature set builds on top of the last

| ID | Description | CV Score (XGB default params) | Public LB Score |
| -- | -- | -- | -- |
| 001 | lagged item counts, item ids, shop ids, date ids (date block, month id, and year id) | | 1.03884 |
| 002 | item categories metadata | | 0.99800 |
| 003 | encoding for shops using previous month's item category mean sales | | |

### Models
