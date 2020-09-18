# competitive-data-science-final-project

## Environment setup

This submission makes use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/), so to install the dependencies, just run `pipenv install` inside the project directory. After that, run `pipenv shell` and you should be all set.

We also use [GNU Make](https://www.gnu.org/software/make/) to process the data and generate the final solution.

Also make sure the kaggle CLI is configured as it is used to fetch the datasets.

## Documentation

The whole process is documented using jupyter notebooks inside [./notebooks](./notebooks). Each notebook is prefixed with a number: the order in which they should be read. To view all, run `jupyter lab` under `<project-root>/notebooks`. The title also contains the analysis step I was in when I wrote the notebook (EDA, feature engineering, hyperparameter tuning).

## Models

After experimenting with the notebooks, I wrote the scripts inside [./src](./src). Then the dependency graph is declared using the [Makefile](./Makefile). After setting up the environment, all you need to do is run `make final-solution` and it will download the datasets, process them, train the models and output the submission.

## Feature sets

Each feature set is generated by a script and the baseline validation score is generated by using an XGB with mostly default parameters and using a time series split with 3 months (31, 32, and 33).

| ID | Description | Baseline validation score |
| -- | -- | -- |
| 000 | baseline: item ids, shop ids, date block num | 1.06215 |
| 001 | month number, year number | 1.09704 |
| 002 | lagged item counts | 0.88733 |
| 003 | item categories metadata | 1.03844 |
| 004 | target encoding using `item_id`, `shop_id`, `category_name`, and `subcategory_name`  | 0.83296 |
| 005 | external indicators: RUB to USD/CNY/EUR conversions, MOEX (Moscow Exchange) index (lagged and same month values) | 1.09703 |
| 006 | 000 + 001 + 002 + 003 + 004 | 0.82393 |
| 007 | 002 after feature selection | 0.88639 |
| 008 | 000 + 001 + 003 + 004 + 007 | 0.82492 |
| 009 | Median prices for item, item+shop, category, shop+category | 1.32087 |
| 010 | Lagged ranks for item price and item cnt over item, category, shop+category | 0.91049 |
| 013 | 008 + 009 + 010 | 0.81898 |
| 014 | 008 + 009 | 0.82728 |
| 015 | 008 + 010 | 0.81816 |
| 016 | deltas for item sales | 0.89988 |
| 017 | 015 + 016 | 0.82299 |
| 018 | deltas for item prices | 1.21004 |
| 019 | 017 + 018 | 0.82063 |
| 020 | revenue and sales / price | 0.88878 |
| 021 | 019 + 020 | 0.82286 |
| 022 | Release dates for item, item+shop, shop | 1.06677 |
| 023 | 021 + 022 | 0.81718 |
| 024 | Months since last seen for item, item+shop | 1.04692 |
| 025 | 023 + 024 | 0.80632 |

OBS: Best score is in bold

## Algorithms

This section contains an explanation of each learning algorithm and the HPO strategy used to configure them.

### XGBoost

Opted for this implementation of gradient-boosted trees since it was easier to run on the GPU, so pretty fast. I also tried LightGBM but for some reason it broke on the larger dataset, so I dropped it.

For HPO I used [optuna](https://optuna.readthedocs.io/en/stable/) with the default TPE sampler and a [HyperBand](https://arxiv.org/abs/1603.06560) pruner.

## Linear Regression

This was done using scikit-learn's [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) implementation, aka `SGDRegressor`.

Since this is also a gradient descent method, I decided to implement my own train loop so I could take advantage of optuna's hyperband pruner as well. After finding the optimal hyperparameters, I run a second pass on the best configuration with an increased amount of iterations to find the optimal number of iterations for his configuration.

For preprocessing, a standard linear model preprocessor was put together, doing one-hot encoding for categorical variables and then scaling the dataset variance to 1 (mean couldn't be scaled as well to preserve the sparseness of the train set matrix).

## Experiments

We define an experiment as a feature-set + an algorithm. Their evaluation are the validation score and the public LB score (also our generalization score). Since our amount of submissions is limited I didn't send every result to the competition and used my validation score to choose the ones I submitted.

| ID | Algorithm | Feature set | Validation Score | Public LB score |
| -- | -- | -- | -- | -- |
| 000 | XGB | 000 | 1.04264 | 1.18194 |
| 004 | XGB | 025 | __0.82619__ | __0.96238__ |
| 005 | SGD | 025 | 0.89283 | 1.04410 |
| 006 | LGB | 025 | | 0.96880 |

## Stacking
