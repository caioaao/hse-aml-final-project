# competitive-data-science-final-project

## Project outline

This file is the summary of my project and should be seen as the main source of documentation. Experiments and EDA were made using jupyter notebooks inside [./notebooks](./notebooks). Each notebook is prefixed with a number: the order in which they should be read. To view all, run `jupyter lab` under `<project-root>/notebooks`. The title also contains the analysis step I was in when I wrote the notebook (EDA, feature engineering, hyperparameter tuning).

After exploring using the notebooks, I ported the logic into scripts and plain python code inside [./src](./src) so it's easier to manage the results. I also used [GNU Make](https://www.gnu.org/software/make/) to define and run the steps that need to be run to generate the final results, reports, etc. All I had to do was run `make all` after changes and it would automatically detect what needed to be re-run.

## Environment setup

This submission makes use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/), so to install the dependencies, just run `pipenv install` inside the project directory. After that, run `pipenv shell` and you should be all set.

We also use [GNU Make](https://www.gnu.org/software/make/) to process the data and generate the final solution.

Also make sure the kaggle CLI is configured as it is used to fetch the datasets.

## Generating the Train Set Samples

This was the biggest challenge since I was sure I needed to exploit the way the test set was generated.

The final solution, which apparently fits pretty well with the test set, in summary was to run, for each month:

1. Get the shops that appeared in the previous month
2. Get the items that appeared in the current month
3. Generate all possible pairs with the selected shops and items

The exploration can be seen on [notebook 22](./notebooks/22-eda train test round 2.ipynb).

## Feature sets

Each feature set is generated by a script and the baseline validation score is generated by using an XGB with mostly default parameters and using a time series split with 3 months (31, 32, and 33).

OBS: Best score is in bold

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
| 025 | 023 + 024 | __0.80632__ |

By comparing the baseline validation score I was able to choose feature sets to mix and match until I got to the dataset I used for the final solution: 025.

## Learning Algorithms

This section contains an explanation of each learning algorithm and the HPO strategy used to configure them.

### XGBoost

For HPO I used [optuna](https://optuna.readthedocs.io/en/stable/) with the default TPE sampler and a [HyperBand](https://arxiv.org/abs/1603.06560) pruner.

After choosing most of the hyperparameters using Optuna, I find the optimal number of estimators (or boost rounds) by running train with early stopping. This probably doesn't yield the best result, but I decided to do so anyway to speed up the HPO.

### Linear Regression

This was done using scikit-learn's [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) implementation, aka `SGDRegressor`.

Since this is also a gradient descent method, I decided to implement my own train loop so I could take advantage of optuna's hyperband pruner as well. After finding the optimal hyperparameters, I run a second pass on the best configuration with an increased amount of iterations to find the optimal number of iterations for his configuration.

Just like in the XGBoost, doing the optimal number of iterations separate from the other hyperparameters is submoptimal, but it was a trade-off I was willing to take to speed up the process.

For preprocessing, a standard linear model preprocessor was put together, doing one-hot encoding for categorical variables and then scaling the dataset variance to 1 (mean couldn't be scaled as well to preserve the sparseness of the train set matrix).

### LightGBM

Used with Optuna for HPO. Optuna's LGBM integration was great since it implements the [stepwise algorithm](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258) paired with random search. It was still slower than the random search for XGBoost and SGD, but the results were good.

## Experiments

We define an experiment as a feature-set + an algorithm. Their evaluation are the validation score and the public LB score (also our generalization score). Since our amount of submissions is limited I didn't send every result to the competition and used my validation score to choose the ones I submitted.

| ID | Algorithm | Feature set | Validation Score | Public LB score |
| -- | -- | -- | -- | -- |
| 000 | XGB | 000 | 1.04264 |  |
| 004 | XGB | 025 | __0.79638__ | __0.91381__ |
| 006 | LGB | 025 | 0.80421 | 0.91941 |

## Stacking

For stacking I created cross-validation predictions for the last 8 months in the train set using a rolling window of 16 months. For instance, train on months 14 to 30 and generate predictions to month 31, then train on months 15 to 31 and generate predictions to month 32, etc.

After that I used this to train the estimator on the second layer. For validation score I trained on the first 7 months and predicted on the 8th.

| ID | Layer 0 IDs | Meta Estimator | Validation Score | Public LB Score |
| -- | -- | -- | -- |
| 0 | 004, 006 | SGD | __0.77996__ | __0.91197__ |
| 1 | 004, 006 | XGB (small) | 0.78051 | 0.91361 |
