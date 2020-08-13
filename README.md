# competitive-data-science-final-project

## Environment setup

This submission makes use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/), so to install the dependencies, just run `pipenv install` inside the project directory. After that, run `pipenv shell` and you should be all set.

## Datasets

Datasets are presumed to be available in `.data` directory under the project root. Use kaggle CLI to set it up (also present under `pipenv shell`):

```sh
kaggle competitions download -c competitive-data-science-predict-future-sales -p .data
```

## Solution

The solution was written in several jupyter notebooks. Each notebook is prefixed with a number: the order in which they should be read. To view all, run `jupyter lab` under `<project-root>/notebooks`.
