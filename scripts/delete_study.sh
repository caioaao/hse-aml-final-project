#!/usr/bin/env bash
pipenv run python <<EOF
import optuna
optuna.delete_study('${2}', storage='sqlite:///${1}')
EOF
