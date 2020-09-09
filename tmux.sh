#!/usr/bin/env bash

tmux new-session -d 'pipenv run jupyter lab'
tmux split-window -v 'vtop'
tmux split-window -h 'watch -d -n 1 nvidia-smi'
tmux -2 attach-session -d
