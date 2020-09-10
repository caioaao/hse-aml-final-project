#!/usr/bin/env bash

tmux new-session -d
tmux split-window -v 'vtop'
tmux split-window -h 'watch -d -n 1 nvidia-smi'
tmux -2 attach-session -d
