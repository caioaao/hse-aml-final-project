#!/usr/bin/env bash
mkdir -p .log
module=`echo "${1}" | sed 's,.py,,g;s,/,.,g'`
log_dir=".log/`uuidgen --time`.txt"
python -m "${module}" "${@:2}" | tee "${log_dir}"
