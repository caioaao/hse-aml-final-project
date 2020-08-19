#!/usr/bin/env bash

module=`echo "${1}" | sed 's,.py,,g;s,/,.,g'`
python -m "${module}" "${@:2}"
