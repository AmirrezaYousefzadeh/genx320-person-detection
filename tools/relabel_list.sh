#!/bin/sh
# Relabel the segments listed in a file (tools/record_dataset.py label), NUM_THREADS at a time.
cd "$(dirname "$0")/.." || exit 1
xargs -P "${NUM_THREADS:-2}" -I{} sh -c 'work/venv/bin/python -I tools/record_dataset.py label {} | cut -c1-100' < "$1"
