#!/usr/bin/env bash
# Reproduce the whole project: data -> validation -> model -> optimizer.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p outputs data/raw
if [ ! -f data/raw/conjura_mmm_data.csv ]; then
  echo "Downloading Conjura MMM dataset (CC BY 4.0) from Figshare..."
  curl -sL -o data/raw/conjura_mmm_data.csv https://ndownloader.figshare.com/files/46779652
  curl -sL -o data/raw/conjura_mmm_data_dictionary.xlsx https://ndownloader.figshare.com/files/44743624
fi
$PY -W ignore src/data_prep.py
(cd src && ../$PY -W ignore simulate.py)
(cd src && ../$PY -W ignore train.py)
(cd src && ../$PY -W ignore optimizer.py)
echo "Done. Launch the app with:  .venv/bin/streamlit run app/app.py"
