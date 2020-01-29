#!/bin/bash

printf "->Creating the virtual environment\n"
virtualenv ~/.venv-tdi-capstone

printf "->Activating the virtual environment\n"
source ~/.venv-tdi-capstone/bin/activate

printf "->Installing Python modules\n"
pip install -r reqs.txt
python -m pip install jupyter
ipython kernel install --user --name=.venv-tdi-capstone

source ~/.venv-tdi-capstone/bin/activate
