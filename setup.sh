#!/bin/bash

printf "->Creating the virtual environment\n"
virtualenv --python=python3 ~/.venv-tdi-capstone

printf "->Activating the virtual environment\n"
source ~/.venv-tdi-capstone/bin/activate

printf "->Installing Python modules\n"
pip3 install -r reqs.txt
python3 -m pip install jupyter
ipython kernel install --user --name=.venv-tdi-capstone

source ~/.venv-tdi-capstone/bin/activate
