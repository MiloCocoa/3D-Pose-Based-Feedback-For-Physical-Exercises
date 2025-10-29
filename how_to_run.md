Create venv in python 3.11.9

'$ python3.11 -m venv .venv'

Install requirements

'$ pip install -r requirements_current.txt'

Script 1: Runs many different training and evaluation iterating through different hyperparameters settings; see the generated file; experiment_log_broad_search.txt for the results.

'$ python run_exp.py'

Script 2: Runs one training and evaluation with a single hyperparameters setting; this should get the maximum accuracy at 64.773%

'$python main.py --ckpt='check_point' --hidden=256 --epoch=200 --dropout=0.6 --lr=0.01 --lr_decay=100'