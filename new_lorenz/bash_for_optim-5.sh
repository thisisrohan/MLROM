#!/bin/sh

# AR AE-ESN 200*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 0 3 15 20 25 30 -e 150 -p 10
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 0 6

# AR AE-ESN 500*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 0 4 15 20 25 30 -e 150 -p 10
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 0 7

# AR AE-ESN 800*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 0 5 15 20 25 30 -e 150 -p 10
python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 0 8
