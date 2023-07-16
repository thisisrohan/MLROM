#!/bin/sh

# AR AE-ESN 200*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_AR_optim.py 0 8 20 40 60 80 -e 200 -p 20
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_ph_combined.py 0 3

# AR AE-ESN 500*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_AR_optim.py 0 9 20 40 60 80 -e 200 -p 20
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_ph_combined.py 0 4

# AR AE-ESN 800*n_ls
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_AR_optim.py 0 10 20 40 60 80 -e 200 -p 20
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/esn_ae_ph_combined.py 0 5
