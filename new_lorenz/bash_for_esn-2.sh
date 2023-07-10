#!/bin/sh

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_optim.py 1 10 1000

sleep 1m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_individual.py 1 1

sleep 1m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 1 1 10 20 30 40

sleep 5m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 1 1
