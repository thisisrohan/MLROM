#!/bin/sh

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 0 0 10 20 30 40

sleep 5m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 0 0

sleep 5m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_optim.py 0 10 1600

sleep 1m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_individual.py 0 2

sleep 1m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_AR_optim.py 0 2 10 20 30 40

sleep 5m

python /home/rkaushik/Documents/Thesis/MLROM/new_lorenz/esn_ae_ph_combined.py 0 2
