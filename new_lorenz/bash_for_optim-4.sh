#!/bin/sh

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-4.py 3 -e 200 -p 20

sleep 5m

python /home/rkaushik/Documents/MLROM/kolmogorov/LoadingARtrained-combinedAERNN-PredictionHorizons.py 4

sleep 2m

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-4.py 4 -e 200 -p 20

sleep 5m

python /home/rkaushik/Documents/MLROM/kolmogorov/LoadingARtrained-combinedAERNN-PredictionHorizons.py 5

sleep 2m

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-4.py 5 -e 200 -p 20

sleep 5m

python /home/rkaushik/Documents/MLROM/kolmogorov/LoadingARtrained-combinedAERNN-PredictionHorizons.py 6