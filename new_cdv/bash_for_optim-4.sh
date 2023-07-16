#!/bin/sh

# GRU 20*num_ls single
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 100 0
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 0 -e 150 -p 10
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 0
sleep 1m

# GRU 50*num_ls single
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 250 0
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 1 -e 150 -p 10
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 1
sleep 1m

# GRU 80*num_ls single
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 0
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 2 -e 150 -p 10
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 2
sleep 1m

# GRU 80*num_ls Euler
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 1
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 3 -e 200 -p 20
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 3
sleep 1m

# GRU 80*num_ls RK2
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 1 0.5 0.5
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 4 -e 200 -p 20
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 4
sleep 1m

# GRU 80*num_ls RK4
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 0.5 0.0 0.5 0.0 0.0 1.0 1/6 1/3 1/3 1/6
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 5 -e 200 -p 20
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 5
sleep 1m

# LSTM 80*num_ls single
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 0 -s LSTM_SingleStep_v1 RNN_LSTM -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 6 -e 150 -p 10 -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 6 -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM
sleep 1m

# SimpleRNN 80*num_ls single
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-1.py 400 0 -s SimpleRNN_SingleStep_v1 RNN_SimpleRNN -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/optim-4.py 7 -e 150 -p 10 -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN
sleep 1m
python /home/rkaushik/Documents/Thesis/MLROM/new_cdv/LoadingARtrained-combinedAERNN-PredictionHorizons.py 7 -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN
