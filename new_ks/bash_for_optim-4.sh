#!/bin/sh

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-1.py 2160 0 -s LSTM_SingleStep_v1 RNN_LSTM -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM

# sleep 2m

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-4.py 6 -e 150 -p 10 -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM

# sleep 5m

python /home/rkaushik/Documents/MLROM/kolmogorov/LoadingARtrained-combinedAERNN-PredictionHorizons.py 7 -r LSTM_AR_v1 AR_RNN_LSTM -a AELSTM_AR_v1 AR_AERNN_LSTM

# sleep 2m

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-1.py 2160 0 -s SimpleRNN_SingleStep_v1 RNN_SimpleRNN -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN

# sleep 2m

python /home/rkaushik/Documents/MLROM/kolmogorov/optim-4.py 7 -e 150 -p 10 -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN

# sleep 5m

python /home/rkaushik/Documents/MLROM/kolmogorov/LoadingARtrained-combinedAERNN-PredictionHorizons.py 8 -r SimpleRNN_AR_v1 AR_RNN_SimpleRNN -a AESimpleRNN_AR_v1 AR_AERNN_SimpleRNN