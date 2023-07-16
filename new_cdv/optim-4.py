#!/usr/bin/env python
# coding: utf-8

import os
import math
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import time as time
import platform as platform

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2
from keras.engine import data_adapter
import h5py

import importlib

tf.keras.backend.set_floatx('float32')

plt.rcParams.update({
    "text.usetex":True,
    "font.family":"serif"
})

from numpy import *

FTYPE = np.float32
ITYPE = np.int32

array = np.array
float32 = np.float32
int32 = np.int32
float64 = np.float64
int64 = np.int64

def main(rnn_idx, epochs, patience, AR_AERNN_str, AR_RNN_str):
    strategy = None
    # strategy = tf.distribute.MirroredStrategy()

    current_sys = platform.system()

    if current_sys == 'Windows':
        dir_sep = '\\'
    else:
        dir_sep = '/'

    print(os.getcwd())

    from tools.misc_tools import create_data_for_RNN, mytimecallback, SaveLosses, plot_losses, readAndReturnLossHistories, plot_histogram_and_save
    
    from tools.ae_v2 import Autoencoder
    
    # from tools.GRU_AR_v1 import AR_RNN_GRU as AR_RNN
    # from tools.AEGRU_AR_v1 import AR_AERNN_GRU as AR_AERNN
    _temp2 = importlib.__import__('tools.'+AR_RNN_str[0], globals(), locals(), [AR_RNN_str[1],], 0)
    AR_RNN = eval('_temp2.'+AR_RNN_str[1])
    
    _temp3 = importlib.__import__('tools.'+AR_AERNN_str[0], globals(), locals(), [AR_AERNN_str[1],], 0)
    AR_AERNN = eval('_temp3.'+AR_AERNN_str[1])
    
    from tools.trainAERNN import trainAERNN

    behaviour = 'initialiseAndTrainFromScratch'
    # behaviour = 'loadCheckpoin?tAndContinueTraining'
    # behaviour = 'loadFinalNetAndPlot'

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)

    if strategy is None:
        if gpus:
            gpu_to_use = 0
            tf.config.set_visible_devices(gpus[gpu_to_use], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print(logical_devices)

    print(tf.config.list_physical_devices())
    print('')
    print(tf.config.list_logical_devices())
    print('')
    print(tf.__version__)


    ###--- Kolmogorov Flow system ---###

    # setting up params (and saving, if applicable)
    if behaviour == 'initialiseAndTrainFromScratch':
        # RNN directory
        dir_name_rnn = os.getcwd()+'/saved_rnn/rnn_{:03d}'.format(rnn_idx)

        # making AR-RNN save directory
        dir_name_ARrnn = os.getcwd() + dir_sep + 'saved_AR_AERNN_rnn'
        if not os.path.isdir(dir_name_ARrnn):
            os.makedirs(dir_name_ARrnn)

        counter = 0
        while True:
            dir_check = 'AR_rnn_' + str(counter).zfill(3)
            if os.path.isdir(dir_name_ARrnn + dir_sep + dir_check):
                counter += 1
            else:
                break

        dir_name_ARrnn = dir_name_ARrnn + dir_sep + dir_check
        os.makedirs(dir_name_ARrnn)
        os.makedirs(dir_name_ARrnn+dir_sep+'plots')
        
        # reading RNN paramaters
        with open(dir_name_rnn + '/RNN_specific_data.txt') as f:
            lines = f.readlines()

        params_rnn_dict = eval(''.join(lines))

        dt_rnn = params_rnn_dict['dt_rnn']
        return_params_arr = False
        params = None
        try:
            normalize_dataset = params_rnn_dict['normalize_dataset']
        except:
            print("'normalize_dataset' not present in rnn_specific_data, set to False.")
            normalize_dataset = False
        try:
            stddev_multiplier = params_rnn_dict['stddev_multiplier']
        except:
            print("'stddev_multiplier' not present in RNN_specific_data, set to None.")
            stddev_multiplier = None
        try:
            skip_intermediate = params_rnn_dict['skip_intermediate']
        except:
            print("'skip_intermediate' not present in RNN_specific_data, set to 1.")
            skip_intermediate = 1
        try:
            normalization_type = params_rnn_dict['normalization_type']
        except:
            print("'normalization_type' not present in RNN_specific_data, set to 'stddev'.")
            normalization_type = 'stddev'
        try:
            dense_layer_act_func = params_rnn_dict['dense_layer_act_func']
        except:
            print("'dense_layer_act_func' not present in RNN_specific_data, set to 'linear'.")
            dense_layer_act_func = 'linear'
        try:
            stateful = params_rnn_dict['stateful']
        except:
            print("'stateful' not present in RNN_specific_data, set to True.")
            stateful = True
        try:
            use_learnable_state = params_rnn_dict['use_learnable_state']
        except:
            print("'use_learnable_state' not present in RNN_specific_data, set to False.")
            use_learnable_state = False
        try:
            use_weights_post_dense = params_rnn_dict['use_weights_post_dense']
        except:
            print("'use_weights_post_dense' not present in RNN_specific_data, set to False.")
            use_weights_post_dense = False
        try:
            use_trainable_weights_with_reslayers = params_rnn_dict['use_trainable_weights_with_reslayers']
        except:
            print("'use_trainable_weights_with_reslayers' not present in RNN_specific_data, set to False.")
            use_trainable_weights_with_reslayers = False
            
        
        # training params
        with open(dir_name_rnn + dir_sep + 'training_specific_params.txt') as f:
            lines = f.readlines()

        tparams_dict = eval(''.join(lines))

        prng_seed = tparams_dict['prng_seed']
        train_split = tparams_dict['train_split']
        val_split = tparams_dict['val_split']
        batch_size = tparams_dict['batch_size']
        try:
            fRMS = tparams_dict['fRMS']
        except:
            fRMS = 0.0

        loss_weights = 0.98
    else:
        # AR-RNN directory
        dir_name_ARrnn = os.getcwd()+'/saved_AR_AERNN_rnn/AR_AErnn_000'

        # reading AR-RNN parameters
        with open(dir_name_ARrnn + '/AR_rnn_specific_data.txt') as f:
            lines = f.readlines()
        
        params_AR_rnn_dict = eval(''.join(lines))

        dir_name_rnn = params_AR_rnn_dict['dir_name_rnn']
        rnn_idx = dir_name_rnn[-3:]
        dir_name_rnn = os.getcwd()+'/saved_rnn/rnn_'+rnn_idx

        dt_rnn = params_AR_rnn_dict['dt_rnn']
        T_sample_input = params_AR_rnn_dict['T_sample_input']
        T_sample_output = params_AR_rnn_dict['T_sample_output']
        T_offset = params_AR_rnn_dict['T_offset']
        return_params_arr = params_AR_rnn_dict['return_params_arr']
        params = params_AR_rnn_dict['params']
        try:
            normalize_dataset = params_AR_rnn_dict['normalize_dataset']
        except:
            print("'normalize_dataset' not present in AR_rnn_specific_data, set to False.")
            normalize_dataset = False
        try:
            stddev_multiplier = params_AR_rnn_dict['stddev_multiplier']
        except:
            print("'stddev_multiplier' not present in RNN_specific_data, set to None.")
            stddev_multiplier = None
        try:
            skip_intermediate = params_AR_rnn_dict['skip_intermediate']
        except:
            print("'skip_intermediate' not present in RNN_specific_data, set to 1.")
            skip_intermediate = 1
        try:
            use_ae_data = params_AR_rnn_dict['use_ae_data']
        except:
            print("'use_ae_data' not present in RNN_specific_data, set to True.")
            use_ae_data = True

        # training params
        with open(dir_name_ARrnn + dir_sep + 'training_specific_params.txt') as f:
            lines = f.readlines()

        tparams_dict = eval(''.join(lines))

        learning_rate_list = tparams_dict['learning_rate_list']
        epochs = tparams_dict['epochs']
        patience = tparams_dict['patience']
        min_delta = tparams_dict['min_delta']
        prng_seed = tparams_dict['prng_seed']
        train_split = tparams_dict['train_split']
        val_split = tparams_dict['val_split']
        batch_size = tparams_dict['batch_size']
        try:
            fRMS = tparams_dict['fRMS']
        except:
            fRMS = 0.0
        try:
            loss_weights = tparams_dict['loss_weights']
        except:
            loss_weights = None
        if 'freeze_layers' in tparams_dict.keys():
            freeze_layers = tparams_dict['freeze_layers']
        else:
            freeze_layers = None
        if 'clipnorm' in tparams_dict.keys():
            clipnorm = tparams_dict['clipnorm']
        else:
            clipnorm = None

    # reading stddev
    with open(dir_name_rnn + '/final_net/final_net_class_dict.txt') as f:
        lines = f.readlines()
    finalnet_dict = eval(''.join(lines))
    stddev = finalnet_dict['stddev']
    # stddev = 0.0

    # reading RNN normalization constants
    normalization_arr_rnn = None
    if normalize_dataset == True:
        with open(dir_name_rnn + '/final_net/rnn_normalization.txt') as f:
            lines = f.readlines()
        normarr_rnn_dict = eval(''.join(lines))
        normalization_arr_rnn = normarr_rnn_dict['normalization_arr']

    if os.path.exists(dir_name_rnn+dir_sep+'normalization_data.npz'):
        with np.load(dir_name_rnn+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
            normalization_arr_rnn = fl['normalization_arr'][0]

    # reading AE directory
    with open(dir_name_rnn + '/sim_data_AE_params.txt') as f:
        lines = f.readlines()

    params_dict = eval(''.join(lines))

    dir_name_ae = params_dict['dir_name_ae']
    ae_idx = dir_name_ae[-3:]
    dir_name_ae = os.getcwd()+'/saved_ae/ae_'+ae_idx
    try:
        use_ae_data = params_dict['use_ae_data']
    except:
        print("'use_ae_data' not present in sim_data_AE_params, set to True.")
        use_ae_data = True

    # reading simulation parameters
    with open(dir_name_ae + dir_sep + 'ae_data.txt') as f:
        lines = f.readlines()
    params_dict = eval(''.join(lines))
    data_dir_idx = params_dict['data_dir_idx']
    normalizeforae_flag = params_dict['normalizeforae_flag']
    normalization_constant_arr_aedata = params_dict['normalization_constant_arr_aedata']
    try:
        ae_data_with_params = params_dict['ae_data_with_params']
    except:
        print("'ae_data_with_params' not present in ae_data, set to 'True'.")
        ae_data_with_params = True

    if os.path.exists(dir_name_ae+dir_sep+'normalization_data.npz'):
        with np.load(dir_name_ae+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
            normalization_constant_arr_aedata = fl['normalization_constant_arr_aedata'][0]

    print('dir_name_AR_AErnn:', dir_name_ARrnn)
    print('dir_name_rnn:', dir_name_rnn)
    print('dir_name_ae:', dir_name_ae)
    print('data_dir_idx:', data_dir_idx)

    # loading data
    dir_name_data = os.getcwd() + dir_sep + 'saved_data' + dir_sep + 'data_' + data_dir_idx

    with open(dir_name_data + dir_sep + 'sim_data_params.txt') as f:
        lines = f.readlines()
    params_dict = eval(''.join(lines))
    params_mat = params_dict['params_mat']
    # init_state = params_dict['init_state']
    t0 = params_dict['t0']
    T = params_dict['T']
    delta_t = params_dict['delta_t']
    return_params_arr = params_dict['return_params_arr']
    normalize_flag_ogdata = params_dict['normalize_flag']
    print('normalize_flag_ogdata:', normalize_flag_ogdata)
    alldata_withparams_flag = params_dict['alldata_withparams_flag']

    with np.load(dir_name_data+dir_sep+'data.npz', allow_pickle=True) as fl:
        all_data = fl['all_data'].astype(FTYPE)
        boundary_idx_arr = fl['boundary_idx_arr']
        normalization_constant_arr_ogdata = fl['normalization_constant_arr'][0]
        initial_t0 = fl['initial_t0']
        init_state_mat = fl['init_state_mat']

        lyapunov_spectrum_mat = fl['lyapunov_spectrum_mat']

    lyapunov_time_arr = np.empty(shape=lyapunov_spectrum_mat.shape[0], dtype=FTYPE)
    for i in range(lyapunov_spectrum_mat.shape[0]):
        lyapunov_time_arr[i] = 1/lyapunov_spectrum_mat[i, 0]
        print('Case : {}, lyapunov exponent : {}, lyapunov time : {}s'.format(i+1, lyapunov_spectrum_mat[i, 0], lyapunov_time_arr[i]))

    print('all_data.shape : ', all_data.shape)

    all_data_shape_og = all_data.shape[1:]

    test_split = 1 - train_split - val_split

    # setting seed for PRNGs
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)

    ###--- GRU ---###
    if behaviour == 'initialiseAndTrainFromScratch':
        # RNN data parameters
        num_lyaptimesteps_totrain = np.array([
            10,
            20, 
            40,
            60,
            # 80,
        ])*dt_rnn/np.mean(lyapunov_time_arr)
        num_timesteps_warmup = 1*np.mean(lyapunov_time_arr)/dt_rnn
        T_sample_input = num_timesteps_warmup*dt_rnn
        T_sample_output = num_lyaptimesteps_totrain*np.mean(lyapunov_time_arr)
        T_offset = T_sample_input
        skip_intermediate = 'full sample'
        stateful = True
        if return_params_arr != False:
            params = params_arr
        else:
            params = None

        # saving AR RNN specific data
        AR_RNN_specific_data = {
            'dt_rnn':dt_rnn,
            'T_sample_input':T_sample_input,
            'T_sample_output':T_sample_output,
            'T_offset':T_offset,
            'boundary_idx_arr':boundary_idx_arr,
            'delta_t':delta_t,
            # 'params':params,
            # 'return_params_arr':return_params_arr,
            'normalize_dataset':normalize_dataset,
            'num_lyaptimesteps_totrain':num_lyaptimesteps_totrain,
            'num_timesteps_warmup':num_timesteps_warmup,
            'dir_name_rnn':dir_name_rnn,
            'dir_name_ae':dir_name_ae,
            'stddev_multiplier':stddev_multiplier,
            'skip_intermediate':skip_intermediate,
            'module':AR_RNN.__module__,
            'normalization_type':normalization_type,
            'use_ae_data':use_ae_data,
            'stateful':stateful,
        }

        with open(dir_name_ARrnn+dir_sep+'AR_RNN_specific_data.txt', 'w') as f:
            f.write(str(AR_RNN_specific_data))

    # setting up training params
    if behaviour == 'initialiseAndTrainFromScratch':
        learning_rate_list = [
            [1e-3, 5e-4, 1e-4],
            [1e-4, 5e-5, 1e-5],
            [1e-5, 5e-6, 1e-6],
            [1e-6, 5e-7, 1e-7],
            # [5e-7],
        ]
        epochs = [
            [epochs]*len(learning_rate_list[0]),
            [epochs]*len(learning_rate_list[1]),
            [epochs]*len(learning_rate_list[2]),
            [epochs]*len(learning_rate_list[3]),
            # [1000],
        ]
        patience = [
            [patience]*len(learning_rate_list[0]),
            [patience]*len(learning_rate_list[1]),
            [patience]*len(learning_rate_list[2]),
            [patience]*len(learning_rate_list[3]),
            # [50],
        ] # parameter for early stopping
        min_delta = 5e-6  # parameter for early stopping
        lambda_reg = 1e-7  # weight for regularizer
        covmat_lmda = 1e-4  # weight for the covmat loss

        if loss_weights is None:
            loss_weights = 1.0
            
        freeze_layers = [
            [],
            [],
            [],
            [],
            [],
        ]
        
        clipnorm = None #1.0
        batch_size = [
            16,
            16,
            16,
            16,
            # 32,
        ]
        
        # saving training params
        training_specific_params = {
            'learning_rate_list':learning_rate_list,
            'epochs':epochs,
            'patience':patience,
            'min_delta':min_delta,
            'prng_seed':prng_seed,
            'train_split':train_split,
            'val_split':val_split,
            'batch_size':batch_size,
            'fRMS':fRMS,
            'loss_weights':loss_weights,
            'stddev':stddev,
            'covmat_lmda':covmat_lmda,
            'freeze_layers':freeze_layers,
            'clipnorm':clipnorm,
        }

        with open(dir_name_ARrnn+dir_sep+'training_specific_params.txt', 'w') as f:
            f.write(str(training_specific_params))
        
        np.savez(
            dir_name_ARrnn+dir_sep+'normalization_data',
            normalization_arr=[normalization_arr_rnn],
        )

    if behaviour == 'initialiseAndTrainFromScratch':
        load_file_rnn = dir_name_rnn + '/final_net/final_net_class_dict.txt'
        wt_file_rnn = dir_name_rnn+'/final_net/final_net_gru_weights.h5'
        
        load_file_ae = dir_name_ae+'/final_net/final_net_class_dict.txt'
        wt_file_ae = dir_name_ae+'/final_net/final_net_ae_weights.h5'

    global_clipnorm = None
    for kk in range(len(T_sample_output)):

        num_outsteps = int((T_sample_output[kk] + 0.5*dt_rnn)//dt_rnn)
        if type(freeze_layers) == type(None):
            freeze_layers_thisoutstep = []
        else:
            if kk > len(freeze_layers) - 1:
                freeze_layers_thisoutstep = freeze_layers[-1]
            else:
                freeze_layers_thisoutstep = freeze_layers[kk]
            
            if type(freeze_layers_thisoutstep) == type(None):
                freeze_layers_thisoutstep = []
                
        if type(batch_size) == type([]):
            if kk > len(batch_size) - 1:
                batch_size_thisoutstep = batch_size[-1]
            else:
                batch_size_thisoutstep = batch_size[kk]
        else:
            batch_size_thisoutstep = batch_size

        total_s_len = 80
        sep_lr_s = ' num_outsteps : {} '.format(num_outsteps)
        sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'>' + sep_lr_s
        sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'<'
        print('\n\n' + '*'*len(sep_lr_s))
        print('' + sep_lr_s+'')
        print('*'*len(sep_lr_s) + '\n\n')

        print('clipnorm : {}, global_clipnorm : {}'.format(clipnorm, global_clipnorm))
        
        trainAERNN(
            create_data_for_RNN,
            Autoencoder,
            AR_RNN,
            all_data,
            AR_AERNN,
            dt_rnn=dt_rnn,
            T_sample_input=T_sample_input,
            T_sample_output=T_sample_output[kk],
            T_offset=T_offset,
            boundary_idx_arr=boundary_idx_arr,
            delta_t=delta_t,
            params=params,
            normalize_dataset=normalize_dataset,
            stddev_multiplier=stddev_multiplier,
            skip_intermediate=skip_intermediate,
            normalization_type=normalization_type,
            normalization_constant_arr_aedata=normalization_constant_arr_aedata,
            normalization_constant_arr_rnndata=normalization_arr_rnn,
            learning_rate_list=learning_rate_list[kk],
            epochs=epochs[kk],
            patience=patience[kk],
            loss_weights=loss_weights,
            min_delta=min_delta,
            lambda_reg=lambda_reg,
            stddev_rnn=stddev,
            stateful=False,
            behaviour=behaviour,
            strategy=strategy,
            dir_name_rnn=dir_name_rnn,
            dir_name_AR_AErnn=dir_name_ARrnn,
            batch_size=batch_size_thisoutstep,
            load_file_rnn=load_file_rnn,
            wt_file_rnn=wt_file_rnn,
            load_file_ae=load_file_ae,
            wt_file_ae=wt_file_ae,
            covmat_lmda=covmat_lmda,
            readAndReturnLossHistories=readAndReturnLossHistories,
            mytimecallback=mytimecallback,
            plot_losses=plot_losses,
            SaveLosses=SaveLosses,
            train_split=train_split,
            test_split=test_split,
            val_split=val_split,
            freeze_layers=freeze_layers_thisoutstep,
            clipnorm=clipnorm,
            global_clipnorm=global_clipnorm,
            use_ae_data=use_ae_data,
        )
        
        wt_file_rnn = dir_name_ARrnn+'/final_net/final_net-{}_outsteps_rnn_weights.h5'.format(num_outsteps)
        wt_file_ae = dir_name_ARrnn+'/final_net/final_net-{}_outsteps_ae_weights.h5'.format(num_outsteps)
        
        with open(dir_name_ARrnn+'/final_net/losses-{}_outsteps.txt'.format(num_outsteps), 'r') as fl:
            lines = fl.readlines()

        loss_dict = eval(''.join(lines))
        train_global_gradnorm_hist = loss_dict['train_global_gradnorm_hist']
        # lr_change = loss_dict['lr_change']
        # trained_epochs = len(train_global_gradnorm_hist)
        # if lr_change[-1] - lr_change[-2] == epochs[kk][-1]:
        #     global_clipnorm = train_global_gradnorm_hist[-1]
        # else:
        #     global_clipnorm = train_global_gradnorm_hist[-patience[kk][-1]]

        # alpha1 = 0.9
        # alpha2 = 0.1
        # global_clipnorm = train_global_gradnorm_hist[0]
        # for i in range(1, len(train_global_gradnorm_hist)):
        #     global_clipnorm = alpha1*global_clipnorm + alpha2*train_global_gradnorm_hist[i]

        # grad_norm_decay = 0.95
        # idxs_to_ignore = 0

        # global_clipnorm = np.max(train_global_gradnorm_hist[idxs_to_ignore:])
        # # global_clipnorm = 0.25 * np.round(4*global_clipnorm)
        # global_clipnorm = grad_norm_decay * global_clipnorm
        idxs_to_ignore = 1
        global_clipnorm_min = 3.0
        global_clipnorm = np.max(train_global_gradnorm_hist[idxs_to_ignore:])
        global_clipnorm = 0.1 * np.round(10*global_clipnorm)
        global_clipnorm = max(global_clipnorm, global_clipnorm_min)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rnn_idx', type=int)
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-r', '--arrnn', type=str, nargs='+', default=['GRU_AR_v1', 'AR_RNN_GRU'])
    parser.add_argument('-a', '--araernn', type=str, nargs='+', default=['AEGRU_AR_v1', 'AR_AERNN_GRU'])

    args = parser.parse_args()
    
    print('rnn_idx : {}, epochs : {}, patience : {}'.format(args.rnn_idx, args.epochs, args.patience))
    print('araernn : {}, arrnn : {}'.format(args.araernn, args.arrnn))

    main(args.rnn_idx, args.epochs, args.patience, args.araernn, args.arrnn)
