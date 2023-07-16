#!/usr/bin/env python
# coding: utf-8
import os
import sys
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
import h5py

import importlib

from numpy import *


def invert_normalization(data, normalization_arr):
    new_data = np.empty_like(data)
    new_data[:] = data[:]
    new_data *= normalization_arr[1]
    new_data += normalization_arr[0]
    return new_data


def main(rnn_layers_units_eachlayer, scalar_weights, RNN_str, AR_AERNN_str, AR_RNN_str):

    tf.keras.backend.set_floatx('float32')

    plt.rcParams.update({
        "text.usetex":True,
        "font.family":"serif",
    })

    FTYPE = np.float32
    ITYPE = np.int32

    array = np.array
    float32 = np.float32
    int32 = np.int32
    float64 = np.float64
    int64 = np.int64

    strategy = None
    # strategy = tf.distribute.MirroredStrategy()

    current_sys = platform.system()

    if current_sys == 'Windows':
        dir_sep = '\\'
    else:
        dir_sep = '/'

    print(os.getcwd())

    from tools.misc_tools import create_data_for_RNN, mytimecallback, SaveLosses, plot_losses, plot_histogram_and_save
    
    from tools.ae_v2 import Autoencoder

    _temp1 = importlib.__import__('tools.'+RNN_str[0], globals(), locals(), [RNN_str[1],], 0)
    RNN_SingleStep = eval('_temp1.'+RNN_str[1])
    
    _temp2 = importlib.__import__('tools.'+AR_RNN_str[0], globals(), locals(), [AR_RNN_str[1],], 0)
    AR_RNN = eval('_temp2.'+AR_RNN_str[1])
    
    _temp3 = importlib.__import__('tools.'+AR_AERNN_str[0], globals(), locals(), [AR_AERNN_str[1],], 0)
    AR_AERNN = eval('_temp3.'+AR_AERNN_str[1])

    behaviour = 'initialiseAndTrainFromScratch'
    # behaviour = 'loadCheckpointAndContinueTraining'
    # behaviour = 'loadFinalNetAndPlot'

    # setting seed for PRNGs
    if behaviour == 'initialiseAndTrainFromScratch':
        prng_seed = 42
        np.random.seed(prng_seed)
        tf.random.set_seed(prng_seed)

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


    ###--- Kolmogorov Flow System ---###

    # setting up params (and saving, if applicable)

    if behaviour == 'initialiseAndTrainFromScratch':
        # making RNN save directory
        dir_name_rnn = os.getcwd() + dir_sep + 'saved_rnn'
        if not os.path.isdir(dir_name_rnn):
            os.makedirs(dir_name_rnn)

        counter = 0
        while True:
            dir_check = 'rnn_' + str(counter).zfill(3)
            if os.path.isdir(dir_name_rnn + dir_sep + dir_check):
                counter += 1
            else:
                break

        dir_name_rnn = dir_name_rnn + dir_sep + dir_check
        os.makedirs(dir_name_rnn)
        os.makedirs(dir_name_rnn+dir_sep+'plots')

        # whether to use AE data or just work on raw data
        use_ae_data = True # if false, specifying ae_idx will only show which dataset to use

        # autoencoder directory
        ae_idx = '008'
        dir_name_ae = os.getcwd()+'{ds}saved_ae{ds}ae_'.format(ds=dir_sep)+ae_idx
    else:
        # RNN directory
        dir_name_rnn = os.getcwd()+'/saved_rnn/rnn_015'

        # reading AE directory
        with open(dir_name_rnn + '/sim_data_AE_params.txt') as f:
            lines = f.readlines()

        params_dict = eval(''.join(lines))

        try:
            use_ae_data = params_dict['use_ae_data']
        except:
            print("'use_ae_data' not present in sim_data_AE_params, set to True.")
            normalize_dataset = True
        
        dir_name_ae = params_dict['dir_name_ae']
        ae_idx = dir_name_ae[-3:]
        dir_name_ae = os.getcwd()+'/saved_ae/ae_'+ae_idx

        # reading RNN paramaters
        with open(dir_name_rnn + '/RNN_specific_data.txt') as f:
            lines = f.readlines()

        params_rnn_dict = eval(''.join(lines))

        dt_rnn = params_rnn_dict['dt_rnn']
        T_sample_input = params_rnn_dict['T_sample_input']
        T_sample_output = params_rnn_dict['T_sample_output']
        T_offset = params_rnn_dict['T_offset']
        return_params_arr = params_rnn_dict['return_params_arr']
        params = params_rnn_dict['params']
        try:
            normalize_dataset = params_rnn_dict['normalize_dataset']
        except:
            print("'normalize_dataset' not present in RNN_specific_data, set to False.")
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
            use_ae_data = params_rnn_dict['use_ae_data']
        except:
            print("'use_ae_data' not present in RNN_specific_data, set to True.")
            use_ae_data = True

        

        normalization_arr = None
        try:
            with open(dir_name_rnn + '/final_net/rnn_normalization.txt') as f:
                lines = f.readlines()
            rnn_norm_arr_dict = eval(lines)
            normalization_arr = rnn_norm_arr_dict['normalization_arr']
        except:
            pass
        if os.path.exists(dir_name_rnn+dir_sep+'normalization_data.npz'):
            with np.load(dir_name_rnn+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
                normalization_arr = fl['normalization_arr'][0]

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

    print('dir_name_rnn:', dir_name_rnn)
    print('use_ae_data : ' + str(use_ae_data) + ', dir_name_ae:', dir_name_ae)
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

    # delaing with normalizing the data before feeding into autoencoder
    num_params = params_mat.shape[1]
    og_vars = all_data.shape[1]
    if alldata_withparams_flag == True:
        og_vars -= num_params

    time_stddev_ogdata = np.std(all_data[:, 0:og_vars], axis=0)
    time_mean_ogdata = np.mean(all_data[:, 0:og_vars], axis=0)
        
    if use_ae_data == True:
        if ae_data_with_params == True and alldata_withparams_flag == False:
            new_all_data = np.empty(shape=(all_data.shape[0], og_vars+num_params), dtype=FTYPE)
            new_all_data[:, 0:og_vars] = all_data[:, 0:og_vars]
            del(all_data)
            all_data = new_all_data
            prev_idx = 0
            for i in range(boundary_idx_arr.shape[0]):
                all_data[prev_idx:boundary_idx_arr[i], num_params:] = params_mat[i]
                prev_idx = boundary_idx_arr[i]

        if normalizeforae_flag == True:
            for i in range(all_data.shape[1]):
                all_data[:, i] -= normalization_constant_arr_aedata[0, i]
                all_data[:, i] /= normalization_constant_arr_aedata[1, i]

        if ae_data_with_params == False:
            all_data = all_data[:, 0:og_vars]
    else:
        # using raw data, neglecting the params attached (if any)
        all_data = all_data[:, 0:og_vars]

    all_data_shape_og = all_data.shape[1:]
    ###--- Autoencoder ---###
    if use_ae_data == True:
        load_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_class_dict.txt'
        wt_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_ae_weights.h5'
        ae_net = Autoencoder(all_data.shape[1], load_file=load_file)
        ae_net.load_weights_from_file(wt_file)

    # create data
    if use_ae_data == True:
        latent_states_all = ae_net.encoder_net.predict(all_data)
        # del(all_data)
    else:
        latent_states_all = all_data
    num_latent_states = latent_states_all.shape[1]

    if behaviour == 'initialiseAndTrainFromScratch':
        # RNN data parameters
        num_lyaptimesteps_totrain = 3 # int(5000/np.mean(lyapunov_time_arr))#
        dt_rnn = 0.5
        T_sample_input = num_lyaptimesteps_totrain*np.mean(lyapunov_time_arr)
        T_sample_output = num_lyaptimesteps_totrain*np.mean(lyapunov_time_arr)
        T_offset = dt_rnn
        normalize_dataset = True # whether the data for the RNN should be normalized by the dataset's mean and std
        normalization_arr = None
        skip_intermediate = 'full sample'
        noise_type = 'normal' # can be 'uniform' or 'normal'

        # can be 'minmax', 'minmax2', 'stddev', or a list with
        # sequential order of any of these; if it is 'minmax'
        # then stddev_multiplier has no effect
        normalization_type = 'stddev'
        stddev_multiplier = 3

        dense_layer_act_func = ['tanh']
        use_weights_post_dense = True
        stateful = True
        use_learnable_state = False
        use_trainable_weights_with_reslayers = False

        if return_params_arr != False:
            params = params_arr
        else:
            params = None

        # timeMeanofSpaceRMS = np.mean(np.mean(latent_states_all**2, axis=1)**0.5)

        # saving simulation data
        sim_data = {
            'params_mat':params_mat,
            'init_state_mat':init_state_mat,
            't0':t0,
            'T':T,
            'delta_t':delta_t,
            'return_params_arr':return_params_arr,
            'dir_name_ae':dir_name_ae,
            'normalize_dataset':normalize_dataset,
            'stddev_multiplier':stddev_multiplier,
            'use_ae_data':use_ae_data,
        }


        with open(dir_name_rnn+dir_sep+'sim_data_AE_params.txt', 'w') as f:
            f.write(str(sim_data))

        # saving RNN specific data
        RNN_specific_data = {
            'dt_rnn':dt_rnn,
            'T_sample_input':T_sample_input,
            'T_sample_output':T_sample_output,
            'T_offset':T_offset,
            'boundary_idx_arr':boundary_idx_arr,
            'delta_t':delta_t,
            'params':params,
            'return_params_arr':return_params_arr,
            'normalize_dataset':normalize_dataset,
            'num_lyaptimesteps_totrain':num_lyaptimesteps_totrain,
            'stddev_multiplier':stddev_multiplier,
            'skip_intermediate':skip_intermediate,
            'module':RNN_SingleStep.__module__,
            'noise_type':noise_type,
            'normalization_type':normalization_type,
            'dense_layer_act_func':dense_layer_act_func,
            'stateful':stateful,
            'use_learnable_state':use_learnable_state,
            'use_weights_post_dense':use_weights_post_dense,
            'use_trainable_weights_with_reslayers':use_trainable_weights_with_reslayers,
        }

        with open(dir_name_rnn+dir_sep+'RNN_specific_data.txt', 'w') as f:
            f.write(str(RNN_specific_data))

    rnn_res_dict = create_data_for_RNN(
        latent_states_all,
        dt_rnn,
        T_sample_input,
        T_sample_output,
        T_offset,
        None,
        boundary_idx_arr,
        delta_t,
        params=params,
        return_numsamples=True,
        normalize_dataset=normalize_dataset,
        stddev_multiplier=stddev_multiplier,
        skip_intermediate=skip_intermediate,
        return_OrgDataIdxArr=False,
        normalization_arr_external=normalization_arr,
        normalization_type=normalization_type)

    data_rnn_input = rnn_res_dict['data_rnn_input']
    data_rnn_output = rnn_res_dict['data_rnn_output']
    org_data_idx_arr_input = rnn_res_dict['org_data_idx_arr_input']
    org_data_idx_arr_output = rnn_res_dict['org_data_idx_arr_output']
    num_samples = rnn_res_dict['num_samples']
    normalization_arr = rnn_res_dict['normalization_arr']
    rnn_data_boundary_idx_arr = rnn_res_dict['rnn_data_boundary_idx_arr']

    temp = np.divide(latent_states_all-normalization_arr[0], normalization_arr[1])
    time_stddev = np.std(temp, axis=0)
    timeMeanofSpaceRMS = np.mean(np.mean(temp**2, axis=1)**0.5)
    del(org_data_idx_arr_input)
    del(org_data_idx_arr_output)
    del(latent_states_all)
    del(temp)

    rnn_res_dict = create_data_for_RNN(
        all_data,
        dt_rnn,
        T_sample_input,
        T_sample_output,
        T_offset,
        None,
        boundary_idx_arr,
        delta_t,
        params=params,
        return_numsamples=True,
        normalize_dataset=False,
        stddev_multiplier=stddev_multiplier,
        skip_intermediate=skip_intermediate,
        return_OrgDataIdxArr=False,
        normalization_arr_external=normalization_arr,
        normalization_type=normalization_type,
        FTYPE=FTYPE,
        ITYPE=ITYPE)

    AR_data_rnn_input = rnn_res_dict['data_rnn_input']
    AR_data_rnn_output = rnn_res_dict['data_rnn_output']
    AR_org_data_idx_arr_input = rnn_res_dict['org_data_idx_arr_input']
    AR_org_data_idx_arr_output = rnn_res_dict['org_data_idx_arr_output']
    AR_num_samples = rnn_res_dict['num_samples']
    AR_normalization_arr = rnn_res_dict['normalization_arr']
    AR_rnn_data_boundary_idx_arr = rnn_res_dict['rnn_data_boundary_idx_arr']

    del(all_data)
    del(AR_org_data_idx_arr_input)
    del(AR_org_data_idx_arr_output)
    del(AR_rnn_data_boundary_idx_arr)

    # setting up training params

    # ph computation parameters
    num_runs = 100
    T_sample_input_AR_ratio = 1
    T_sample_output_AR_ratio = 3

    if behaviour == 'initialiseAndTrainFromScratch':
        learning_rate_list = [1e-2, 1e-3, 1e-4, 1e-5]
        epochs = 150
        patience = 10 # parameter for early stopping
        min_delta = 1e-6  # parameter for early stopping
        lambda_reg = 1e-7  # weight for regularizer
        train_split = 0.8
        val_split = 0.1
        test_split = 1 - train_split - val_split
        batch_size = 32
        fRMS = 1.3e-2
        zoneout_rate = 0.0
        rnncell_dropout_rate = 0.0
        denselayer_dropout_rate = 0.0


        stddev = fRMS*np.mean(time_stddev[0:og_vars])

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
            'timeMeanofSpaceRMS':timeMeanofSpaceRMS,
            'stddev':stddev,
            'zoneout_rate':zoneout_rate,
            'rnncell_dropout_rate':rnncell_dropout_rate,
            'denselayer_dropout_rate':denselayer_dropout_rate,
        }

        with open(dir_name_rnn+dir_sep+'training_specific_params.txt', 'w') as f:
            f.write(str(training_specific_params))

        np.savez(
            dir_name_rnn+dir_sep+'normalization_data',
            normalization_arr=[normalization_arr],
        )

    else:
        # dir_name_rnn_og = dir_name_rnn
        # dir_name_rnn_temp = '/home/rkaushik/Documents/Thesis/MLROM/CDV/saved_rnn/rnn_'+dir_name_rnn_og[-3:]
        # dir_name_rnn = dir_name_rnn_temp

        with open(dir_name_rnn + dir_sep + 'training_specific_params.txt') as f:
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

        test_split = 1 - train_split - val_split

        # setting seed for PRNGs
        np.random.seed(prng_seed)
        tf.random.set_seed(prng_seed)


    cum_samples = rnn_data_boundary_idx_arr[-1]
    # idx = np.arange(cum_samples)
    # np.random.shuffle(idx)
    num_train_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_val_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_test_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_samples_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    begin_idx = 0
    for i in range(len(rnn_data_boundary_idx_arr)):
        num_samples = batch_size * int( np.round((rnn_data_boundary_idx_arr[i] - begin_idx)//batch_size) )
        num_train_arr[i] = batch_size * int( np.round(train_split*num_samples/batch_size) )
        num_val_arr[i] = batch_size * int( np.round(val_split*num_samples/batch_size) )
        num_test_arr[i] = batch_size * int( np.round((num_samples - num_train_arr[i] - num_val_arr[i])/batch_size) )
        num_samples_arr[i] = num_train_arr[i] + num_val_arr[i] + num_test_arr[i]
        begin_idx = rnn_data_boundary_idx_arr[i]

    # defining shapes
    training_input_shape = [np.sum(num_train_arr)]
    training_input_shape.extend(data_rnn_input.shape[1:])

    training_output_shape = [np.sum(num_train_arr)]
    training_output_shape.extend(data_rnn_output.shape[1:])

    val_input_shape = [np.sum(num_val_arr)]
    val_input_shape.extend(data_rnn_input.shape[1:])

    val_output_shape = [np.sum(num_val_arr)]
    val_output_shape.extend(data_rnn_output.shape[1:])

    testing_input_shape = [np.sum(num_test_arr)]
    testing_input_shape.extend(data_rnn_input.shape[1:])

    testing_output_shape = [np.sum(num_test_arr)]
    testing_output_shape.extend(data_rnn_output.shape[1:])

    # defining required arrays
    training_data_rnn_input = np.empty(shape=training_input_shape, dtype=FTYPE)
    training_data_rnn_output = np.empty(shape=training_output_shape, dtype=FTYPE)

    val_data_rnn_input = np.empty(shape=val_input_shape, dtype=FTYPE)
    val_data_rnn_output = np.empty(shape=val_output_shape, dtype=FTYPE)

    testing_data_rnn_input = np.empty(shape=testing_input_shape, dtype=FTYPE)
    testing_data_rnn_output = np.empty(shape=testing_output_shape, dtype=FTYPE)

    AR_ls_testing_data_rnn_input = np.empty(shape=testing_input_shape, dtype=FTYPE)
    AR_ls_testing_data_rnn_output = np.empty(shape=testing_output_shape, dtype=FTYPE)

    AR_testing_data_rnn_input = np.empty(shape=tuple(testing_input_shape[0:2])+tuple(AR_data_rnn_input.shape[2:]), dtype=FTYPE)
    AR_testing_data_rnn_output = np.empty(shape=tuple(testing_input_shape[0:2])+tuple(AR_data_rnn_input.shape[2:]), dtype=FTYPE)

    begin_idx = 0
    training_data_rolling_count = 0
    val_data_rolling_count = 0
    testing_data_rolling_count = 0
    for i in range(len(boundary_idx_arr)):
        idx = np.arange(begin_idx, rnn_data_boundary_idx_arr[i])
        # np.random.shuffle(idx)
        # num_samples = idx.shape[0]
        # num_train = int( np.round(train_split*num_samples/batch_size) )*batch_size
        # num_val = int( np.round(val_split*num_samples/batch_size) )*batch_size
        
        num_samples = num_samples_arr[i]
        num_train = num_train_arr[i]
        num_val = num_val_arr[i]
        num_test = num_test_arr[i]
        
        nbatches_train = num_train // batch_size
        nbatches_val = num_val // batch_size
        nbatches_test = num_test // batch_size

        for j in range(batch_size):
            training_data_rnn_input[training_data_rolling_count+j:training_data_rolling_count+num_train:batch_size] = data_rnn_input[idx[0:num_train]][j*nbatches_train:(j+1)*nbatches_train]
            training_data_rnn_output[training_data_rolling_count+j:training_data_rolling_count+num_train:batch_size] = data_rnn_output[idx[0:num_train]][j*nbatches_train:(j+1)*nbatches_train]
            
            val_data_rnn_input[val_data_rolling_count+j:val_data_rolling_count+num_val:batch_size] = data_rnn_input[idx[num_train:num_train+num_val]][j*nbatches_val:(j+1)*nbatches_val]
            val_data_rnn_output[val_data_rolling_count+j:val_data_rolling_count+num_val:batch_size] = data_rnn_output[idx[num_train:num_train+num_val]][j*nbatches_val:(j+1)*nbatches_val]

            testing_data_rnn_input[testing_data_rolling_count+j:testing_data_rolling_count+num_test:batch_size] = data_rnn_input[idx[num_train+num_val:num_samples]][j*nbatches_test:(j+1)*nbatches_test]
            testing_data_rnn_output[testing_data_rolling_count+j:testing_data_rolling_count+num_test:batch_size] = data_rnn_output[idx[num_train+num_val:num_samples]][j*nbatches_test:(j+1)*nbatches_test]
            

        AR_testing_data_rnn_input[testing_data_rolling_count:testing_data_rolling_count+num_test] = AR_data_rnn_input[idx[num_train+num_val:num_samples]]
        AR_testing_data_rnn_output[testing_data_rolling_count:testing_data_rolling_count+num_test] = AR_data_rnn_output[idx[num_train+num_val:num_samples]]

        AR_ls_testing_data_rnn_input[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_input[idx[num_train+num_val:num_samples]]
        AR_ls_testing_data_rnn_output[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_output[idx[num_train+num_val:num_samples]]

        # training_data_rnn_input[training_data_rolling_count:training_data_rolling_count+num_train] = data_rnn_input[idx[0:num_train]]
        # training_data_rnn_output[training_data_rolling_count:training_data_rolling_count+num_train] = data_rnn_output[idx[0:num_train]]
        training_data_rolling_count += num_train

        # val_data_rnn_input[val_data_rolling_count:val_data_rolling_count+num_val] = data_rnn_input[idx[num_train:num_train+num_val]]
        # val_data_rnn_output[val_data_rolling_count:val_data_rolling_count+num_val] = data_rnn_output[idx[num_train:num_train+num_val]]
        val_data_rolling_count += num_val

        # num_test = num_samples-num_train-num_val+1
        # testing_data_rnn_input[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_input[idx[num_train+num_val:]]
        # testing_data_rnn_output[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_output[idx[num_train+num_val:]]
        testing_data_rolling_count += num_test

        begin_idx = rnn_data_boundary_idx_arr[i]

    # cleaning up
    del(data_rnn_input)
    del(data_rnn_output)
    del(AR_data_rnn_input)
    del(AR_data_rnn_output)

    # further shuffling
    if stateful == False:
        idx = np.arange(0, training_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        training_data_rnn_input = training_data_rnn_input[idx]
        training_data_rnn_output = training_data_rnn_output[idx]

        idx = np.arange(0, val_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        val_data_rnn_input = val_data_rnn_input[idx]
        val_data_rnn_output = val_data_rnn_output[idx]

        idx = np.arange(0, testing_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        testing_data_rnn_input = testing_data_rnn_input[idx]
        testing_data_rnn_output = testing_data_rnn_output[idx]

        del(idx)

    s_in = AR_testing_data_rnn_input.shape
    AR_testing_data_rnn_input = AR_testing_data_rnn_input.reshape((1, s_in[0]*s_in[1]) + s_in[2:])

    s_out = AR_testing_data_rnn_output.shape
    AR_testing_data_rnn_output = AR_testing_data_rnn_output.reshape((1, s_out[0]*s_out[1]) + s_out[2:])

    T_sample_input_AR = T_sample_input_AR_ratio*np.mean(lyapunov_time_arr)#50.1*dt_rnn
    num_sample_input_AR = int((T_sample_input_AR+0.5*dt_rnn)//dt_rnn)

    T_sample_output_AR = T_sample_output_AR_ratio*np.mean(lyapunov_time_arr)
    num_sample_output_AR = int((T_sample_output_AR+0.5*dt_rnn)//dt_rnn)

    num_offset_AR = num_sample_input_AR
    T_offset_AR = num_offset_AR*dt_rnn

    batch_idx = np.random.randint(low=0, high=AR_testing_data_rnn_input.shape[0])
    maxpossible_num_runs = AR_testing_data_rnn_input.shape[1]-(num_sample_input_AR+num_sample_output_AR)

    num_runs = np.min([num_runs, maxpossible_num_runs])

    print('num_runs : ', num_runs)

    data_idx_arr = np.linspace(0, maxpossible_num_runs-1, num_runs, dtype=np.int32)

    AR_data_in = np.empty(shape=(num_runs, num_sample_input_AR)+tuple(s_in[2:]))
    AR_data_out = np.empty(shape=(num_runs, num_sample_output_AR)+tuple(s_out[2:]))

    for i in range(num_runs):
        d_idx = data_idx_arr[i]
        AR_data_in[i] = AR_testing_data_rnn_input[0, d_idx:d_idx+num_sample_input_AR]
        AR_data_out[i] = AR_testing_data_rnn_input[0, d_idx+num_sample_input_AR:d_idx+num_sample_input_AR+num_sample_output_AR]

    del(AR_testing_data_rnn_input)
    del(AR_testing_data_rnn_output)
    AR_testing_data_rnn_input = AR_data_in
    AR_testing_data_rnn_output = AR_data_out

    AR_testing_data_rnn_input = np.reshape(
        AR_testing_data_rnn_input,
        (
            AR_testing_data_rnn_input.shape[0],
            AR_testing_data_rnn_input.shape[1],
        ) + tuple(all_data_shape_og)
    )

    AR_testing_data_rnn_output = np.reshape(
        AR_testing_data_rnn_output,
        (
            AR_testing_data_rnn_output.shape[0],
            AR_testing_data_rnn_output.shape[1],
        ) + tuple(all_data_shape_og)
    )


    T_sample_input_ls_AR_ratio = 1
    T_sample_output_ls_AR_ratio = 3

    s_in = AR_ls_testing_data_rnn_input.shape
    AR_ls_testing_data_rnn_input = AR_ls_testing_data_rnn_input.reshape((1, s_in[0]*s_in[1]) + s_in[2:])

    s_out = AR_ls_testing_data_rnn_output.shape
    AR_ls_testing_data_rnn_output = AR_ls_testing_data_rnn_output.reshape((1, s_out[0]*s_out[1]) + s_out[2:])

    T_sample_input_ls_AR = T_sample_input_ls_AR_ratio*np.mean(lyapunov_time_arr)#50.1*dt_rnn
    num_sample_input_ls_AR = int((T_sample_input_ls_AR+0.5*dt_rnn)//dt_rnn)

    T_sample_output_ls_AR = T_sample_output_ls_AR_ratio*np.mean(lyapunov_time_arr)
    num_sample_output_ls_AR = int((T_sample_output_ls_AR+0.5*dt_rnn)//dt_rnn)

    num_offset_ls_AR = num_sample_input_ls_AR
    T_offset_ls_AR = num_offset_ls_AR*dt_rnn

    batch_idx = np.random.randint(low=0, high=AR_ls_testing_data_rnn_input.shape[0])
    maxpossible_num_runs = AR_ls_testing_data_rnn_input.shape[1]-(num_sample_input_ls_AR+num_sample_output_ls_AR)

    num_runs = np.min([num_runs, maxpossible_num_runs])

    print('num_runs : ', num_runs)

    data_idx_arr = np.linspace(0, maxpossible_num_runs-1, num_runs, dtype=np.int32)

    AR_ls_data_in = np.empty(shape=(num_runs, num_sample_input_ls_AR)+tuple(s_in[2:]))
    AR_ls_data_out = np.empty(shape=(num_runs, num_sample_output_ls_AR)+tuple(s_out[2:]))

    for i in range(num_runs):
        d_idx = data_idx_arr[i]
        AR_ls_data_in[i] = AR_ls_testing_data_rnn_input[0, d_idx:d_idx+num_sample_input_ls_AR]
        AR_ls_data_out[i] = AR_ls_testing_data_rnn_input[0, d_idx+num_sample_input_ls_AR:d_idx+num_sample_input_ls_AR+num_sample_output_ls_AR]

    del(AR_ls_testing_data_rnn_input)
    del(AR_ls_testing_data_rnn_output)
    AR_ls_testing_data_rnn_input = AR_ls_data_in
    AR_ls_testing_data_rnn_output = AR_ls_data_out


    s =  '      training_data_rnn_input.shape : {}\n'.format(training_data_rnn_input.shape)
    s += '     training_data_rnn_output.shape : {}\n'.format(training_data_rnn_output.shape)
    s += '       testing_data_rnn_input.shape : {}\n'.format(testing_data_rnn_input.shape)
    s += '      testing_data_rnn_output.shape : {}\n'.format(testing_data_rnn_output.shape)
    s += '           val_data_rnn_input.shape : {}\n'.format(val_data_rnn_input.shape)
    s += '          val_data_rnn_output.shape : {}\n\n'.format(val_data_rnn_output.shape)
    s += '    AR_testing_data_rnn_input.shape : {}\n'.format(AR_testing_data_rnn_input.shape)
    s += '   AR_testing_data_rnn_output.shape : {}\n\n'.format(AR_testing_data_rnn_output.shape)
    s += ' AR_ls_testing_data_rnn_input.shape : {}\n'.format(AR_ls_testing_data_rnn_input.shape)
    s += 'AR_ls_testing_data_rnn_output.shape : {}'.format(AR_ls_testing_data_rnn_output.shape)

    with open(dir_name_rnn + '/data_shape.txt', 'w') as f:
        f.write(s)

    print(s+'\n\n')

    # Initialize network
    if behaviour == 'initialiseAndTrainFromScratch':
        # rnn_layers_units = [500]*3
        # scalar_weights = None
        # scalar_weights = [
        #     0.5, 
        #     0.0, 0.5,
        #     0.0, 0.0, 1.0,
        #     1/6, 1/3, 1/3, 1/6
        # ] # RK4
        # scalar_weights = [
        #     1.0,
        #     0.25, 0.25,
        #     1/6, 1/6, 2/3
        # ] # TVD RK3
        # scalar_weights = [
        #     1.0,
        #     0.5, 0.5
        # ] # TVD RK2
        num_rnn_layers = 1
        if not isinstance(scalar_weights, type(None)):
            num_rnn_layers += int( ((8*len(scalar_weights)+1)**0.5 - 1)/2 )

        rnn_layers_units = [rnn_layers_units_eachlayer]*num_rnn_layers

        # timeMeanofSpaceRMS = np.mean(np.mean(latent_states_all**2, axis=1)**0.5)
        print('timeMeanofSpaceRMS :', timeMeanofSpaceRMS)
        print('stddev :', stddev)

        data_dim = num_latent_states

        dense_dim = [rnn_layers_units[-1]]*(len(dense_layer_act_func)-1)
        dense_dim.append(data_dim)
            
        if strategy is not None:
            with strategy.scope():
                rnn_net = RNN_SingleStep(
                    data_dim=data_dim,
                    # in_steps=int(T_sample_input // dt_rnn),
                    # out_steps=int(T_sample_output // dt_rnn),
                    dt_rnn=dt_rnn,
                    lambda_reg=lambda_reg,
                    reg_name='L2',
                    rnn_layers_units=rnn_layers_units,
                    dense_layer_act_func=dense_layer_act_func,
                    load_file=None,
                    # T_input=T_sample_input,
                    # T_output=T_sample_output,
                    stddev=stddev,
                    noise_type=noise_type,
                    dense_dim=dense_dim,
                    use_learnable_state=use_learnable_state,
                    stateful=stateful,
                    zoneout_rate=zoneout_rate,
                    batch_size=batch_size,
                    use_weights_post_dense=use_weights_post_dense,
                    rnncell_dropout_rate=rnncell_dropout_rate,
                    denselayer_dropout_rate=denselayer_dropout_rate,
                    scalar_weights=scalar_weights, # corresponding to RK4
                    use_trainable_weights_with_reslayers=use_trainable_weights_with_reslayers,
                )
        else:
            rnn_net = RNN_SingleStep(
                data_dim=data_dim,
                # in_steps=int(T_sample_input // dt_rnn),
                # out_steps=int(T_sample_output // dt_rnn),
                dt_rnn=dt_rnn,
                lambda_reg=lambda_reg,
                reg_name='L2',
                rnn_layers_units=rnn_layers_units,
                dense_layer_act_func=dense_layer_act_func,
                load_file=None,
                # T_input=T_sample_input,
                # T_output=T_sample_output,
                stddev=stddev,
                noise_type=noise_type,
                dense_dim=dense_dim,
                use_learnable_state=use_learnable_state,
                stateful=stateful,
                zoneout_rate=zoneout_rate,
                batch_size=batch_size,
                use_weights_post_dense=use_weights_post_dense,
                rnncell_dropout_rate=rnncell_dropout_rate,
                denselayer_dropout_rate=denselayer_dropout_rate,
                scalar_weights=scalar_weights, # corresponding to RK4
                use_trainable_weights_with_reslayers=use_trainable_weights_with_reslayers,
            )
        save_path = dir_name_rnn+dir_sep+'final_net'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rnn_net.save_class_dict(save_path+dir_sep+'final_net_class_dict.txt')
    else:
        load_file = dir_name_rnn + dir_sep + 'final_net' + dir_sep + 'final_net_class_dict.txt'
        if strategy is not None:
            with strategy.scope():
                rnn_net = RNN_SingleStep(
                    load_file=load_file,
                    # T_input=T_sample_input,
                    # T_output=T_sample_output,
                    batch_size=batch_size,
                    
                )
        else:
            rnn_net = RNN_SingleStep(
                load_file=load_file,
                # T_input=T_sample_input,
                # T_output=T_sample_output,
                batch_size=batch_size
            )

        rnn_net.build(input_shape=(batch_size, None, num_latent_states))
        
        if behaviour == 'loadCheckpointAndContinueTraining':
            wt_file = tf.train.latest_checkpoint(dir_name_rnn+dir_sep+'checkpoints')
        elif behaviour == 'loadFinalNetAndPlot':
            wt_file = dir_name_rnn+dir_sep+'final_net'+dir_sep+'final_net_gru_weights.h5'
            # wt_file = dir_name_rnn+dir_sep+'final_net'+dir_sep+'f2'#+dir_sep+'saved_model.pb'
            rnn_net.load_weights_from_file(wt_file)
        
        # this forces the model to initialize its kernel weights/biases
        # temp = rnn_net.predict(tf.ones(shape=[batch_size, int(T_sample_input//dt_rnn), rnn_net.data_dim]))
        # this loads just the kernel wieghts and biases of the model
        # rnn_net.load_weights_from_file(wt_file)

        # rnn_net = tf.keras.models.load_model(wt_file)

    if behaviour == 'initialiseAndTrainFromScratch':
        val_loss_hist = []
        train_loss_hist = []
        lr_change=[0, 0]
        savelosses_cb_vallossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
        savelosses_cb_trainlossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
        starting_lr_idx = 0
        num_epochs_left = epochs
        earlystopping_wait = 0
    elif behaviour == 'loadCheckpointAndContinueTraining':
        val_loss_hist, train_loss_hist, lr_change, starting_lr_idx, num_epochs_left, val_loss_arr_fromckpt, train_loss_arr_fromckpt, earlystopping_wait = readAndReturnLossHistories(
            dir_name_ae=dir_name_rnn,
            dir_sep=dir_sep,
            epochs=epochs,
            learning_rate_list=learning_rate_list,
            return_earlystopping_wait=True)
        savelosses_cb_vallossarr = val_loss_arr_fromckpt
        savelosses_cb_trainlossarr = train_loss_arr_fromckpt
    elif behaviour == 'loadFinalNetAndPlot':
        with open(dir_name_rnn+'{ds}final_net{ds}losses.txt'.format(ds=dir_sep), 'r') as f:
            lines = f.readlines()
        
        losses_dict = eval(''.join(lines))

        val_loss_hist = losses_dict['val_loss_hist']
        train_loss_hist = losses_dict['train_loss_hist']
        lr_change = losses_dict['lr_change']
        test_loss = losses_dict['test_loss']

    train_NMSE_hist = []
    val_NMSE_hist = []

    train_MSE_hist = []
    val_MSE_hist = []

    class NMSE(tf.keras.metrics.MeanSquaredError):
        def __init__(self, divisor_arr, name='NMSE', **kwargs):
            super(NMSE, self).__init__(name, **kwargs)
            self.divisor_arr = divisor_arr

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = y_true / self.divisor_arr
            y_pred = y_pred / self.divisor_arr
            return super(NMSE, self).update_state(y_true, y_pred, sample_weight)

    # compiling the network
    rnn_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),
        loss=losses.MeanSquaredError(),
        metrics=['mse', NMSE(divisor_arr=time_stddev)],
        run_eagerly=False
    )

    if behaviour == 'loadCheckpointAndContinueTraining':
        # this loads the weights/attributes of the optimizer as well
        if strategy is not None:
            with strategy.scope():
                rnn_net.load_weights(wt_file)
        else:
            rnn_net.load_weights(wt_file)

    if behaviour == 'initialiseAndTrainFromScratch' or behaviour == 'loadCheckpointAndContinueTraining':
        # implementing early stopping
        baseline = None
        if behaviour == 'loadCheckpointAndContinueTraining':
            baseline = np.min(val_loss_hist)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_NMSE',
            patience=patience,
            restore_best_weights=True,
            verbose=True,
            min_delta=min_delta,
            baseline=baseline
        )
        #** the two lines below are useless because wait is set to 0 in on_train_begin
        # early_stopping_cb.wait = earlystopping_wait
        # print('early_stopping_cb.wait : {}\n'.format(early_stopping_cb.wait))

        # time callback for each epoch
        timekeeper_cb = mytimecallback()

        # model checkpoint callback
        dir_name_ckpt = dir_name_rnn+dir_sep+'checkpoints'
        if not os.path.isdir(dir_name_ckpt):
            os.makedirs(dir_name_ckpt)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=dir_name_ckpt+dir_sep+'checkpoint',#+'/checkpoint--loss={loss:.4f}--vall_loss={val_loss:.4f}',
            monitor='val_NMSE',
            save_best_only=True,
            save_weights_only=True,
            verbose=2,
            initial_value_threshold=baseline,
            period=1  # saves every `period` epochs
        )

        # save losses callback
        savelosses_cb = SaveLosses(
            filepath=dir_name_ckpt+dir_sep+'LossHistoriesCheckpoint',
            val_loss_arr=savelosses_cb_vallossarr,
            train_loss_arr=savelosses_cb_trainlossarr,
            total_epochs=epochs,
            period=1)

        for i in range(starting_lr_idx, len(learning_rate_list)):
            learning_rate = learning_rate_list[i]
            K.set_value(rnn_net.optimizer.lr, learning_rate)

            savelosses_cb.update_lr_idx(i)

            if i == starting_lr_idx:
                EPOCHS = num_epochs_left
                savelosses_cb.update_offset(epochs-num_epochs_left)
            else:
                EPOCHS = epochs
                savelosses_cb.update_offset(0)

            total_s_len = 80
            sep_lr_s = ' LEARNING RATE : {} '.format(learning_rate)
            sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'-' + sep_lr_s
            sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'-'
            print('\n\n' + '-'*len(sep_lr_s))
            print('\n' + sep_lr_s+'\n')
            print('-'*len(sep_lr_s) + '\n\n')
            
            history = rnn_net.fit(training_data_rnn_input, training_data_rnn_output,
                epochs=EPOCHS,
                batch_size=batch_size,
                # validation_split=val_split/train_split,
                validation_data=(val_data_rnn_input, val_data_rnn_output),
                callbacks=[early_stopping_cb, timekeeper_cb, checkpoint_cb, savelosses_cb],
                verbose=1,
                shuffle=not stateful,
            )

            val_loss_hist.extend(history.history['val_loss'])
            train_loss_hist.extend(history.history['loss'])
            
            val_NMSE_hist.extend(history.history['val_NMSE'])
            train_NMSE_hist.extend(history.history['NMSE'])

            val_MSE_hist.extend(history.history['val_mse'])
            train_MSE_hist.extend(history.history['mse'])
            
            if i == starting_lr_idx:
                lr_change[i+1] += len(history.history['val_loss'])
            else:
                lr_change.append(lr_change[i]+len(history.history['val_loss']))


    if behaviour == 'initialiseAndTrainFromScratch' or behaviour == 'loadCheckpointAndContinueTraining':
        for layer in rnn_net.rnn_list:
            if layer.stateful == True:
                layer.reset_states()
        print(testing_data_rnn_input.shape, testing_data_rnn_output.shape)
        eval_dict = rnn_net.evaluate(
            testing_data_rnn_input, testing_data_rnn_output,
            batch_size=batch_size,
        )

        save_path = dir_name_rnn+dir_sep+'final_net'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)


        with open(save_path+dir_sep+'losses.txt', 'w') as f:
            f.write(str({
                'val_loss_hist':val_loss_hist,
                'train_loss_hist':train_loss_hist,
                'val_MSE_hist':val_MSE_hist,
                'train_MSE_hist':train_MSE_hist,
                'val_NMSE_hist':val_NMSE_hist,
                'train_NMSE_hist':train_NMSE_hist,
                'lr_change':lr_change,
                'test_loss':eval_dict[0],
                'test_MSE':eval_dict[1],
                'test_NMSE':eval_dict[2],
            }))
            
        if normalize_dataset == True:
            with open(save_path+dir_sep+'rnn_normalization.txt', 'w') as f:
                f.write(str({
                    'normalization_arr':normalization_arr
                }))

        rnn_net.save_everything(
            file_name=save_path+dir_sep+'final_net')

    xlabel_kwargs = {'fontsize':15}
    ylabel_kwargs = {'fontsize':15}
    legend_kwargs = {'fontsize':12}

    # plotting losses
    dir_name_plot = dir_name_rnn + '/plots'
    if not os.path.isdir(dir_name_plot):
        os.makedirs(dir_name_plot)

    # Visualize loss history
    fig, ax = plot_losses(
        training_loss=train_loss_hist,
        val_loss=val_loss_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
    )

    plt.savefig(dir_name_plot + '{ds}loss_history.pdf'.format(ds=dir_sep), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    fig, ax = plot_losses(
        training_loss=train_MSE_hist,
        val_loss=val_MSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training MSE', 'Validation MSE'],
        xlabel='Epoch',
        ylabel='MSE',
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
    )
    plt.savefig(dir_name_plot+'/MSE_history.pdf', dpi=300, bbox_inches='tight')
    # plt.clf()
    plt.close()

    fig, ax = plot_losses(
        training_loss=train_NMSE_hist,
        val_loss=val_NMSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training NMSE', 'Validation NMSE'],
        xlabel='Epoch',
        ylabel='NMSE',
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
    )
    plt.savefig(dir_name_plot+'/NMSE_history.pdf', dpi=300, bbox_inches='tight')
    # plt.clf()
    plt.close('all')

    ###--- Autoregressive Mode ---###

    error_threshold = 0.5
    num_runs = AR_testing_data_rnn_input.shape[0]

    T_sample_input_AR = AR_testing_data_rnn_input.shape[1]*dt_rnn + 0.01*dt_rnn
    T_sample_output_AR = AR_testing_data_rnn_output.shape[1]*dt_rnn + 0.01*dt_rnn
    
    AR_rnn_net = AR_RNN(
        load_file=save_path+'/final_net_class_dict.txt',
        T_input=T_sample_input_AR,
        T_output=T_sample_output_AR,
        stddev=0.0,
        batch_size=num_runs,
        lambda_reg=lambda_reg,
    )
    AR_rnn_net.build(input_shape=tuple(AR_testing_data_rnn_input.shape[0:2]) + tuple(testing_data_rnn_input.shape[2:]))
    AR_rnn_net.load_weights_from_file(save_path+'/final_net_gru_weights.h5')

    ae_data_normalization_arr = normalization_constant_arr_aedata

    AR_AERNN_net = AR_AERNN(
        ae_net,
        AR_rnn_net,
        normalization_arr,
        ae_data_normalization_arr,
        covmat_lmda=0.0,
        time_stddev_ogdata=time_stddev_ogdata,
        time_mean_ogdata=time_mean_ogdata,
        loss_weights=None,
        clipnorm=None,
        global_clipnorm=None
    )

    savefig_fname = 'pre_ARtraining-testingdata--combinedAERNN--ZEROoutsteps'
    npsavedata_fname = '/prediction_horizons-testingdata--combinedAERNN--ZEROoutsteps'
    plot_dir = '/plots'

    analysis_time = time.time()

    sidx1 = dir_name_rnn[::-1].index('/')
    sidx2 = dir_name_rnn[-sidx1-2::-1].index('/')
    print(dir_name_rnn[-(sidx1+sidx2+1):])
    print('num_runs :', num_runs)

    prediction_horizon_arr_og = np.empty(shape=num_runs)
    prediction_horizon_arr_new = np.empty(shape=num_runs)
    prediction_lst = np.array(AR_AERNN_net(AR_testing_data_rnn_input, training=False))
    prediction_lst = invert_normalization(prediction_lst, ae_data_normalization_arr)

    data_in_og = AR_testing_data_rnn_input
    data_out_og = AR_testing_data_rnn_output

    energySpectrum_dataout = 0.0
    energySpectrum_pred = 0.0

    avg_time = 0.
    for i in range(num_runs):
        run_time = time.time()
        lyap_time = lyapunov_time_arr[0]

        data_out = data_out_og[i]
        data_out = invert_normalization(data_out, ae_data_normalization_arr)

        ### Error and prediction horizon
        #-- og way --#
        error_og = (data_out - prediction_lst[i])**2
        error_og = np.divide(error_og, time_stddev_ogdata**2)
        error_og = np.reshape(error_og, (error_og.shape[0], -1))
        error_og = np.mean(error_og, axis=1)**0.5
        #-- new way --#
        error_new = (data_out - prediction_lst[i])**2
        error_new = np.reshape(error_new, (error_new.shape[0], -1))
        error_new = np.sum(error_new, axis=1) / np.sum(time_stddev_ogdata**2)
        error_new = error_new**0.5

        predhor_idx_og = np.where(error_og >= error_threshold)[0]
        if predhor_idx_og.shape[0] == 0:
            predhor_idx_og = error_og.shape[0]
        else:
            predhor_idx_og = predhor_idx_og[0]

        predhor_idx_new = np.where(error_new >= error_threshold)[0]
        if predhor_idx_new.shape[0] == 0:
            predhor_idx_new = error_new.shape[0]
        else:
            predhor_idx_new = predhor_idx_new[0]

        prediction_horizon_arr_og[i] = predhor_idx_og*dt_rnn/lyap_time
        prediction_horizon_arr_new[i] = predhor_idx_new*dt_rnn/lyap_time
        
        run_time = time.time() - run_time
        avg_time = (avg_time*i + run_time)/(i+1)
        eta = avg_time * (num_runs-1 - i)
        # print('    {} / {} -- run_time : {:.2f} s -- eta : {:.0f}h {:.0f}m {:.0f}s'.format(
        #     i+1,
        #     num_runs,
        #     run_time,
        #     float(eta // 3600),
        #     float((eta%3600)//60),
        #     float((eta%3600)%60),
        # ))

    median_idx = int(np.round(0.5*num_runs-1))
    quartile_1_idx = int(np.round(0.25*num_runs-1))
    quartile_3_idx = int(np.round(0.75*num_runs-1))

    prediction_horizon_arr_og.sort()
    prediction_horizon_arr_new.sort()

    median_og = prediction_horizon_arr_og[median_idx]
    quartile_1_og = prediction_horizon_arr_og[quartile_1_idx]
    quartile_3_og = prediction_horizon_arr_og[quartile_3_idx]
    IQR_og = quartile_3_og - quartile_1_og

    median_new = prediction_horizon_arr_new[median_idx]
    quartile_1_new = prediction_horizon_arr_new[quartile_1_idx]
    quartile_3_new = prediction_horizon_arr_new[quartile_3_idx]
    IQR_new = quartile_3_new - quartile_1_new

    prediction_horizon_og = np.mean(prediction_horizon_arr_og)
    stddev_ph_og = np.std(prediction_horizon_arr_og)

    prediction_horizon_new = np.mean(prediction_horizon_arr_new)
    stddev_ph_new = np.std(prediction_horizon_arr_new)

    s1 = 'ORIGINAL (multi-point) ERROR; error_threshold = {}\n'.format(error_threshold)
    s1 += 'prediction_horizon : {}, median : {}\n'.format(prediction_horizon_og, median_og)
    s1 += 'ph_min : {}, ph_max : {}\n'.format(prediction_horizon_arr_og.min(), prediction_horizon_arr_og.max())
    s1 += 'stddev : {}, IQR : {}\n'.format(stddev_ph_og, IQR_og)
    s1 += '1st quartile : {}, 3rd quartile : {}'.format(quartile_1_og, quartile_3_og)

    print('\n'+s1)

    s2 = 'NEW (cumulative) ERROR; error_threshold = {}\n'.format(error_threshold)
    s2 += 'prediction_horizon : {}, median : {}\n'.format(prediction_horizon_new, median_new)
    s2 += 'ph_min : {}, ph_max : {}\n'.format(prediction_horizon_arr_new.min(), prediction_horizon_arr_new.max())
    s2 += 'stddev : {}, IQR : {}\n'.format(stddev_ph_new, IQR_new)
    s2 += '1st quartile : {}, 3rd quartile : {}'.format(quartile_1_new, quartile_3_new)

    print('\n'+s2)

    plot_histogram_and_save(
        prediction_horizon_arr_og, median_og,
        save_dir=dir_name_rnn+plot_dir,
        savefig_fname=savefig_fname+'--OG_error',
    )

    plot_histogram_and_save(
        prediction_horizon_arr_new, median_new,
        save_dir=dir_name_rnn+plot_dir,
        savefig_fname=savefig_fname+'--NEW_error',
    )

    np.savez(
        dir_name_rnn+npsavedata_fname,
        prediction_horizon_arr_og=prediction_horizon_arr_og,
        prediction_horizon_arr_new=prediction_horizon_arr_new,
        error_threshold=error_threshold,
    )

    with open(dir_name_rnn+npsavedata_fname+'--statistics.txt', 'w') as fl:
        fl.write(s1+'\n\n')
        fl.write(s2)

    print('analysis time : {} s\n'.format(time.time() - analysis_time))
    
    median_actual_data_og = median_og
    prediction_horizon_actual_data_og = prediction_horizon_og
    median_actual_data_new = median_new
    prediction_horizon_actual_data_new = prediction_horizon_new

    ################################################################################
    # only latent space prediction horizon #

    print('\n-- latent space AR prediction --\n')

    T_sample_input_AR = AR_ls_testing_data_rnn_input.shape[1]*dt_rnn + 0.01*dt_rnn
    T_sample_output_AR = AR_ls_testing_data_rnn_output.shape[1]*dt_rnn + 0.01*dt_rnn

    num_runs = AR_ls_testing_data_rnn_input.shape[0]

    data_in = AR_ls_testing_data_rnn_input
    data_out = AR_ls_testing_data_rnn_output
    data_in_og = data_in
    data_out_og = data_out

    del(AR_rnn_net)
    AR_rnn_net = AR_RNN(
        load_file=save_path+'/final_net_class_dict.txt',
        T_input=T_sample_input_AR,
        T_output=T_sample_output_AR,
        stddev=0.0,
        batch_size=num_runs,
        lambda_reg=lambda_reg,
    )
    AR_rnn_net.build(input_shape=data_in_og.shape)
    AR_rnn_net.load_weights_from_file(save_path+'/final_net_gru_weights.h5')

    # for layer in AR_rnn_net.rnn_list:
    #     if layer.stateful:
    #         layer.reset_states()

    savefig_fname = 'pre_ARtraining-testingdata--latentspace'
    npsavedata_fname = '/prediction_horizons-testingdata--latentspace'
    plot_dir = '/plots'

    sidx1 = dir_name_rnn[::-1].index('/')
    sidx2 = dir_name_rnn[-sidx1-2::-1].index('/')
    print(dir_name_rnn[-(sidx1+sidx2+1):])
    print('num_runs :', num_runs)

    prediction_horizon_arr = np.empty(shape=num_runs)
    prediction = np.array(AR_rnn_net(data_in_og, training=False))
    # prediction = invert_normalization(prediction, normalization_arr)

    energySpectrum_dataout = 0.0
    energySpectrum_pred = 0.0

    avg_time = 0.
    for i in range(num_runs):
        run_time = time.time()
        lyap_time = lyapunov_time_arr[0]

        data_out = data_out_og[i]
        # data_out = invert_normalization(data_out, normalization_arr)

        ### Error and prediction horizon
        # error = np.linalg.norm(data_out[:, :] - prediction[i, :, :], axis=1)
        error = (data_out - prediction[i])**2
        # error /= norm_sq_time_average(data_out)**0.5
        error = np.divide(error, time_stddev**2)
        error = np.reshape(error, (error.shape[0], -1))
        error = np.mean(error, axis=1)**0.5

        predhor_idx = np.where(error >= error_threshold)[0]
        if predhor_idx.shape[0] == 0:
            predhor_idx = error.shape[0]
        else:
            predhor_idx = predhor_idx[0]

        prediction_horizon_arr[i] = predhor_idx*dt_rnn/lyap_time

        run_time = time.time() - run_time
        avg_time = (avg_time*i + run_time)/(i+1)
        eta = avg_time * (num_runs-1 - i)
        # print('    {} / {} -- run_time : {:.2f} s -- eta : {:.0f}h {:.0f}m {:.0f}s'.format(
        #     i+1,
        #     num_runs,
        #     run_time,
        #     float(eta // 3600),
        #     float((eta%3600)//60),
        #     float((eta%3600)%60),
        # ))

    median_idx = int(np.round(0.5*num_runs-1))
    quartile_1_idx = int(np.round(0.25*num_runs-1))
    quartile_3_idx = int(np.round(0.75*num_runs-1))

    prediction_horizon_arr.sort()

    median = prediction_horizon_arr[median_idx]
    quartile_1 = prediction_horizon_arr[quartile_1_idx]
    quartile_3 = prediction_horizon_arr[quartile_3_idx]
    IQR = quartile_3 - quartile_1

    prediction_horizon = np.mean(prediction_horizon_arr)
    stddev_ph = np.std(prediction_horizon_arr)

    s = 'error_threshold = {}\n'.format(error_threshold)
    s += 'prediction_horizon : {}, median : {}\n'.format(prediction_horizon, median)
    s += 'ph_min : {}, ph_max : {}\n'.format(prediction_horizon_arr.min(), prediction_horizon_arr.max())
    s += 'stddev : {}, IQR : {}\n'.format(stddev_ph, IQR)
    s += '1st quartile : {}, 3rd quartile : {}'.format(quartile_1, quartile_3)

    print('\n'+s)

    plot_histogram_and_save(
        prediction_horizon_arr, median,
        save_dir=dir_name_rnn+plot_dir,
        savefig_fname=savefig_fname,
    )
    plt.close('all')

    np.savez(
        dir_name_rnn+npsavedata_fname,
        prediction_horizon_arr=prediction_horizon_arr,
        error_threshold=error_threshold,
    )

    with open(dir_name_rnn+npsavedata_fname+'--statistics.txt', 'w') as fl:
        fl.write(s)

    print('analysis time : {} s\n'.format(time.time() - analysis_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rnn_layers_units_eachlayer', type=int)
    parser.add_argument('scalar_weights', nargs='+', type=str)
    parser.add_argument('-s', '--rnnss', type=str, nargs='+', default=['GRU_SingleStep_v1', 'RNN_GRU'])
    parser.add_argument('-r', '--arrnn', type=str, nargs='+', default=['GRU_AR_v1', 'AR_RNN_GRU'])
    parser.add_argument('-a', '--araernn', type=str, nargs='+', default=['AEGRU_AR_v1', 'AR_AERNN_GRU'])
    
    args = parser.parse_args()
    
    print('rnn_layers_units_eachlayer : {}, scalar_weights : {}'.format(args.rnn_layers_units_eachlayer, args.scalar_weights))
    print('rnnss : {}, araernn : {}, arrnn : {}'.format(args.rnnss, args.araernn, args.arrnn))
    
    if len(args.scalar_weights) == 1 and args.scalar_weights[0] == '0':
        scalar_weights = None
    else:
        scalar_weights = [eval('np.float64('+elem+')') for elem in args.scalar_weights]
    print('scalar_weights : {}'.format(scalar_weights))

    main(args.rnn_layers_units_eachlayer, scalar_weights, args.rnnss, args.araernn, args.arrnn)
