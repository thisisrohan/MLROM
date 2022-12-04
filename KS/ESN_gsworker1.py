import os
import sys
import math
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import time as time

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2
import h5py

from tools.misc_tools import create_data_for_RNN, plot_losses
from tools.ae_v5 import Autoencoder
from tools.ESN_v3 import ESN

tf.keras.backend.set_floatx('float32')

FTYPE = np.float32
ITYPE = np.int32

array = np.array
float32 = np.float32
int32 = np.int32
float64 = np.float64
int64 = np.int64

dir_sep = '/'

prng_seed = 42
np.random.seed(prng_seed)
tf.random.set_seed(prng_seed)

worker_id = 1
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


#---------- making rnn_all save directory ----------#
gs_idx = '003'
dir_name_rnn_all = os.getcwd() + '/grid_search/gridsearch_' + gs_idx + '/worker_{}'.format(worker_id)
if not os.path.isdir(dir_name_rnn_all):
    os.makedirs(dir_name_rnn_all)

# whether to use AE data or just work on raw data
use_ae_data = False # if false, specifying ae_idx will only show which dataset to use

# autoencoder directory
ae_idx = '046'
dir_name_ae = os.getcwd()+'/saved_ae/ae_'+ae_idx

# reading simulation parameters
with open(dir_name_ae + '/ae_data.txt') as f:
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

if os.path.exists(dir_name_ae+'/normalization_data.npz'):
    with np.load(dir_name_ae+'/normalization_data.npz', allow_pickle=True) as fl:
        normalization_constant_arr_aedata = fl['normalization_constant_arr_aedata'][0]

#---------- loading data ----------#
dir_name_data = os.getcwd() + '/saved_data/data_' + data_dir_idx
    
with open(dir_name_data + '/sim_data_params.txt') as f:
    lines = f.readlines()
params_dict = eval(''.join(lines))
params_mat = params_dict['params_mat']
# init_state = params_dict['init_state']
t0 = params_dict['t0']
T = params_dict['T']
delta_t = params_dict['delta_t']
numpoints_xgrid = params_dict['numpoints_xgrid']
length = params_dict['length']
return_params_arr = params_dict['return_params_arr']
normalize_flag_ogdata = params_dict['normalize_flag']
alldata_withparams_flag = params_dict['alldata_withparams_flag']

xgrid = length*np.linspace(0, 1, numpoints_xgrid)

with np.load(dir_name_data+'/data.npz', allow_pickle=True) as fl:
    all_data = fl['all_data'].astype(FTYPE)
    boundary_idx_arr = fl['boundary_idx_arr']
    normalization_constant_arr_ogdata = fl['normalization_constant_arr'][0]
    initial_t0 = fl['initial_t0']
    init_state_mat = fl['init_state_mat']

    lyapunov_spectrum_mat = fl['lyapunov_spectrum_mat'].astype(FTYPE)
lyapunov_time_arr = np.empty(shape=lyapunov_spectrum_mat.shape[0], dtype=FTYPE)
for i in range(lyapunov_spectrum_mat.shape[0]):
    lyapunov_time_arr[i] = 1/lyapunov_spectrum_mat[i, 0]


#---------- dealing with normalizing the data before feeding into autoencoder ----------#
if use_ae_data == True:
    if normalizeforae_flag == True:
        for i in range(numpoints_xgrid):
            all_data[:, i] -= normalization_constant_arr_aedata[0, i]
            all_data[:, i] /= normalization_constant_arr_aedata[1, i]

    if ae_data_with_params == False:
        all_data = all_data[:, 0:numpoints_xgrid]
else:
    # using raw data, neglecting the params attached (if any)
    all_data = all_data[:, 0:numpoints_xgrid]


#---------- loading autoencoder and creating latent states ----------#
if use_ae_data == True:
    load_file = dir_name_ae+'/final_net/final_net_class_dict.txt'
    wt_file = dir_name_ae+'/final_net/final_net_ae_weights.h5'
    ae_net = Autoencoder(all_data.shape[1], load_file=load_file)
    ae_net.load_weights_from_file(wt_file)

# create data
if use_ae_data == True:
    latent_states_all = ae_net.encoder_net.predict(all_data)
    del(all_data)
else:
    latent_states_all = all_data
num_latent_states = latent_states_all.shape[1]


#---------- reading grid search params for this worker_id ----------#
with np.load(os.getcwd()+'/grid_search/gridsearch_'+gs_idx+'/gsp_worker_{}.npz'.format(worker_id), allow_pickle=True) as fl:
    gsp = fl['gsp']


#---------- Training ESNs ----------#

### global ESN params
dt_rnn = 0.2
num_input_tsteps = 10000
T_sample_input = num_input_tsteps*dt_rnn
T_sample_output = T_sample_input
T_offset = dt_rnn
normalize_dataset = True # whether the data for the RNN should be normalized by the dataset's mean and std
normalization_arr = None
stddev_multiplier = 3
skip_intermediate = 'full sample'
noise_type = 'normal' # can be 'uniform' or 'normal'
normalization_type = 'stddev'

epochs = 10
lambda_reg = 1e-4 # weight for regularizer
min_delta = 1e-6
patience = 5
train_split = 0.8
val_split = 0.1
test_split = 1 - train_split - val_split
batch_size = 1
fRMS = 0.25/100
use_best = False

if return_params_arr != False:
    params = params_arr
else:
    params = None

### creating timeseries data for RNN
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

stddev = fRMS*timeMeanofSpaceRMS

### creating training/testing data
cum_samples = rnn_data_boundary_idx_arr[-1]
num_train = 0
num_val = 0
begin_idx = 0
for i in range(len(boundary_idx_arr)):
    num_samples = rnn_data_boundary_idx_arr[i] - begin_idx
    num_train += int( (1-test_split-val_split)*num_samples )
    num_val += int(val_split*num_samples)
    begin_idx = rnn_data_boundary_idx_arr[i]

# defining shapes
training_input_shape = [num_train]
training_input_shape.extend(data_rnn_input.shape[1:])

training_output_shape = [num_train]
training_output_shape.extend(data_rnn_output.shape[1:])

val_input_shape = [num_val]
val_input_shape.extend(data_rnn_input.shape[1:])

val_output_shape = [num_train]
val_output_shape.extend(data_rnn_output.shape[1:])

testing_input_shape = [cum_samples-num_train-num_val]
testing_input_shape.extend(data_rnn_input.shape[1:])

testing_output_shape = [cum_samples-num_train-num_val]
testing_output_shape.extend(data_rnn_output.shape[1:])

# defining required arrays
training_data_rnn_input = np.empty(shape=training_input_shape)
training_data_rnn_output = np.empty(shape=training_output_shape)

val_data_rnn_input = np.empty(shape=val_input_shape)
val_data_rnn_output = np.empty(shape=val_output_shape)

testing_data_rnn_input = np.empty(shape=testing_input_shape)
testing_data_rnn_output = np.empty(shape=testing_output_shape)

begin_idx = 0
training_data_rolling_count = 0
val_data_rolling_count = 0
testing_data_rolling_count = 0
for i in range(len(boundary_idx_arr)):
    idx = np.arange(begin_idx, rnn_data_boundary_idx_arr[i])
    num_samples = idx.shape[0]
    num_train = int( (1-test_split-val_split)*num_samples )
    num_val = int(val_split*num_samples)

    training_data_rnn_input[training_data_rolling_count:training_data_rolling_count+num_train] = data_rnn_input[idx[0:num_train]]
    training_data_rnn_output[training_data_rolling_count:training_data_rolling_count+num_train] = data_rnn_output[idx[0:num_train]]
    training_data_rolling_count += num_train

    val_data_rnn_input[val_data_rolling_count:val_data_rolling_count+num_val] = data_rnn_input[idx[num_train:num_train+num_val]]
    val_data_rnn_output[val_data_rolling_count:val_data_rolling_count+num_val] = data_rnn_output[idx[num_train:num_train+num_val]]
    val_data_rolling_count += num_val

    num_test = num_samples-num_train-num_val+1
    testing_data_rnn_input[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_input[idx[num_train+num_val:]]
    testing_data_rnn_output[testing_data_rolling_count:testing_data_rolling_count+num_test] = data_rnn_output[idx[num_train+num_val:]]
    testing_data_rolling_count += num_test

    begin_idx = rnn_data_boundary_idx_arr[i]

# cleaning up
del(data_rnn_input)
del(data_rnn_output)


### training individual ESNs
for ESNidx in range(gsp.shape[0]):

    totalESNcomputetime = time.time()

    total_s_len = 80
    sep_lr_s = ' ESN : {:03d}/{:03d} '.format(ESNidx+1, gsp.shape[0])
    sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'-' + sep_lr_s
    sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'-'
    print('\n\n' + '-'*len(sep_lr_s))
    print(sep_lr_s)
    print('-'*len(sep_lr_s) + '\n\n')

    gs_ESNparams = gsp[ESNidx]
    
    counter = 0
    while True:
        dir_check = '/ESN_' + str(counter).zfill(3)
        if os.path.isdir(dir_name_rnn_all + dir_check):
            counter += 1
        else:
            break

    dir_name_rnn = dir_name_rnn_all + dir_check
    os.makedirs(dir_name_rnn)
    
    ESN_layers_units = [ITYPE(gs_ESNparams[0])]
    omega_in = [FTYPE(gs_ESNparams[1])]
    rho_res = [FTYPE(gs_ESNparams[2])]
    alpha = [FTYPE(gs_ESNparams[3])]
    degree_of_connectivity = [ITYPE(gs_ESNparams[4])]
    stateful = True
    usebias_Win = [False]
    ESN_cell_activations = ['tanh']
    usebias_Wout = True
    
    # computing sparsity
    sparsity = [1-degree_of_connectivity[i]/(ESN_layers_units[i]-1) for i in range(len(ESN_layers_units))]
        

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
        'num_input_tsteps':num_input_tsteps,
        'stddev_multiplier':stddev_multiplier,
        'skip_intermediate':skip_intermediate,
        'module':ESN.__module__,
        'noise_type':noise_type,
        'normalization_type':normalization_type,
        'use_ae_data':use_ae_data,
    }

    with open(dir_name_rnn+dir_sep+'RNN_specific_data.txt', 'w') as f:
        f.write(str(RNN_specific_data))

    # saving training params
    training_specific_params = {
        'epochs':epochs,
        'prng_seed':prng_seed,
        'train_split':train_split,
        'val_split':val_split,
        'batch_size':batch_size,
        'fRMS':fRMS,
        'timeMeanofSpaceRMS':timeMeanofSpaceRMS,
        'stddev':stddev,
        'lambda_reg':lambda_reg,
        'min_delta':min_delta,
        'patience':patience,
        'use_best':use_best,
    }

    with open(dir_name_rnn+dir_sep+'training_specific_params.txt', 'w') as f:
        f.write(str(training_specific_params))
    
    np.savez(
        dir_name_rnn+dir_sep+'normalization_data',
        normalization_arr=[normalization_arr],
    )
    
    if return_params_arr != False:
        data_dim = num_latent_states + 3
    else:
        data_dim = num_latent_states


    rnn_net = ESN(
        data_dim=data_dim,
        dt_rnn=dt_rnn,
        lambda_reg=lambda_reg,
        ESN_layers_units=ESN_layers_units,
        stddev=stddev,
        noise_type=noise_type,
        stateful=stateful,
        omega_in=omega_in,
        sparsity=sparsity,
        rho_res=rho_res,
        usebias_Win=usebias_Win,
        alpha=alpha,
        ESN_cell_activations=ESN_cell_activations,
        prng_seed=prng_seed,
        usebias_Wout=usebias_Wout,
    )
    save_path = dir_name_rnn+dir_sep+'final_net'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    rnn_net.save_class_dict(save_path+dir_sep+'final_net_class_dict.txt')
    
    rnn_net.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=losses.MeanSquaredError(),
        metrics=['mse'],
        run_eagerly=False
    )
    
    Wout_best = 0
    val_mse_best = np.inf
    Wout_candidate = 0
    wait = 0
    
    val_loss_hist = []
    for i in range(epochs):
        epoch_totaltime = time.time()
    
        total_s_len = 30
        sep_lr_s = ' EPOCH : {}/{} '.format(i+1, epochs)
        sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*' ' + sep_lr_s
        sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*' '
        print('\n    ' + sep_lr_s)
        print('    '+'-'*len(sep_lr_s) + '\n')

        Hb = 0
        Yb = 0
        epoch_time = time.time()
        for j in range(training_data_rnn_input.shape[0]):
            batch_time = time.time()
            h = np.array(rnn_net(training_data_rnn_input[j:j+1], training=True))
#             h = rnn_net(training_data_rnn_input[j:j+1], training=True)
            # print(h.shape)
            h = h[0]
            # y = tf.constant(training_data_rnn_output[j])
            y = training_data_rnn_output[j]
            if usebias_Wout == True:
                h = np.concatenate((h, np.ones(shape=(h.shape[0], 1))), axis=1)
            Hb = Hb + np.matmul(np.transpose(h), h)
            Yb = Yb + np.matmul(np.transpose(y), h)
#             Hb = Hb + tf.linalg.matmul(tf.transpose(h), h)
#             Yb = Yb + tf.linalg.matmul(tf.transpose(y), h)

            print('    {} / {} -- batch_time : {} sec'.format(
                j+1,
                training_data_rnn_input.shape[0],
                time.time() - batch_time
            ))
        print('\n    epoch train time : {} sec'.format(time.time() - epoch_time))

        Wout = np.matmul(
            Yb,
            np.linalg.inv(Hb + lambda_reg*np.eye(Hb.shape[0]))
        )
        Wout = np.transpose(Wout)
#         Wout = tf.linalg.matmul(
#             Yb,
#             tf.linalg.inv(Hb + lambda_reg*tf.eye(Hb.shape[0]))
#         )
#         Wout = tf.transpose(Wout)
        
        Wout_candidate = Wout_candidate*i/(i+1) + Wout*1/(i+1)
        # tf.keras.backend.set_value(rnn_net.Wout, Wout_candidate)
        tf.keras.backend.set_value(rnn_net.Wout.kernel, Wout_candidate[0:ESN_layers_units[-1], :])
        if usebias_Wout == True:
            tf.keras.backend.set_value(rnn_net.Wout.bias, Wout_candidate[-1, :])

        for layer in rnn_net.ESN_layers:
            layer.reset_states()

        print('\n    computing val mse')
        val_mse = 0
        for j in range(val_data_rnn_input.shape[0]):
            batch_time = time.time()
            val_pred = np.array(rnn_net(val_data_rnn_input[j:j+1], training=False))
            temp = (val_pred - val_data_rnn_output[j:j+1])**2
            temp = np.mean(temp, axis=-1) # do a sqrt here to get rmse
            temp = np.mean(temp, axis=-1)
            temp = np.mean(temp, axis=-1)
            val_mse = val_mse*j/(j+1) + temp*1/(j+1)
            print('    {} / {} -- batch_time : {} sec'.format(
                j+1,
                val_data_rnn_input.shape[0],
                time.time() - batch_time
            ))

        for layer in rnn_net.ESN_layers:
            layer.reset_states()

        val_loss_hist.append(val_mse)

        print('    val_mse : {}'.format(val_mse))
        if val_mse + min_delta <= val_mse_best:
                print('    val_mse improved from {}'.format(val_mse_best))
                Wout_best = Wout_candidate
                val_mse_best = val_mse
                wait = 0
        else:
            wait += 1
            print('    val_mse did not improve from {}, wait : {}'.format(val_mse_best, wait))

        print('\n    Total epoch computation time : {} sec'.format(time.time()-epoch_totaltime))

        if wait >= patience:
            print('\n    ****early stopping****')
            break
        
#         val_loss_hist.extend(history.history['val_loss'])
#         train_loss_hist.extend(history.history['loss'])
        
#         if i == starting_lr_idx:
#             lr_change[i+1] += len(history.history['val_loss'])
#         else:
#             lr_change.append(lr_change[i]+len(history.history['val_loss']))

    print('    ------------------')
    print('    training completed')
    print('    ------------------')
    if use_best == True:
        # tf.keras.backend.set_value(rnn_net.Wout, Wout_best)
        tf.keras.backend.set_value(rnn_net.Wout.kernel, Wout_best[0:ESN_layers_units[-1], :])
        if usebias_Wout == True:
            tf.keras.backend.set_value(rnn_net.Wout.bias, Wout_best[-1, :])
    print('\n    computing test mse')
    test_mse = 0
    for j in range(testing_data_rnn_input.shape[0]):
        batch_time = time.time()
        test_pred = np.array(rnn_net(testing_data_rnn_input[j:j+1], training=False))
        temp = (test_pred - testing_data_rnn_output[j:j+1])**2
        temp = np.mean(temp, axis=-1) # do a sqrt here to get rmse
        temp = np.mean(temp, axis=-1)
        temp = np.mean(temp, axis=-1)
        test_mse = test_mse*j/(j+1) + temp*1/(j+1)
        print('    {} / {} -- batch_time : {} sec'.format(
                j+1,
                testing_data_rnn_input.shape[0],
                time.time() - batch_time
            ))
    print('    test_mse : {}'.format(test_mse))

    for layer in rnn_net.ESN_layers:
        layer.reset_states()

    ### saving the test loss and the ESN
    save_path = dir_name_rnn+dir_sep+'final_net'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    with open(save_path+dir_sep+'losses.txt', 'w') as f:
        f.write(str({
            'val_loss_hist':val_loss_hist,
            'test_loss':test_mse
        }))
        
    if normalize_dataset == True:
        with open(save_path+dir_sep+'rnn_normalization.txt', 'w') as f:
            f.write(str({
                'normalization_arr':normalization_arr
            }))

    rnn_net.save_everything(
        file_name=save_path+dir_sep+'final_net')

    print('    ESN compute-analyse-save time total : {} sec'.format(time.time() - totalESNcomputetime))
    
