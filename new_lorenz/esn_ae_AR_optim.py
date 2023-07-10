#!/usr/bin/env python
# coding: utf-8

from numpy import *
def main(esn_dir_idx, num_lyaptimesteps_totrain, gpu_to_use):
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

    tf.keras.backend.set_floatx('float32')

    plt.rcParams.update({
        "text.usetex":True,
        "font.family":"serif"
    })


    # In[3]:


    colab_flag = False
    FTYPE = np.float32
    ITYPE = np.int32

    array = np.array
    float32 = np.float32
    int32 = np.int32
    float64 = np.float64
    int64 = np.int64

    strategy = None
    # strategy = tf.distribute.MirroredStrategy()


    # In[4]:


    current_sys = platform.system()

    if current_sys == 'Windows':
        dir_sep = '\\'
    else:
        dir_sep = '/'


    # In[5]:


    if colab_flag == True:
        from google.colab import drive
        drive.mount('/content/drive')
        os.chdir('/content/drive/MyDrive/Github/MLROM/KS/')


    # In[6]:


    print(os.getcwd())


    # In[7]:


    from tools.misc_tools import create_data_for_RNN, mytimecallback, SaveLosses, plot_losses, plot_reconstructed_data_KS, plot_latent_states_KS , readAndReturnLossHistories, sigmoidWarmupAndDecayLRSchedule
    from tools.ae_v2 import Autoencoder
    from tools.ESN_v2_ensembleAR import ESN_ensemble as AR_RNN
    from tools.AEESN_AR_v1 import AR_AERNN_ESN as AR_AERNN
    from tools.trainAEESN_ensemble import trainAERNN


    # In[8]:


    behaviour = 'initialiseAndTrainFromScratch'
    # behaviour = 'loadCheckpointAndContinueTraining'
    # behaviour = 'loadFinalNetAndPlot'


    # In[ ]:





    # In[9]:


    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)

    if colab_flag == False:
        if strategy is None:
            if gpus:
                # gpu_to_use = 1
                tf.config.set_visible_devices(gpus[gpu_to_use], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(logical_devices)


    # In[10]:


    # print(tf.test.gpu_device_name())
    print(tf.config.list_physical_devices())
    print('')
    print(tf.config.list_logical_devices())
    print('')
    print(tf.__version__)


    # # KS System

    # In[11]:


    # setting up params (and saving, if applicable)

    if behaviour == 'initialiseAndTrainFromScratch':
        # RNN directory
        esn_dir_idx = '{:3d}'.format(esn_dir_idx)
        esn_dir_idx = esn_dir_idx.replace(' ', '0')
        dir_name_rnn = os.getcwd()+'/saved_ESN_ensemble/ESN_ensemble_' + esn_dir_idx

        # making AR-RNN save directory
        dir_name_ARrnn = os.getcwd() + dir_sep + 'saved_AR_AEESN_rnn'
        if not os.path.isdir(dir_name_ARrnn):
            os.makedirs(dir_name_ARrnn)

        counter = 0
        while True:
            dir_check = 'AR_ESN_ensemble_' + str(counter).zfill(3)
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
        return_params_arr = params_rnn_dict['return_params_arr']
        params = params_rnn_dict['params']
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
        dir_name_ARrnn = os.getcwd()+'/saved_AR_AERNN_rnn/AR_rnn_014'

        # reading AR-RNN parameters
        with open(dir_name_ARrnn + '/AR_RNN_specific_data.txt') as f:
            lines = f.readlines()
        
        params_AR_rnn_dict = eval(''.join(lines))

        dir_name_rnn = params_AR_rnn_dict['dir_name_rnn']
        rnn_idx = dir_name_rnn[-3:]
        dir_name_rnn = os.getcwd()+'/saved_ESN/ESN_'+rnn_idx

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
            print("'stddev_multiplier' not present in AR_RNN_specific_data, set to None.")
            stddev_multiplier = None
        try:
            skip_intermediate = params_AR_rnn_dict['skip_intermediate']
        except:
            print("'skip_intermediate' not present in AR_RNN_specific_data, set to 1.")
            skip_intermediate = 1
        try:
            use_ae_data = params_AR_rnn_dict['use_ae_data']
        except:
            print("'use_ae_data' not present in AR_RNN_specific_data, set to True.")
            use_ae_data = True
        try:
            normalization_type = params_AR_rnn_dict['normalization_type']
        except:
            print("'normalization_type' not present in AR_RNN_specific_data, set to 'stddev'.")
            normalization_type = 'stddev'

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
        covmat_lmda = tparams_dict['covmat_lmda']
        try:
            lambda_reg = tparams_dict['lambda_reg']
        except:
            lambda_reg = 1e-6
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
    with open(dir_name_rnn + '/final_net/0_final_net_class_dict.txt') as f:
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
    if os.path.exists(dir_name_ae+dir_sep+'normalization_data.npz'):
        with np.load(dir_name_ae+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
            normalization_constant_arr_aedata = fl['normalization_constant_arr_aedata'][0]
    try:
        ae_data_with_params = params_dict['ae_data_with_params']
    except:
        print("'ae_data_with_params' not present in ae_data, set to True.")
        ae_data_with_params = True

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


    test_split = 1 - train_split - val_split

    # setting seed for PRNGs
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)


    # In[ ]:





    # In[12]:


    lyapunov_time_arr = np.empty(shape=lyapunov_spectrum_mat.shape[0], dtype=FTYPE)
    for i in range(lyapunov_spectrum_mat.shape[0]):
        lyapunov_time_arr[i] = 1/lyapunov_spectrum_mat[i, 0]
        print('Case : {}, lyapunov exponent : {}, lyapunov time : {}s'.format(i+1, lyapunov_spectrum_mat[i, 0], lyapunov_time_arr[i]))


    # In[ ]:





    # In[ ]:





    # In[13]:


    # delaing with normalizing the data before feeding into autoencoder
    num_params = params_mat.shape[1]
    og_vars = all_data.shape[1]
    if alldata_withparams_flag == True:
        og_vars -= num_params

    # if use_ae_data == True:
    #     if ae_data_with_params == True and alldata_withparams_flag == False:
    #         new_all_data = np.empty(shape=(all_data.shape[0], og_vars+num_params), dtype=FTYPE)
    #         new_all_data[:, 0:og_vars] = all_data[:, 0:og_vars]
    #         del(all_data)
    #         all_data = new_all_data
    #         prev_idx = 0
    #         for i in range(boundary_idx_arr.shape[0]):
    #             all_data[prev_idx:boundary_idx_arr[i], num_params:] = params_mat[i]
    #             prev_idx = boundary_idx_arr[i]

    #     if normalizeforae_flag == True:
    #         for i in range(all_data.shape[1]):
    #             all_data[:, i] -= normalization_constant_arr_aedata[0, i]
    #             all_data[:, i] /= normalization_constant_arr_aedata[1, i]

    #     if ae_data_with_params == False:
    #         all_data = all_data[:, 0:og_vars]
    # else:
    #     # using raw data, neglecting the params attached (if any)
    #     all_data = all_data[:, 0:og_vars]

    if use_ae_data == True and ae_data_with_params == False:
        all_data = all_data[:, 0:og_vars]
    else:
        all_data = all_data[:, 0:og_vars]
        
    normalization_constant_arr_aedata = normalization_constant_arr_aedata[:, 0:all_data.shape[1]]


    # In[14]:


    # In[15]:


    print('all_data.shape : ', all_data.shape)
    print('all_data.dtype : ', all_data.dtype)


    # # Autoencoder

    # In[16]:


    # if use_ae_data == True:
    #     load_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_class_dict.txt'
    #     wt_file = dir_name_ae+dir_sep+'final_net'+dir_sep+'final_net_ae_weights.h5'


    # In[17]:


    # if use_ae_data == True:
    #     ae_net = Autoencoder(all_data.shape[1], load_file=load_file)
    #     ae_net.load_weights_from_file(wt_file)


    # In[ ]:





    # # ESN

    # In[18]:


    if behaviour == 'initialiseAndTrainFromScratch':
        # RNN data parameters
        num_lyaptimesteps_totrain = np.array(num_lyaptimesteps_totrain)*dt_rnn/np.mean(lyapunov_time_arr)
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
            'params':params,
            'return_params_arr':return_params_arr,
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


    # In[ ]:





    # In[19]:


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
            [200]*len(learning_rate_list[0]),
            [200]*len(learning_rate_list[1]),
            [200]*len(learning_rate_list[2]),
            [200]*len(learning_rate_list[3]),
            # [1000],
        ]
        patience = [
            [10]*len(learning_rate_list[0]),
            [10]*len(learning_rate_list[1]),
            [10]*len(learning_rate_list[2]),
            [10]*len(learning_rate_list[3]),
            # [50],
        ] # parameter for early stopping
        min_delta = 1e-6  # parameter for early stopping
        lambda_reg = 1e-9  # weight for regularizer
        covmat_lmda = 1e-4  # weight for the covmat loss

        if loss_weights is None:
            loss_weights = 1.0
            
        freeze_layers = [
            [],
            [],
            [],
        ]
        
        clipnorm = None #1.0
        batch_size = 32
        
        train_alpha = [False]*len(learning_rate_list)
        train_omega_in = [False]*len(learning_rate_list)
        train_rho_res = [False]*len(learning_rate_list)
        
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
            'lambda_reg':lambda_reg,
        }

        with open(dir_name_ARrnn+dir_sep+'training_specific_params.txt', 'w') as f:
            f.write(str(training_specific_params))
        
        np.savez(
            dir_name_ARrnn+dir_sep+'normalization_data',
            normalization_arr=[normalization_arr_rnn],
        )


    # In[20]:


    rnn_kwargs = {}
    if behaviour == 'initialiseAndTrainFromScratch' or behaviour == 'loadCheckpointAndContinueTraining':
        load_file_rnn = dir_name_rnn + '/final_net/final_net_class_dict.txt'
        wt_file_rnn = dir_name_rnn+'/final_net/final_net_ESN_weights.hdf5'
        
        load_file_ae = dir_name_ae+'/final_net/final_net_class_dict.txt'
        wt_file_ae = dir_name_ae+'/final_net/final_net_ae_weights.h5'
        
        rnn_kwargs = {
            'train_alpha':train_alpha,
            'train_omega_in':train_omega_in,
            'train_rho_res':train_rho_res,
            'wts_to_be_loaded':True,
        }


    # In[21]:


    def find_and_return_load_wt_file_lists(
            load_dir,
            wt_matcher='weights.hdf5',
            classdict_matcher='class_dict.txt',
        ):
        contents_load_dir = [f for f in os.listdir(load_dir) if os.path.isfile(os.path.join(load_dir, f))]
        load_files_lst = [f for f in contents_load_dir if f.endswith(classdict_matcher)]
        wt_files_lst = [f for f in contents_load_dir if f.endswith(wt_matcher)]

        load_files_lst_startingidx = []
        for i in range(len(load_files_lst)):
            fn = load_files_lst[i]
            idx = fn.find('_')
            load_files_lst_startingidx.append(int(fn[0:idx]))

        wt_files_lst_startingidx = []
        for i in range(len(wt_files_lst)):
            fn = wt_files_lst[i]
            idx = fn.find('_')
            wt_files_lst_startingidx.append(int(fn[0:idx]))

        load_files_sortidx = np.argsort(load_files_lst_startingidx)
        wt_files_sortidx = np.argsort(wt_files_lst_startingidx)

        load_files_lst = np.array(load_files_lst)[load_files_sortidx]
        wt_files_lst = np.array(wt_files_lst)[wt_files_sortidx]

        load_file_rnn = [load_dir + '/' + fn for fn in load_files_lst]
        wt_file_rnn = [load_dir + '/' +  fn for fn in wt_files_lst]
        
        return load_file_rnn, wt_file_rnn


    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[22]:


    load_dir = dir_name_rnn + '/final_net'
    load_file_rnn, wt_file_rnn = find_and_return_load_wt_file_lists(load_dir)


    # In[23]:


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

        total_s_len = 80
        sep_lr_s = ' num_outsteps : {} '.format(num_outsteps)
        sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'>' + sep_lr_s
        sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'<'
        print('\n\n' + '*'*len(sep_lr_s))
        print('' + sep_lr_s+'')
        print('*'*len(sep_lr_s) + '\n\n')

        if behaviour == 'loadCheckpointAndContinueTraining':
            if kk < len(T_sample_output) - 1:
                temp = int((T_sample_output[kk+1] + 0.5*dt_rnn)//dt_rnn)
            else:
                temp = num_outsteps
            checkfile1 = dir_name_ARrnn+'/final_net/final_net-{}_outsteps_rnn_weights.hdf5'.format(temp)
            checkfile2 = dir_name_ARrnn+'/final_net/final_net-{}_outsteps_ae_weights.h5'.format(temp)
            check1 = os.path.exists(checkfile1)
            check2 = os.path.exists(checkfile2)
            if check1 and check2:
                # move on to checking the next time-step
                continue
            else:
                pass
        
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
            batch_size=batch_size,
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
            ESN_flag=True,
            rnn_kwargs=rnn_kwargs,
        )
        
        
        load_dir = dir_name_ARrnn+'/final_net/{}_outsteps'.format(num_outsteps)
        load_file_rnn, wt_file_rnn = find_and_return_load_wt_file_lists(
            load_dir,
            wt_matcher='ESN_weights.hdf5',
            classdict_matcher='ESN_class_dict.txt'
        )
        
        load_file_ae = load_dir + '/final_net-{}_outsteps_ae_class_dict.txt'.format(num_outsteps)
        wt_file_ae = load_dir + '/final_net-{}_outsteps_ae_weights.h5'.format(num_outsteps)
        
        with open(load_dir+'/losses-{}_outsteps.txt'.format(num_outsteps), 'r') as fl:
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

        idxs_to_ignore = 1
        global_clipnorm_min = 3.0
        global_clipnorm = np.max(train_global_gradnorm_hist[idxs_to_ignore:])
        global_clipnorm = 0.1 * np.round(10*global_clipnorm)
        global_clipnorm = max(global_clipnorm, global_clipnorm_min)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_to_use', type=int)
    parser.add_argument('esn_dir_idx', type=int)
    parser.add_argument('num_lyaptimesteps_totrain', nargs='+', type=int)
    
    args = parser.parse_args()
    
    print('gpu_to_use : {} ; esn_dir_idx : {} ; num_lyaptimesteps_totrain : {}'.format(args.gpu_to_use, args.esn_dir_idx, args.num_lyaptimesteps_totrain))
    
    main(args.esn_dir_idx, args.num_lyaptimesteps_totrain, args.gpu_to_use)
