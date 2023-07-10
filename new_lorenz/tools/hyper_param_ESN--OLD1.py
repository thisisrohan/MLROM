import os
import math
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

tf.keras.backend.set_floatx('float32')

plt.rcParams.update({"text.usetex":True}) # enable tex rendering in matplotlib

FTYPE = np.float32
ITYPE = np.int32

strategy = None
# strategy = tf.distribute.MirroredStrategy()

dir_sep = '/'

class NMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, divisor_arr, name='NMSE', **kwargs):
        super(NMSE, self).__init__(name, **kwargs)
        self.divisor_arr = divisor_arr

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true / self.divisor_arr
        y_pred = y_pred / self.divisor_arr
        return super(NMSE, self).update_state(y_true, y_pred, sample_weight)

class real_MSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, og_vars, name='real_MSE', **kwargs):
        super().__init__(name, **kwargs)
        self.og_vars = og_vars

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:, 0:self.og_vars]
        y_pred = y_pred[:, 0:self.og_vars]
        return super().update_state(y_true, y_pred, sample_weight)
    
class params_MSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, og_vars, name='params_MSE', **kwargs):
        super().__init__(name, **kwargs)
        self.og_vars = og_vars

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:, self.og_vars:]
        y_pred = y_pred[:, self.og_vars:]
        return super().update_state(y_true, y_pred, sample_weight)
        
def get_ensemble_prediction(ensemble_lst, inputs, kwargs={}):
    pred = 0
    for i in range(len(ensemble_lst)):
        pred += np.array(ensemble_lst[i](inputs, **kwargs))
    pred /= len(ensemble_lst)
    return pred
    
def rescale_data(data, normalization_arr):
    '''
    data - [num_batches x num_timesteps x num_states]
    normalization_arr = [2 x num_states]
    '''
    new_data = data.copy()
    shape = new_data.shape
    for i in range(data.shape[-1]):
        new_data[:, i] -= normalization_arr[0, i]
        new_data[:, i] /= normalization_arr[1, i]

    return new_data

def norm_sq_time_average(data):
    data_norm_sq = np.zeros(shape=data.shape[0])
    for i in range(data.shape[1]):
        data_norm_sq[:] += data[:, i]**2
    # integrating using the trapezoidal rule
    norm_sq_time_avg = np.sum(data_norm_sq) - 0.5*(data_norm_sq[0]+data_norm_sq[-1])
    norm_sq_time_avg /= data_norm_sq.shape[0]
    return norm_sq_time_avg

def invert_normalization(data, normalization_arr):
    new_data = data.copy()
    shape = new_data.shape
    for i in range(shape[-1]):
        if len(shape) == 2:
            new_data[:, i] *= normalization_arr[1, i]
            new_data[:, i] += normalization_arr[0, i]
        elif len(shape) == 3:
            new_data[:, :, i] *= normalization_arr[1, i]
            new_data[:, :, i] += normalization_arr[0, i]
    return new_data

def plot_histogram_and_save(
    prediction_horizon_arr, median,
    save_dir,
    savefig_fname='pre_ARtraining',
    bin_width=0.1,
    bin_begin=0.0):
    
    fig, ax = plt.subplots()
    prediction_horizon_arr.sort()

    ph_mean = np.mean(prediction_horizon_arr)
    ph_stddev = np.std(prediction_horizon_arr)
    ph_max = np.max(prediction_horizon_arr)
    ph_min = np.min(prediction_horizon_arr)
    
    bin_end = bin_width*np.round((np.max(prediction_horizon_arr)+0.5*bin_width)//bin_width)
    nbins = int(np.round(bin_end/bin_width))

    ax.hist(prediction_horizon_arr, bins=nbins, range = [bin_begin, bin_end], density=True)
    ax.axvline(ph_mean, linewidth=0.9, linestyle='--', color='k')

    ax.set_xlabel('Prediction Horizon (Lyapunov times)')
    ax.set_ylabel('PDF')

    ax.grid(True)
    # ax.set_axisbelow(True)

    ax.text(
        0.01 + ax.transAxes.inverted().transform(ax.transData.transform([ph_mean, 0]))[0],
        0.8,
        'mean',
        rotation=90,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor=np.array([255,255,153])/255, alpha=1, boxstyle='square,pad=0.2'),
        transform=ax.transAxes
    )

    text_xy = [0.95, 0.95]
    ax.text(
        text_xy[0],
        text_xy[1],
        'mean : {:.4f}\nmedian : {:.4f}\nmax : {:.4f}\nmin : {:.4f}\nstddev : {:.4f}'.format(
            ph_mean,
            median,
            ph_max,
            ph_min,
            ph_stddev,
        ),
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round",
            ec=(0.6, 0.6, 1),
            fc=(0.9, 0.9, 1),
        ),
        # bbox=dict(facecolor='C0', alpha=0.5, boxstyle='round,pad=0.2'),
        horizontalalignment='right',
        verticalalignment='top'
    )

    ax.set_title('nbins = {}'.format(nbins))
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    fig.savefig(save_dir+'/'+savefig_fname+'.png', dpi=300, bbox_inches='tight')
    fig.clear()
    plt.close()


def trainESN_and_return_PH(
        x, # [fRMS, lambda_reg, rho_res, omega_in, alpha, degree_of_connectivity]
        time_stddev,
        og_vars,
        ESN,
        plot_losses,
        dir_name_rnn,
        boundary_idx_arr,
        lyapunov_time_arr,
        sim_data_dict,
        RNN_specific_data_dict,
        training_specific_params_dict,
        normalization_arr,
        training_data_rnn_input,
        training_data_rnn_output,
        testing_data_rnn_input,
        testing_data_rnn_output,
        val_data_rnn_input,
        val_data_rnn_output,
        return_params_arr,
        normalize_dataset,
        dt_rnn,
        noise_type,
        num_ensemble_mems=1,
        ESN_layers_units = [1500],
        stateful = True,
        usebias_Win = [False],
        ESN_cell_activations = ['tanh'],
        usebias_Wout = True,
        activation_post_Wout = 'linear',
        use_weights_post_dense = False,
        prng_seed=42,
        epochs=1,
        patience=2,  # parameter for early stopping
        min_delta=1e-6,  # parameter for early stopping
        batch_size=1,
        num_runs=100,
        T_sample_input_AR_ratio=1,
        T_sample_output_AR_ratio=5,
        use_best=False,
        error_threshold=0.5,
    ):
    
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)
    
    # making ae save directory
    dir_name_rnn = dir_name_rnn + dir_sep + 'tested_ESN'
    if not os.path.isdir(dir_name_rnn):
        os.makedirs(dir_name_rnn)

    counter = 0
    while True:
        dir_check = 'test_ESN_' + str(counter).zfill(3)
        if os.path.isdir(dir_name_rnn + dir_sep + dir_check):
            counter += 1
        else:
            break
            
    dir_name_rnn = dir_name_rnn + dir_sep + dir_check
    os.makedirs(dir_name_rnn)
    os.makedirs(dir_name_rnn+dir_sep+'plots')
    
    with open(dir_name_rnn+dir_sep+'sim_data_AE_params.txt', 'w') as f:
        f.write(str(sim_data_dict))
        
    with open(dir_name_rnn+dir_sep+'RNN_specific_data.txt', 'w') as f:
        f.write(str(RNN_specific_data_dict))
    
    x = np.array(x).flatten()
    fRMS = x[0]
    lambda_reg = x[1]
    rho_res = [x[2]]
    omega_in = [x[3]]
    alpha = [x[4]]
    degree_of_connectivity = [x[5]]
    
    np.savez(
        dir_name_rnn+dir_sep+'normalization_data',
        normalization_arr=[normalization_arr],
    )

    # computing sparsity
    sparsity = [1-degree_of_connectivity[i]/(ESN_layers_units[i]-1) for i in range(len(ESN_layers_units))]
    stddev = fRMS * np.mean(time_stddev[0:og_vars])
    
    training_specific_params_dict['fRMS'] = fRMS
    training_specific_params_dict['lambda_reg'] = lambda_reg
    training_specific_params_dict['stddev'] = stddev
    
    with open(dir_name_rnn+dir_sep+'training_specific_params.txt', 'w') as f:
        f.write(str(training_specific_params_dict))

    
    save_path = dir_name_rnn+dir_sep+'final_net'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    ensemble_lst = []
    for i in range(num_ensemble_mems):
        rnn_net = ESN(
            data_dim=training_data_rnn_input.shape[2],
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
            prng_seed=np.random.randint(low=0, high=prng_seed)*prng_seed,
            usebias_Wout=usebias_Wout,
            use_weights_post_dense=use_weights_post_dense,
            activation_post_Wout=activation_post_Wout,
            scalar_weights=[],
        )
        rnn_net.build(input_shape=(1,) + training_data_rnn_input.shape[1:])
        rnn_net.save_class_dict(save_path+dir_sep+'{}_final_net_class_dict.txt'.format(i))
        ensemble_lst.append(rnn_net)    

    val_loss_hist = []
    train_loss_hist = []
    
    for i_en in range(num_ensemble_mems):
        print('--- ENSEMBLE MEMBER {}/{} ---'.format(i_en+1, num_ensemble_mems))
        rnn_net = ensemble_lst[i_en]
        # compiling the network
        rnn_net.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=losses.MeanSquaredError(),
            metrics=['mse'],
            run_eagerly=False
        )

        lambda_reg = float(lambda_reg)

        Wout_best = 0
        val_mse_best = np.inf
        Wout_candidate = 0
        wait = 0
        if use_weights_post_dense == True:
            postWout_candidate = 0
            h_activation = tf.keras.activations.get(activation_post_Wout)

        hidden_units = ESN_layers_units[-1]
        output_units = rnn_net.data_dim

        Hb_shape = [hidden_units, hidden_units]
        Yb_shape = [output_units, hidden_units]
        if usebias_Wout == True:
            Hb_shape[0] += 1
            Hb_shape[1] += 1
            Yb_shape[1] += 1

        Hb = np.zeros(shape=Hb_shape, dtype=FTYPE)
        Yb = np.zeros(shape=Yb_shape, dtype=FTYPE)
        eye_Hb = np.eye(Hb.shape[0], dtype=FTYPE)

        num_batches = training_data_rnn_input.shape[0]


        for i in range(epochs):
            # for layer in rnn_net.ESN_layers:
            #     layer.reset_states()

            epoch_totaltime = time.time()

            total_s_len = 80
            sep_lr_s = ' EPOCH : {} '.format(i+1)
            sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'-' + sep_lr_s
            sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'-'
            print('\n\n' + '-'*len(sep_lr_s))
            print('\n' + sep_lr_s+'\n')
            print('-'*len(sep_lr_s) + '\n\n')

            # '''
            ### computing Wout
            Hb[:, :] = 0
            Yb[:, :] = 0
            epoch_time = time.time()
            avg_time = 0.
            for j in range(training_data_rnn_input.shape[0]):
                batch_time = time.time()
                h = np.array(rnn_net(training_data_rnn_input[j:j+1], manual_training=True))
                h = h[0]
                # y = invert_fn(training_data_rnn_output[j])
                y = training_data_rnn_output[j]
                if usebias_Wout == True:
                    h = np.concatenate((h, np.ones(shape=(h.shape[0], 1))), axis=1)
                Hb = Hb + np.matmul(np.transpose(h), h)
                Yb = Yb + np.matmul(np.transpose(y), h)
                batch_time = time.time() - batch_time
                avg_time = (avg_time*j + batch_time)/(j+1)
                eta = avg_time * (num_runs-1 - j)
                print('{} / {} -- Wout batch_time : {:.2f} s -- eta : {:.0f}h {:.0f}m {:.0f}s'.format(
                    j+1,
                    num_runs,
                    batch_time,
                    float(eta // 3600),
                    float((eta%3600)//60),
                    float((eta%3600)%60),
                ))

            Wout = np.matmul(
                Yb,
                np.linalg.inv(Hb + lambda_reg*np.eye(Hb.shape[0]))
            )
            Wout = np.transpose(Wout)

            if use_weights_post_dense == True:
                ### computing postWout
                HYb = 0
                HHb = 0
                for j in range(training_data_rnn_input.shape[0]):
                    batch_time = time.time()
                    h = np.array(rnn_net(training_data_rnn_input[j:j+1], training=True))
                    h = h[0]
                    h = np.matmul(h, Wout[0:ESN_layers_units[-1], :])
                    if usebias_Wout == True:
                        h = h + Wout[ESN_layers_units[-1]:, :]
                    h = np.array(h_activation(h))
                    y = training_data_rnn_output[j]
                    HYb = HYb + np.sum(h*y, axis=0)
                    HHb = HHb + np.sum(h**2, axis=0)

                    print('{} / {} -- postWout batch_time : {} sec'.format(
                        j+1,
                        training_data_rnn_input.shape[0],
                        time.time() - batch_time
                    ))

                postWout = HYb / (HHb + lambda_reg)

            print('\nepoch_time : {} sec'.format(time.time() - epoch_time))


            Wout_candidate = Wout_candidate*i/(i+1) + Wout*1/(i+1)
            tf.keras.backend.set_value(rnn_net.Wout.kernel, Wout_candidate[0:ESN_layers_units[-1], :])
            if usebias_Wout == True:
                tf.keras.backend.set_value(rnn_net.Wout.bias, Wout_candidate[-1, :])

            if use_weights_post_dense == True:
                postWout_candidate = postWout_candidate*i/(i+1) + postWout*1/(i+1)
                tf.keras.backend.set_value(rnn_net.postWout.individual_weights, postWout_candidate)

            for layer in rnn_net.ESN_layers:
                layer.reset_states()

            print('\nval mse')
            # '''
            val_mse = 0
            for j in range(val_data_rnn_input.shape[0]):
                batch_time = time.time()
                val_pred = np.array(rnn_net(val_data_rnn_input[j:j+1], training=False))
                temp = (val_pred - val_data_rnn_output[j:j+1])**2
                temp = np.mean(temp, axis=-1) # do a sqrt here to get rmse
                temp = np.mean(temp, axis=-1)
                temp = np.mean(temp, axis=-1)
                val_mse = val_mse*j/(j+1) + temp*1/(j+1)
                print('{} / {} -- batch_time : {} sec'.format(
                    j+1,
                    val_data_rnn_input.shape[0],
                    time.time() - batch_time
                ))


            for layer in rnn_net.ESN_layers:
                layer.reset_states()

            print('\ntraining mse')
            # '''
            train_mse = 0
            for j in range(training_data_rnn_input.shape[0]):
                batch_time = time.time()
                train_pred = np.array(rnn_net(training_data_rnn_input[j:j+1], training=False))
                temp = (train_pred - training_data_rnn_output[j:j+1])**2
                temp = np.mean(temp, axis=-1) # do a sqrt here to get rmse
                temp = np.mean(temp, axis=-1)
                temp = np.mean(temp, axis=-1)
                train_mse = train_mse*j/(j+1) + temp*1/(j+1)
                print('{} / {} -- batch_time : {} sec'.format(
                    j+1,
                    training_data_rnn_input.shape[0],
                    time.time() - batch_time
                ))

            for layer in rnn_net.ESN_layers:
                layer.reset_states()

            val_loss_hist.append(val_mse)
            train_loss_hist.append(train_mse)

            # print('\ntest_mse : {}'.format(test_mse))
            print('\ntrain_mse : {}'.format(train_mse))
            print('val_mse : {}'.format(val_mse))
            if val_mse + min_delta <= val_mse_best:
                print('val_mse improved from {}'.format(val_mse_best))
                Wout_best = Wout_candidate
                val_mse_best = val_mse
                wait = 0
            else:
                wait += 1
                print('val_mse did not improve from {}, wait : {}'.format(val_mse_best, wait))

            print('\nTotal epoch computation time : {} sec'.format(time.time()-epoch_totaltime))

            if wait >= patience:
                print('\nearly stopping')
                break


        if use_best == True:
            tf.keras.backend.set_value(rnn_net.Wout.kernel, Wout_best[0:ESN_layers_units[-1], :])
            if usebias_Wout == True:
                tf.keras.backend.set_value(rnn_net.Wout.bias, Wout_best[-1, :])
        print('\ntest mse')
        test_mse = 0
        for j in range(testing_data_rnn_input.shape[0]):
            print('{} / {}'.format(j+1, testing_data_rnn_input.shape[0]))
            test_pred = np.array(rnn_net(testing_data_rnn_input[j:j+1], training=False))
            temp = (test_pred - testing_data_rnn_output[j:j+1])**2
            temp = np.mean(temp, axis=-1) # do a sqrt here to get rmse
            temp = np.mean(temp, axis=-1)
            temp = np.mean(temp, axis=-1)
            test_mse = test_mse*j/(j+1) + temp*1/(j+1)
        print('test_mse : {}'.format(test_mse))

        for layer in rnn_net.ESN_layers:
            layer.reset_states()
            
        with open(save_path+dir_sep+'{}_losses.txt'.format(i_en), 'w') as f:
            f.write(str({
                'val_loss_hist':val_loss_hist,
                'train_loss_hist':train_loss_hist,
        #             'lr_change':lr_change,
                'test_loss':test_mse
            }))


        rnn_net.save_everything(
            file_name=save_path+dir_sep+'{}_final_net'.format(i_en))


    if normalize_dataset == True:
        with open(save_path+dir_sep+'rnn_normalization.txt', 'w') as f:
            f.write(str({
                'normalization_arr':normalization_arr
            }))
            
    
    s_in = testing_data_rnn_input.shape
    testing_data_rnn_input = testing_data_rnn_input.reshape((1, s_in[0]*s_in[1]) + s_in[2:])

    s_out = testing_data_rnn_output.shape
    testing_data_rnn_output = testing_data_rnn_output.reshape((1, s_out[0]*s_out[1]) + s_out[2:])

    T_sample_input_AR = T_sample_input_AR_ratio*np.mean(lyapunov_time_arr)#50.1*dt_rnn
    num_sample_input_AR = int((T_sample_input_AR+0.5*dt_rnn)//dt_rnn)

    T_sample_output_AR = T_sample_output_AR_ratio*np.mean(lyapunov_time_arr)
    num_sample_output_AR = int((T_sample_output_AR+0.5*dt_rnn)//dt_rnn)

    num_offset_AR = num_sample_input_AR
    T_offset_AR = num_offset_AR*dt_rnn
    
    data_to_consider = 'testing'

    data_in = eval(data_to_consider+'_data_rnn_input')
    data_out = eval(data_to_consider+'_data_rnn_output')

    batch_idx = np.random.randint(low=0, high=data_in.shape[0])
    maxpossible_num_runs = data_in.shape[1]-(num_sample_input_AR+num_sample_output_AR)
    
    num_runs = np.min([num_runs, maxpossible_num_runs])

    data_idx_arr = np.linspace(0, maxpossible_num_runs-1, num_runs, dtype=np.int32)

    savefig_fname = 'pre_ARtraining-'+data_to_consider+'data'
    npsavedata_fname = '/prediction_horizons-'+data_to_consider+'data'
    plot_dir = '/plots'

    analysis_time = time.time()

    sidx1 = dir_name_rnn[::-1].index('/')
    sidx2 = dir_name_rnn[-sidx1-2::-1].index('/')
    print(dir_name_rnn[-(sidx1+sidx2+1):])
    print('num_runs :', num_runs)

    prediction_horizon_arr = np.empty(shape=num_runs)

    for i in range(num_runs):
        data_idx = data_idx_arr[i]

        # for j in range(len(rnn_data_boundary_idx_arr)):
        #     if data_idx < rnn_data_boundary_idx_arr[j]:
        #         case_idx = j
        #         break
        lyap_time = lyapunov_time_arr[0]

        ### picking the data
        data_ = data_in[0:1, data_idx:data_idx+(num_sample_input_AR+num_sample_output_AR), :]

        ### doing the predictions
        prediction_lst = []
        
        for rnn_net in ensemble_lst:
            for layer in rnn_net.ESN_layers:
                layer.reset_states()

        input_preds = np.array(get_ensemble_prediction(
            ensemble_lst,
            data_[:, 0:num_sample_input_AR, :],
            {'training':False}
        ))[0]

        prediction_lst.append(input_preds[-1])

        for j in range(1, num_sample_output_AR):
            data_in_j = np.array([[prediction_lst[-1]]])
            output = np.array(get_ensemble_prediction(
                ensemble_lst,
                data_in_j,
                {'training':False}
            ))[0, 0]
            prediction_lst.append(output)
        prediction_lst = np.stack(prediction_lst)
        # prediction_lst = invert_normalization(prediction_lst, normalization_arr)
        
        data_out = data_[0, num_sample_input_AR:num_sample_input_AR+num_sample_output_AR, :]
        # data_out = invert_normalization(data_out, normalization_arr)

        ### Error and prediction horizon
        # error = np.linalg.norm(data_out[:, :] - prediction[i, :, :], axis=1)
        error = (data_out[:, :] - prediction_lst[:, :])**2
        # error /= norm_sq_time_average(data_out)**0.5
        error = np.mean(np.divide(error, time_stddev**2), axis=1)**0.5

        predhor_idx = np.where(error >= error_threshold)[0]
        if predhor_idx.shape[0] == 0:
            predhor_idx = error.shape[0]
        else:
            predhor_idx = predhor_idx[0]

        prediction_horizon_arr[i] = predhor_idx*dt_rnn/lyap_time

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

    npsavedata_fname = '/prediction_horizons-'+data_to_consider+'data'
    np.savez(
        dir_name_rnn+npsavedata_fname,
        prediction_horizon_arr=prediction_horizon_arr,
        error_threshold=error_threshold,
    )

    with open(dir_name_rnn+npsavedata_fname+'--statistics.txt', 'w') as fl:
        fl.write(s)

    print('analysis time : {} s\n'.format(time.time() - analysis_time))
    
    return median
