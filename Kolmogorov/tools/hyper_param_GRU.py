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

plt.rcParams.update({
    "text.usetex":True,
    "font.family":"serif",
}) # enable tex rendering in matplotlib

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
        bin_begin=0.0,
        xlabel_kwargs={"fontsize":15},
        ylabel_kwargs={"fontsize":15},
        title_kwargs={"fontsize":18},
        legend_kwargs={"fontsize":12},
        title_text = None,
    ):
    
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

    ax.set_xlabel('Prediction Horizon (Lyapunov times)', **xlabel_kwargs)
    ax.set_ylabel('PDF', **ylabel_kwargs)

    ax.grid(True)
    # ax.set_axisbelow(True)

    ax.text(
        0.01 + ax.transAxes.inverted().transform(ax.transData.transform([ph_mean, 0]))[0],
        0.8,
        'mean',
        rotation=90,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor=np.array([255,255,153])/255, alpha=0.6, boxstyle='square,pad=0.2'),
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
            alpha=0.6,
        ),
        # bbox=dict(facecolor='C0', alpha=0.5, boxstyle='round,pad=0.2'),
        horizontalalignment='right',
        verticalalignment='top',
        **legend_kwargs
    )

    if title_text == None:
        title_text = 'nbins = {}'.format(nbins)
    ax.set_title(title_text, **title_kwargs)
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    fig.savefig(save_dir+'/'+savefig_fname+'.pdf', dpi=300, bbox_inches='tight')
    fig.clear()
    plt.close()


def trainGRU_and_return_PH(
        x, # [fRMS, lambda_reg, zoneout]
        time_stddev,
        og_vars,
        RNN_GRU,
        AR_RNN,
        AR_AERNN,
        ae_net,
        mytimecallback,
        SaveLosses,
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
        AR_testing_data_rnn_input,
        AR_testing_data_rnn_output,
        return_params_arr,
        normalize_dataset,
        dt_rnn,
        noise_type,
        ae_data_normalization_arr,
        time_stddev_ogdata,
        time_mean_ogdata,
        T_sample_input,
        T_sample_output,
        rnn_layers_units = [1500],
        stateful = True,
        reg_name='L2',
        dense_layer_act_func=['tanh'],
        use_learnable_state=False,
        use_weights_post_dense=True,
        rnncell_dropout_rate=0.0,
        denselayer_dropout_rate=0.0,
        scalar_weights=[1.0],
        prng_seed=42,
        epochs=1,
        learning_rate_list = [1e-3, 1e-4, 1e-5],
        patience=10,  # parameter for early stopping
        min_delta=1e-6,  # parameter for early stopping
        batch_size=64,
        num_runs=100,
        T_sample_input_AR_ratio=1,
        T_sample_output_AR_ratio=5,
        use_best=False,
        error_threshold=0.5,
        xlabel_kwargs={'fontsize':15},
        ylabel_kwargs={'fontsize':15},
        legend_kwargs={'fontsize':12},
    ):
    
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)
    
    # making ae save directory
    dir_name_rnn = dir_name_rnn + dir_sep + 'tested_rnn'
    if not os.path.isdir(dir_name_rnn):
        os.makedirs(dir_name_rnn)

    counter = 0
    while True:
        dir_check = 'test_rnn_' + str(counter).zfill(3)
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
    fRMS = np.float64(x[0])
    lambda_reg = np.float64(x[1])
    zoneout_rate = np.float64(x[2])
    
    np.savez(
        dir_name_rnn+dir_sep+'normalization_data',
        normalization_arr=[normalization_arr],
    )

    # computing sparsity
    stddev = fRMS * np.mean(time_stddev[0:og_vars])
    
    training_specific_params_dict['fRMS'] = fRMS
    training_specific_params_dict['lambda_reg'] = lambda_reg
    training_specific_params_dict['stddev'] = stddev
    training_specific_params_dict['zoneout_rate'] = zoneout_rate
    
    with open(dir_name_rnn+dir_sep+'training_specific_params.txt', 'w') as f:
        f.write(str(training_specific_params_dict))

    
    save_path = dir_name_rnn+dir_sep+'final_net'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    rnn_net = RNN_GRU(
            data_dim=training_data_rnn_input.shape[-1],
            dt_rnn=dt_rnn,
            lambda_reg=lambda_reg,
            reg_name='L2',
            rnn_layers_units=rnn_layers_units,
            dense_layer_act_func=dense_layer_act_func,
            load_file=None,
            stddev=stddev,
            noise_type=noise_type,
            dense_dim=None,
            use_learnable_state=use_learnable_state,
            stateful=stateful,
            zoneout_rate=zoneout_rate,
            batch_size=batch_size,
            use_weights_post_dense=use_weights_post_dense,
            rnncell_dropout_rate=rnncell_dropout_rate,
            denselayer_dropout_rate=denselayer_dropout_rate,
            scalar_weights=scalar_weights,
        )

    train_loss_hist = []
    val_loss_hist = []
    
    train_NMSE_hist = []
    val_NMSE_hist = []

    train_MSE_hist = []
    val_MSE_hist = []

    lr_change=[0, 0]
    savelosses_cb_vallossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
    savelosses_cb_trainlossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
    starting_lr_idx = 0
    num_epochs_left = epochs
    earlystopping_wait = 0
    
    # compiling the network
    rnn_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),
        loss=losses.MeanSquaredError(),
        metrics=['mse', NMSE(divisor_arr=time_stddev)],
        run_eagerly=False
    )

    # implementing early stopping
    baseline = None
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
#             validation_split=val_split/train_split,
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

    if normalize_dataset == True:
        with open(save_path+dir_sep+'rnn_normalization.txt', 'w') as f:
            f.write(str({
                'normalization_arr':normalization_arr
            }))
            
    for layer in rnn_net.rnn_list:
        if layer.stateful == True:
            layer.reset_states()
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

    rnn_net.save_everything(
        file_name=save_path+dir_sep+'final_net')
    
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
    plt.clf()


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
    plt.clf()
    
    plt.close('all')
    
    num_runs = AR_testing_data_rnn_input.shape[0]
    
    AR_rnn_net = AR_RNN(
        load_file=save_path+'/final_net_class_dict.txt',
        T_input=T_sample_input,
        T_output=T_sample_output,
        stddev=0.0,
        batch_size=num_runs,
        lambda_reg=lambda_reg,
    )
    AR_rnn_net.build(input_shape=tuple(AR_testing_data_rnn_input.shape[0:2]) + tuple(testing_data_rnn_input.shape[2:]))
    AR_rnn_net.load_weights_from_file(save_path+'/final_net_gru_weights.h5')

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

    prediction_horizon_arr = np.empty(shape=num_runs)
    prediction = np.array(AR_AERNN_net(AR_testing_data_rnn_input, training=False))
    prediction = invert_normalization(prediction, ae_data_normalization_arr)

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
        # error = np.linalg.norm(data_out[:, :] - prediction[i, :, :], axis=1)
        error = (data_out[:, :] - prediction[i, :, :])**2
        # error /= norm_sq_time_average(data_out)**0.5
        error = np.mean(np.divide(error, time_stddev_ogdata**2), axis=1)**0.5

        predhor_idx = np.where(error >= error_threshold)[0]
        if predhor_idx.shape[0] == 0:
            predhor_idx = error.shape[0]
        else:
            predhor_idx = predhor_idx[0]

        prediction_horizon_arr[i] = predhor_idx*dt_rnn/lyap_time
        
        run_time = time.time() - run_time
        avg_time = (avg_time*i + run_time)/(i+1)
        eta = avg_time * (num_runs-1 - i)
        print('    {} / {} -- run_time : {:.2f} s -- eta : {:.0f}h {:.0f}m {:.0f}s'.format(
            i+1,
            num_runs,
            run_time,
            float(eta // 3600),
            float((eta%3600)//60),
            float((eta%3600)%60),
        ))

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

    np.savez(
        dir_name_rnn+npsavedata_fname,
        prediction_horizon_arr=prediction_horizon_arr,
        error_threshold=error_threshold,
    )

    with open(dir_name_rnn+npsavedata_fname+'--statistics.txt', 'w') as fl:
        fl.write(s)

    print('analysis time : {} s\n'.format(time.time() - analysis_time))
    
    return median
