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

def trainAE_and_return_testError(
        x, # [fRMS, lambda_reg, {contractive_lmda}]
        time_stddev,
        og_vars,
        Autoencoder,
        mytimecallback,
        SaveLosses,
        plot_losses,
        readAndReturnLossHistories,
        plot_reconstructed_data,
        plot_latent_states,
        dir_name_ae,
        training_data,
        testing_data,
        val_data,
        boundary_idx_arr,
        prng_seed=42,
        learning_rate_list=[1e-3, 1e-4, 1e-5],
        epochs=2000,
        patience=200,  # parameter for early stopping
        min_delta=1e-6,  # parameter for early stopping
        batch_size=2**6,
        contractive_lmda=0.0,
        use_weights_post_dense=True,
        dropout_rate=0.0,
        latent_space_dim=2,
        enc_layers=[16, 12, 8, 8, 4, 4, 2],
        dec_layers=[2, 4, 4, 8, 8, 12, 16],
        enc_layer_act_func='elu', #'modified_relu_2'
        enc_final_layer_act_func='tanh',
        dec_layer_act_func='elu', #'modified_relu_2'
        dec_final_layer_act_func='tanh',
        reg_name='L2',
        xlabel_kwargs={'fontsize':15},
        ylabel_kwargs={'fontsize':15},
        legend_kwargs={'fontsize':12},
    ):
    
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)
    
    # making ae save directory
    dir_name_ae = dir_name_ae + dir_sep + 'tested_ae'
    if not os.path.isdir(dir_name_ae):
        os.makedirs(dir_name_ae)

    counter = 0
    while True:
        dir_check = 'test_ae_' + str(counter).zfill(3)
        if os.path.isdir(dir_name_ae + dir_sep + dir_check):
            counter += 1
        else:
            break
            
    dir_name_ae = dir_name_ae + dir_sep + dir_check
    os.makedirs(dir_name_ae)
    os.makedirs(dir_name_ae+dir_sep+'plots')
    
    x = np.array(x).flatten()
    fRMS = x[0]
    lambda_reg = x[1]
    if len(x) == 3:
        contractive_lmda = x[2]
    
    stddev = fRMS * np.mean(time_stddev[0:og_vars])
    
    training_specific_params = {
        'learning_rate_list':learning_rate_list,
        'epochs':epochs,
        'patience':patience,
        'min_delta':min_delta,
        'prng_seed':prng_seed,
        'batch_size':batch_size,
        'fRMS':fRMS,
        'stddev':stddev,
        'contractive_lmda':contractive_lmda,
        'dropout_rate':dropout_rate,
    }

    with open(dir_name_ae+dir_sep+'training_specific_params.txt', 'w') as f:
        f.write(str(training_specific_params))
    
    
    ae_net = Autoencoder(
        data_dim=training_data.shape[1],
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        latent_space_dim=latent_space_dim,
        lambda_reg=lambda_reg,
        reg_name=reg_name,
        enc_layer_act_func=enc_layer_act_func,
        enc_final_layer_act_func=enc_final_layer_act_func,
        dec_layer_act_func=dec_layer_act_func,
        dec_final_layer_act_func=dec_final_layer_act_func,
        load_file=None,
        stddev=stddev,
        contractive_lmda=contractive_lmda,
        dropout_rate=dropout_rate,
        use_weights_post_dense=use_weights_post_dense,)
        
    save_path = dir_name_ae+dir_sep+'final_net'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    ae_net.save_class_dict(save_path+dir_sep+'final_net_class_dict.txt')  
        
        
    val_loss_hist = []
    train_loss_hist = []
    lr_change=[0, 0]
    savelosses_cb_vallossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
    savelosses_cb_trainlossarr = np.ones(shape=epochs*len(learning_rate_list))*np.NaN
    starting_lr_idx = 0
    num_epochs_left = epochs
    
    train_MSE_hist = []
    val_MSE_hist = []

    train_NMSE_hist = []
    val_NMSE_hist = []

    train_ls_jacobian_norm_hist = []

    train_real_MSE_hist = []
    val_real_MSE_hist = []

    train_params_MSE_hist = []
    val_params_MSE_hist = []

    # compiling the network
    ae_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),
        loss=losses.MeanSquaredError(),
    #     loss=losses.BinaryCrossentropy(from_logits=False),
        run_eagerly=False,
        metrics=['mse', NMSE(divisor_arr=tf.constant(time_stddev)), real_MSE(og_vars), params_MSE(og_vars)]
    )


    metric_to_use = 'val_mse'
    # implementing early stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=metric_to_use,
        patience=patience,
        restore_best_weights=True,
        verbose=True,
        min_delta=min_delta
    )

    # time callback for each epoch
    timekeeper_cb = mytimecallback()

    # model checkpoint callback
    dir_name_ckpt = dir_name_ae+dir_sep+'checkpoints'
    if not os.path.isdir(dir_name_ckpt):
        os.makedirs(dir_name_ckpt)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=dir_name_ckpt+dir_sep+'checkpoint',#+'/checkpoint--loss={loss:.4f}--vall_loss={val_loss:.4f}',
        monitor=metric_to_use,
        save_best_only=True,
        save_weights_only=True,
        verbose=2,
        period=1  # saves every 5 epochs
    )

    # save losses callback
    savelosses_cb = SaveLosses(
        filepath=dir_name_ckpt+dir_sep+'LossHistoriesCheckpoint',
        val_loss_arr=savelosses_cb_vallossarr,
        train_loss_arr=savelosses_cb_trainlossarr,
        total_epochs=epochs,
        period=1)

    # training the network
    for i in range(starting_lr_idx, len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        K.set_value(ae_net.optimizer.lr, learning_rate)

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
        
        history = ae_net.fit(training_data, training_data,
            epochs=EPOCHS,
            batch_size=batch_size,
#             validation_split=val_split/train_split,
            validation_data=(val_data, val_data),
            callbacks=[early_stopping_cb, timekeeper_cb, checkpoint_cb, savelosses_cb],
            verbose=1,
            shuffle=True,
        )

        val_loss_hist.extend(history.history['val_loss'])
        train_loss_hist.extend(history.history['loss'])
        
        val_MSE_hist.extend(history.history['val_mse'])
        train_MSE_hist.extend(history.history['mse'])
        
        val_NMSE_hist.extend(history.history['val_NMSE'])
        train_NMSE_hist.extend(history.history['NMSE'])
        
        train_ls_jacobian_norm_hist.append(history.history['ls_jacobian_norm'])
        
        val_real_MSE_hist.extend(history.history['val_real_MSE'])
        train_real_MSE_hist.extend(history.history['real_MSE'])
        
        val_params_MSE_hist.extend(history.history['val_params_MSE'])
        train_params_MSE_hist.extend(history.history['params_MSE'])
        
        if i == starting_lr_idx:
            lr_change[i+1] += len(history.history['val_loss'])
        else:
            lr_change.append(lr_change[i]+len(history.history['val_loss']))
    
    temp = []
    for lst in train_ls_jacobian_norm_hist:
        temp.extend(lst)
    train_ls_jacobian_norm_hist_og = train_ls_jacobian_norm_hist
    train_ls_jacobian_norm_hist = np.array(temp)
    
    
    test_metrics = ae_net.evaluate(
        testing_data, testing_data,
    )
    train_metrics = ae_net.evaluate(training_data, training_data)
    val_metrics = ae_net.evaluate(val_data, val_data)

    save_path = dir_name_ae+dir_sep+'final_net'

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
            'val_real_MSE_hist':val_real_MSE_hist,
            'train_real_MSE_hist':train_real_MSE_hist,
            'val_params_MSE_hist':val_params_MSE_hist,
            'train_params_MSE_hist':train_params_MSE_hist,
            'train_ls_jacobian_norm_hist':train_ls_jacobian_norm_hist,
            'lr_change':lr_change,
            'test_loss':test_metrics[0],
            'test_mse':test_metrics[1],
            'train_loss':train_metrics[0],
            'train_mse':train_metrics[1],
            'val_loss':val_metrics[0],
            'val_mse':val_metrics[1],
        }))

    ae_net.save_everything(
        file_name=save_path+dir_sep+'final_net')
    
    # plotting losses
    dir_name_plot = dir_name_ae + '/plots'
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

    plt.savefig(dir_name_plot + '{ds}loss_history.png'.format(ds=dir_sep), dpi=300, bbox_inches='tight')
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
    plt.savefig(dir_name_plot+'/MSE_history.png', dpi=300, bbox_inches='tight')
    # plt.show()
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
    plt.savefig(dir_name_plot+'/NMSE_history.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


    fig, ax = plot_losses(
        training_loss=train_ls_jacobian_norm_hist,
        val_loss=None,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=[r"$\| \nabla_{x} z^{encoded} \|$"],
        xlabel='Epoch',
        ylabel=r"$\| \nabla_{x} v^{encoded} \|$",
        plot_type='plot',
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
    )
    plt.savefig(dir_name_plot+'/train_ls_jacobian_norm_hist.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


    fig, ax = plot_losses(
        training_loss=train_real_MSE_hist,
        val_loss=val_real_MSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training MSE (actual vars)', 'Validation MSE (actual vars)'],
        xlabel='Epoch',
        ylabel='MSE (actual vars)',
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
    )
    plt.savefig(dir_name_plot+'/real_MSE_history.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


    fig, ax = plot_losses(
        training_loss=train_params_MSE_hist,
        val_loss=val_params_MSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training MSE (params)', 'Validation MSE (params)'],
        xlabel='Epoch',
        ylabel='MSE (params)',
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        legend_kwargs=legend_kwargs,
        
    )
    plt.savefig(dir_name_plot+'/params_MSE_history.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()
    
    plt.close('all')
    
    return test_metrics[1]
