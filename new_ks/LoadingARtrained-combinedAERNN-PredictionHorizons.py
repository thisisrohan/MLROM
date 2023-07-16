#!/usr/bin/env python
# coding: utf-8

import os
import math
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, fft

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
    "font.family":"serif",
})

from numpy import *

FTYPE = np.float32
ITYPE = np.int32

array = np.array
float32 = np.float32
int32 = np.int32
float64 = np.float64
int64 = np.int64


def invert_normalization(data, normalization_arr):
    new_data = np.empty_like(data)
    new_data[:] = data[:]
    new_data *= normalization_arr[1]
    new_data += normalization_arr[0]
    return new_data

def prediction_horizons(**kwargs):
    num_outsteps = kwargs['num_outsteps']
    dir_name_AR_AErnn = kwargs['dir_name_AR_AErnn']
    Autoencoder = kwargs['Autoencoder']
    # all_data = kwargs['all_data']
    data_rnn_input = kwargs['data_rnn_input']
    data_rnn_output = kwargs['data_rnn_output']
    AR_RNN = kwargs['AR_RNN']
    T_sample_input_rnn = kwargs['T_sample_input_rnn']
    T_sample_output_rnn = kwargs['T_sample_output_rnn']
    AR_AERNN = kwargs['AR_AERNN']
    normalization_constant_arr_rnn = kwargs['normalization_constant_arr_rnn']
    normalization_constant_arr_aedata = kwargs['normalization_constant_arr_aedata']
    time_stddev_ogdata = kwargs['time_stddev_ogdata']
    time_mean_ogdata = kwargs['time_mean_ogdata']
    batch_size = kwargs['batch_size']
    num_runs = kwargs.pop('num_runs', 100)
    error_threshold = kwargs.pop('error_threshold', 0.5)
    rnn_data_boundary_idx_arr = kwargs['rnn_data_boundary_idx_arr']
    lyapunov_time_arr = kwargs['lyapunov_time_arr']
    savefig_fname = kwargs['savefig_fname']
    data_to_consider = kwargs['data_to_consider']
    bin_width = kwargs.pop('bin_width', 0.05)
    bin_begin = kwargs.pop('bin_begin', 0.0)
    density = kwargs.pop('hist_pdf_flag', True)
    rnn_wt_extension = kwargs.pop('rnn_wt_extension', 'h5')
    ae_load_file = kwargs.pop('ae_load_file', None)
    ae_wt_file = kwargs.pop('ae_wt_file', None)
    rnn_load_file = kwargs.pop('rnn_load_file', None)
    rnn_wt_file = kwargs.pop('rnn_wt_file', None)
    use_ae_data = kwargs.pop('use_ae_data', True)
    xlabel_kwargs = kwargs.pop('xlabel_kwargs', {'fontsize':15})
    ylabel_kwargs = kwargs.pop('ylabel_kwargs', {'fontsize':15})
    title_kwargs = kwargs.pop('title_kwargs', {'fontsize':18})
    legend_kwargs = kwargs.pop('legend_kwargs', {'fontsize':12}),
    plot_histogram_and_save = kwargs.pop('plot_histogram_and_save')
    
    if ae_load_file == None:
        ae_load_file = dir_name_AR_AErnn+'/final_net/final_net-{}_outsteps_ae_class_dict.txt'.format(num_outsteps)
    if ae_wt_file == None:
        ae_wt_file = dir_name_AR_AErnn+'/final_net/final_net-{}_outsteps_ae_weights.h5'.format(num_outsteps)

    if rnn_load_file == None:
        rnn_load_file = dir_name_AR_AErnn+'/final_net/final_net-{}_outsteps_rnn_class_dict.txt'.format(num_outsteps)
    if rnn_wt_file == None:
        rnn_wt_file = dir_name_AR_AErnn+'/final_net/final_net-{}_outsteps_rnn_weights.'.format(num_outsteps)+rnn_wt_extension
    
    if use_ae_data == True:
        ae_net = Autoencoder(data_rnn_input.shape[2:], load_file=ae_load_file)
        ae_net.load_weights_from_file(ae_wt_file)
    else:
        ae_net = None
        normalization_constant_arr_aedata = normalization_constant_arr_rnn
        normalization_constant_arr_rnn = None

    data_in_og = data_rnn_input
    data_out_og = data_rnn_output
    num_runs = np.min([num_runs, data_in_og.shape[0]])
    print('num_runs :', num_runs)

    rnn_net = AR_RNN(
        load_file=rnn_load_file,
        T_input=T_sample_input_rnn,
        T_output=T_sample_output_rnn,
        stddev=0.0,
        batch_size=num_runs,
        # stateful=stateful,
    )
        
    rnn_net.build(input_shape=(num_runs, data_rnn_input.shape[1], rnn_net.data_dim))
    rnn_net.load_weights_from_file(rnn_wt_file)
    
    AR_AERNN_net = AR_AERNN(
        ae_net,
        rnn_net,
        normalization_constant_arr_rnn,
        normalization_constant_arr_aedata,
        0.0,
        time_stddev_ogdata,
        time_mean_ogdata,
    )
    # AR_AERNN_net.build(input_shape=(batch_size,)+data_rnn_input.shape[1:])
    
    # data_idx_arr = np.arange(data_in_og.shape[0])
    # np.random.shuffle(data_idx_arr)
    data_idx_arr = np.linspace(0, data_in_og.shape[0]-1, num_runs, dtype=np.int32)

    prediction_horizon_arr_og = np.empty(shape=num_runs)
    prediction_horizon_arr_new = np.empty(shape=num_runs)
    # pod_eigvals_dataout_arr = np.empty(shape=(num_runs, data_out_og.shape[-1]))
    # pod_eigvals_pred_arr = np.empty(shape=(num_runs, data_out_og.shape[-1]))
    # pod_covmat_dataout = np.zeros(shape=(data_out_og.shape[-1], data_out_og.shape[-1]))
    # pod_covmat_pred = np.zeros(shape=(data_out_og.shape[-1], data_out_og.shape[-1]))

    # prediction = rnn_net.predict(data_in_og[data_idx_arr[0:num_runs], :, :])
    prediction = np.array(AR_AERNN_net(data_in_og[data_idx_arr[0:num_runs]], training=False))
    prediction = invert_normalization(prediction, normalization_constant_arr_aedata)

    # energySpectrum_dataout = 0.0
    # energySpectrum_pred = 0.0
    
    dt_rnn = rnn_net.dt_rnn
    for i in range(num_runs):
        data_idx = data_idx_arr[i]

        # for j in range(len(rnn_data_boundary_idx_arr)):
        #     if data_idx < rnn_data_boundary_idx_arr[j]:
        #         case_idx = j
        #         break
        lyap_time = lyapunov_time_arr[0]

        data_out = data_out_og[data_idx]
        # data_out = rescale_data(data_out, normalization_arr)
        data_out = invert_normalization(data_out, normalization_constant_arr_aedata)

        # pod_dataout = data_out - np.mean(data_out, axis=0)
        # pod_dataout = np.matmul(pod_dataout.transpose(), pod_dataout) / (pod_dataout.shape[0] - 1)
        # pod_covmat_dataout += pod_dataout
        # pod_dataout = np.abs(np.linalg.eigvals(pod_dataout))
        # pod_dataout = np.sort(pod_dataout)
        # pod_dataout = pod_dataout[::-1]
        # pod_eigvals_dataout_arr[i, :] = pod_dataout
        
        # pod_prediction = prediction[i, :, :] - np.mean(prediction[i, :, :], axis=0)
        # pod_prediction = np.matmul(pod_prediction.transpose(), pod_prediction) / (pod_prediction.shape[0] - 1)
        # pod_covmat_pred += pod_prediction
        # pod_prediction = np.abs(np.linalg.eigvals(pod_prediction))
        # pod_prediction = np.sort(pod_prediction)
        # pod_prediction = pod_prediction[::-1]
        # pod_eigvals_pred_arr[i, :] = pod_prediction
        
        # FourierCoeffs_dataout = fft.fft(data_out, axis=1)
        # energySpectrum_dataout_i = FourierCoeffs_dataout.real**2 + FourierCoeffs_dataout.imag**2
        # energySpectrum_dataout_i = np.mean(energySpectrum_dataout_i, axis=0)
        # normalizer = np.sum(energySpectrum_dataout_i)
        # energySpectrum_dataout_i = energySpectrum_dataout_i / normalizer
        # energySpectrum_dataout = (i*energySpectrum_dataout + energySpectrum_dataout_i)/(i+1)
        
        # FourierCoeffs_pred = fft.fft(prediction[i, :, :], axis=1)
        # energySpectrum_pred_i = FourierCoeffs_pred.real**2 + FourierCoeffs_pred.imag**2
        # energySpectrum_pred_i = np.mean(energySpectrum_pred_i, axis=0)
        # energySpectrum_pred_i = energySpectrum_pred_i / normalizer
        # energySpectrum_pred = (i*energySpectrum_pred + energySpectrum_pred_i)/(i+1)
        
        # prediction = rnn_net.predict(data_in_og[data_idx:data_idx+1, :, :])

        ### Error and prediction horizon
        #-- og way --#
        error_og = (data_out - prediction[i])**2
        error_og = np.divide(error_og, time_stddev_ogdata**2)
        error_og = np.reshape(error_og, (error_og.shape[0], -1))
        error_og = np.mean(error_og, axis=1)**0.5
        #-- new way --#
        error_new = (data_out - prediction[i])**2
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

    # pod_eigvals_dataout_arr_mean = np.mean(pod_eigvals_dataout_arr, axis=0)
    # pod_eigvals_pred_arr_mean = np.mean(pod_eigvals_pred_arr, axis=0)
    
    # pod_covmat_dataout /= num_runs
    # pod_covmat_pred /= num_runs

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

    if savefig_fname != None:
        npsavedata_fname = '/prediction_horizons-'+data_to_consider+'data--{}outsteps'.format(num_outsteps)
        np.savez(
            dir_name_AR_AErnn+npsavedata_fname,
            prediction_horizon_arr_og=prediction_horizon_arr_og,
            prediction_horizon_arr_new=prediction_horizon_arr_new,
            error_threshold=error_threshold,
        )

        with open(dir_name_AR_AErnn+npsavedata_fname+'--statistics.txt', 'w') as fl:
            fl.write(s1+'\n\n')
            fl.write(s2)
        # npsavepod_fname = '/pod_eigvals-'+data_to_consider+'data--{}outsteps'.format(num_outsteps)
        # np.savez(
        #     dir_name_AR_AErnn+npsavepod_fname,
        #     pod_eigvals_dataout_arr=pod_eigvals_dataout_arr,
        #     pod_eigvals_pred_arr=pod_eigvals_pred_arr,
        # )



    if savefig_fname is not None:
        plot_histogram_and_save(
            prediction_horizon_arr_og, median_og,
            save_dir=dir_name_AR_AErnn+'/plots/',
            savefig_fname=savefig_fname+'--{}outsteps.pdf'.format(num_outsteps)+'--OG_error',
        )

        plot_histogram_and_save(
            prediction_horizon_arr_new, median_new,
            save_dir=dir_name_AR_AErnn+'/plots/',
            savefig_fname=savefig_fname+'--{}outsteps.pdf'.format(num_outsteps)+'--NEW_error',
        )

    # fig_eigvals, ax_eigvals = plt.subplots()
    # ax_eigvals.semilogy(pod_eigvals_dataout_arr_mean, linestyle='--', marker='s', linewidth=0.9, markersize=2)
    # ax_eigvals.semilogy(pod_eigvals_pred_arr_mean, linestyle='--', marker='^', linewidth=0.9, markersize=2)
    # ax_eigvals.grid(True, which='both', axis='y')
    # ax_eigvals.grid(True, which='major', axis='x')    
    # ax_eigvals.legend([r'True Data', r'Predicted Data'], **legend_kwargs)
    # ax_eigvals.set_axisbelow(True)
    # ax_eigvals.set_title('Eigenvalues of the covariance matrix', **title_kwargs)
    # if savefig_fname is not None:
    #     fig_eigvals.savefig(
    #         dir_name_AR_AErnn+'/plots/'+savefig_fname+'--eigvals--{}outsteps.pdf'.format(num_outsteps),
    #         dpi=300,
    #         bbox_inches='tight')
    #     fig_eigvals.clear()
    #     plt.close()
    # else:
    #     plt.show()
    #     print('')

        

    # fig_covmat = plt.figure(figsize=(5.0*3, 5.0*1))
    # subplot1 = 1
    # subplot2 = subplot1 + 1
    
    # pod_covmat_dataout = np.divide(np.transpose(pod_covmat_dataout), np.diag(pod_covmat_dataout)).transpose()
    # pod_covmat_pred = np.divide(np.transpose(pod_covmat_pred), np.diag(pod_covmat_dataout)).transpose()

    # vmin_snap = 0.5
    # vmax_snap = 0.5
    # vmin = np.min([
    #     pod_covmat_dataout.min(),
    #     pod_covmat_pred.min()
    # ])
    # vmin = min(vmin, -1.0)
    # vmin = -vmin_snap*np.round(-vmin/vmin_snap + 0.5)
    # vmax = np.max([
    #     pod_covmat_dataout.max(),
    #     pod_covmat_pred.max()
    # ])
    # vmax = max(vmax, 1.0)
    # vmax = vmax_snap*np.round(vmax/vmax_snap + 0.5)

    # # plotting the original data
    # ax_covmat_orig = fig_covmat.add_subplot(1, 3, subplot1)
    # im_orig = ax_covmat_orig.imshow(
    #     pod_covmat_dataout,
    #     aspect='equal',
    #     origin='upper',
    #     vmin=vmin,
    #     vmax=vmax
    # )
    # ax_covmat_orig.set_title('Covariance Matrix (True Data)', **title_kwargs)
    # # xticks = np.arange(0, N, int((xticks_snapto+0.5*delta_t)//delta_t))
    # # ax_covmat_orig.set_xticks(ticks=xticks)
    # # ax_covmat_orig.set_xticklabels(np.round(xticks*delta_t, 1))
    # # ax_covmat_orig.tick_params(axis='x', rotation=270+45)
    # # yticks = np.linspace(0, 1, num_yticks)*(len(xgrid)-1)
    # # yticklabels = np.round(xgrid[0]+np.linspace(0, 1, yticks.shape[0])*(xgrid[-1]-xgrid[0]), 2)
    # # ax_covmat_orig.set_yticks(ticks=yticks)
    # # ax_covmat_orig.set_yticklabels(yticklabels)
    # # ax_covmat_orig.set_xlabel(xlabel)
    # # ax_covmat_orig.set_ylabel(ylabel)

    # # plotting the predicted data
    # ax_covmat_predict = fig_covmat.add_subplot(1, 3, subplot2, sharey=ax_covmat_orig, sharex=ax_covmat_orig)
    # im_predict = ax_covmat_predict.imshow(
    #     pod_covmat_pred,
    #     aspect='equal',
    #     origin='upper',
    #     vmin=vmin,
    #     vmax=vmax
    # )
    # ax_covmat_predict.set_title('Covariance Matrix (Predicted Data)', **title_kwargs)
    # # ax_covmat_predict.tick_params(axis='x', rotation=270+45)
    # # ax_covmat_predict.set_xlabel(xlabel)
    # # ax_covmat_predict.set_ylabel(ylabel)

    # # subplots adjustment to account for colorbars
    # fig_covmat.subplots_adjust(
    #     bottom=0.2,
    #     left=0.1,
    # )

    # # original data and recon data colorbar
    # cb_xbegin = ax_covmat_orig.transData.transform([0, 0])
    # cb_xbegin = fig_covmat.transFigure.inverted().transform(cb_xbegin)[0]
    # cb_xend = ax_covmat_predict.transData.transform([pod_covmat_dataout.shape[-1], 0])
    # cb_xend = fig_covmat.transFigure.inverted().transform(cb_xend)[0]

    # cb_ax = fig_covmat.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
    # cbar = fig_covmat.colorbar(im_predict, cax=cb_ax, orientation='horizontal')

    # # computing the normalized error
    # subplot3 = subplot2+1
    # error = np.abs(pod_covmat_pred-pod_covmat_dataout)
    # vmax_error_snap = 0.8
    # vmax_error = np.max(error)
    # vmax_error = vmax_error_snap*np.round(vmax_error/vmax_error_snap + 0.5)
    # # error = 100*error / np.abs(pod_covmat_dataout)
    # # plotting the normalized error
    # ax_covmat_error = fig_covmat.add_subplot(1, 3, subplot3, sharey=ax_covmat_orig, sharex=ax_covmat_orig)
    # im_error = ax_covmat_error.imshow(
    #     error,
    #     aspect='equal',
    #     origin='upper',
    #     vmin=0.0,
    #     vmax=vmax_error,
    # )
    # ax_covmat_error.set_title(r'Error', **title_kwargs)
    # # ax_error.tick_params(axis='x', rotation=270+45)
    # # ax_error.set_xlabel(xlabel)
    # # ax_error.set_ylabel(ylabel)

    # # error colorbar
    # cbe_xbegin = ax_covmat_error.transData.transform([0, 0])
    # cbe_xbegin = fig_covmat.transFigure.inverted().transform(cbe_xbegin)[0]
    # cbe_xend = ax_covmat_error.transData.transform([pod_covmat_dataout.shape[-1], 0])
    # cbe_xend = fig_covmat.transFigure.inverted().transform(cbe_xend)[0]
    # error_cb_ax = fig_covmat.add_axes([cbe_xbegin, 0.0, cbe_xend-cbe_xbegin, 0.025])
    # cbar_error = fig_covmat.colorbar(im_error, cax=error_cb_ax, orientation='horizontal')
    
    # if savefig_fname is not None:
    #     fig_covmat.savefig(
    #         dir_name_AR_AErnn+'/plots/'+savefig_fname+'--covmat--{}outsteps.pdf'.format(num_outsteps),
    #         dpi=300,
    #         bbox_inches='tight')
    #     fig_covmat.clear()
    #     plt.close()
    # else:
    #     plt.show()
    #     print('')
    

    
    # k = fft.fftfreq(numpoints_xgrid, d=1/numpoints_xgrid)
    # idx = np.where(k<0)[0]
    # k[idx] += numpoints_xgrid
    
    # fig_Fourier, ax_Fourier = plt.subplots()
    # ax_Fourier.semilogy(k, energySpectrum_dataout, linestyle='--', marker='s', linewidth=0.9, markersize=2)
    # ax_Fourier.semilogy(k, energySpectrum_pred, linestyle='--', marker='^', linewidth=0.9, markersize=2)
    # ax_Fourier.grid(True, which='both', axis='y')
    # ax_Fourier.grid(True, which='major', axis='x')
    # ax_Fourier.legend([r'True Data', r'Predicted Data'], **legend_kwargs)
    # ax_Fourier.set_axisbelow(True)
    # ax_Fourier.set_title(r'Squared magnitude of Fourier coefficients', **title_kwargs)
    # ax_Fourier.set_ylabel(r'$\|a_k \|^2 \ / \ \left( \sum_j \|a_j^{(true)} \|^2 \right)$', **ylabel_kwargs)
    # if savefig_fname is not None:
    #     fig_Fourier.savefig(
    #         dir_name_AR_AErnn+'/plots/'+savefig_fname+'--FourierCoeffs--{}outsteps.pdf'.format(num_outsteps),
    #         dpi=300,
    #         bbox_inches='tight')
    #     fig_Fourier.clear()
    #     plt.close()
    # else:
    #     plt.show()
    #     print('')
    
    return

def main(AR_rnn_idx, AR_AERNN_str, AR_RNN_str):
    from tools.misc_tools import create_data_for_RNN, plot_histogram_and_save
    
    from tools.ae_v11 import Autoencoder
    
    # from tools.GRU_AR_v1 import AR_RNN_GRU as AR_RNN
    # from tools.AEGRU_AR_v1 import AR_AERNN_GRU as AR_AERNN
    _temp2 = importlib.__import__('tools.'+AR_RNN_str[0], globals(), locals(), [AR_RNN_str[1],], 0)
    AR_RNN = eval('_temp2.'+AR_RNN_str[1])
    
    _temp3 = importlib.__import__('tools.'+AR_AERNN_str[0], globals(), locals(), [AR_AERNN_str[1],], 0)
    AR_AERNN = eval('_temp3.'+AR_AERNN_str[1])
    
    strategy = None
    # strategy = tf.distribute.MirroredStrategy()

    current_sys = platform.system()

    if current_sys == 'Windows':
        dir_sep = '\\'
    else:
        dir_sep = '/'

    print(os.getcwd())

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

    ### setting up params (and saving, if applicable)
    ### AR AE-RNN directory
    dir_name_AR_AErnn = os.getcwd()+'/saved_AR_AERNN_rnn/AR_rnn_{:03d}'.format(AR_rnn_idx)

    ### reading AR-RNN parameters
    with open(dir_name_AR_AErnn + '/AR_RNN_specific_data.txt') as f:
        lines = f.readlines()

    params_AR_AErnn_dict = eval(''.join(lines))

    dir_name_rnn = params_AR_AErnn_dict['dir_name_rnn']
    # rnn_idx = dir_name_rnn[-3:]
    # dir_name_rnn = os.getcwd()+'/saved_rnn/rnn_'+rnn_idx
    idx1 = dir_name_rnn[::-1].find('/')
    idx2 = dir_name_rnn[:-idx1-1][::-1].find('/')
    dir_name_rnn = os.getcwd() + dir_name_rnn[-idx1-idx2-1-1:]

    dir_name_ae = params_AR_AErnn_dict['dir_name_ae']
    ae_idx = dir_name_ae[-3:]
    dir_name_ae = os.getcwd()+'/saved_ae/ae_'+ae_idx

    dt_rnn = params_AR_AErnn_dict['dt_rnn']
    # T_sample_input = params_AR_AErnn_dict['T_sample_input']
    T_sample_output = params_AR_AErnn_dict['T_sample_output']
    if type(T_sample_output) != type(np.array([])):
        if type(T_sample_output) != type([]):
            T_sample_output = [T_sample_output]
        T_sample_output = np.array(T_sample_output)
    num_outsteps = np.int32(np.round(T_sample_output/dt_rnn))
    # T_offset = params_AR_AErnn_dict['T_offset']
    return_params_arr = False
    params = None
    try:
        # this is the normalization flag for the data fed into the rnn
        normalize_dataset = params_AR_AErnn_dict['normalize_dataset']
    except:
        print("'normalize_dataset' not present in AR_rnn_specific_data, set to False.")
        normalize_dataset = False
    try:
        use_ae_data = params_AR_AErnn_dict['use_ae_data']
    except:
        print("'use_ae_data' not present in AR_rnn_specific_data, set to True.")
        use_ae_data = True

    ### reading RNN normalization constants
    normalization_arr_rnn = None
    # if normalize_dataset == True:
    #     with open(dir_name_AR_AErnn + '/final_net/rnn_normalization.txt') as f:
    #         lines = f.readlines()
    #     normarr_rnn_dict = eval(''.join(lines))
    #     normalization_arr_rnn = normarr_rnn_dict['normalization_arr']

    if os.path.exists(dir_name_AR_AErnn+'/normalization_data.npz'):
        with np.load(dir_name_AR_AErnn+'/normalization_data.npz', allow_pickle=True) as fl:
            normalization_arr_rnn = fl['normalization_arr'][0]

    ### training params
    with open(dir_name_AR_AErnn + dir_sep + 'training_specific_params.txt') as f:
        lines = f.readlines()

    tparams_dict = eval(''.join(lines))

    prng_seed = tparams_dict['prng_seed']
    train_split = tparams_dict['train_split']
    val_split = tparams_dict['val_split']
    batch_size = tparams_dict['batch_size']

    ### reading simulation parameters
    with open(dir_name_ae + dir_sep + 'ae_data.txt') as f:
        lines = f.readlines()
    params_dict = eval(''.join(lines))
    data_dir_idx = params_dict['data_dir_idx']
    normalizeforae_flag = params_dict['normalizeforae_flag']
    try:
        ae_data_with_params = params_dict['ae_data_with_params']
    except:
        print("'ae_data_with_params' not present in ae_data, set to 'True'.")
        ae_data_with_params = True

    if os.path.exists(dir_name_ae+dir_sep+'normalization_data.npz'):
        with np.load(dir_name_ae+dir_sep+'normalization_data.npz', allow_pickle=True) as fl:
            normalization_constant_arr_aedata = fl['normalization_constant_arr_aedata'][0]

    print('dir_name_AR_AErnn:', dir_name_AR_AErnn)
    print('dir_name_rnn:', dir_name_rnn)
    print('dir_name_ae:', dir_name_ae)
    print('data_dir_idx:', data_dir_idx)

    ### loading data
    dir_name_data = os.getcwd() + dir_sep + 'saved_data' + dir_sep + 'data_' + data_dir_idx

    with h5py.File(dir_name_data + '/data.h5', 'r') as f:
        t_recorded_samples = np.array(f['t'])
        
        N = int(0.5*(np.array(f['num_wavenumbers'])-1))
        print(N, type(N))
        N_ref = int(np.array(f['N_ref']))
        
        try:
            u_ref = np.array(f['u_reference'], dtype=FTYPE)
            v_ref = np.array(f['v_reference'], dtype=FTYPE)
        except:
            uh = np.empty(shape=(len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128)
            uh[:, :, N:] = np.array(f['uh'])
            uh[:, 0:N, 0:N] = np.conjugate(uh[:, N+1:, N+1:][:, ::-1, ::-1])
            uh[:, N+1:, 0:N] = np.conjugate(uh[:, 0:N, N+1:][:, ::-1, ::-1])

            vh = np.empty(shape=(len(t_recorded_samples), 2*N+1, 2*N+1), dtype=np.complex128)
            vh[:, :, N:] = np.array(f['vh'])
            vh[:, 0:N, 0:N] = np.conjugate(vh[:, N+1:, N+1:][:, ::-1, ::-1])
            vh[:, N+1:, 0:N] = np.conjugate(vh[:, 0:N, N+1:][:, ::-1, ::-1])

            u_ref = np.fft.irfft2(np.fft.ifftshift(uh), s=(N_ref, N_ref))
            del(uh)
            v_ref = np.fft.irfft2(np.fft.ifftshift(vh), s=(N_ref, N_ref))

    all_data = np.empty(shape=(u_ref.shape[0], 2, u_ref.shape[1], u_ref.shape[2]), dtype=FTYPE)
    all_data[:, 0, :, :] = u_ref
    del(u_ref)
    all_data[:, 1, :, :] = v_ref
    del(v_ref)

    lyapunov_time_arr = [13.06493504]

    delta_t = 1.
    T = t_recorded_samples[-1]

    with np.load(dir_name_data+'/sim_data.npz', 'r') as f:
        delta_t = float(f['dTr'])

    print('all_data.shape : ', all_data.shape)

    # delaing with normalizing the data before feeding into autoencoder
    time_stddev_ogdata = np.std(all_data, axis=0)
    time_mean_ogdata = np.mean(all_data, axis=0)

    # a = 30000
    # all_data = all_data[0:a]
    boundary_idx_arr = [all_data.shape[0]]

    all_data_shape_og = all_data.shape[1:]
    print('all_data.shape : ', all_data.shape)

    test_split = 1 - train_split - val_split

    # setting seed for PRNGs
    np.random.seed(prng_seed)
    tf.random.set_seed(prng_seed)

    # set which data to use for plotting histogram
    data_to_consider = 'testing' # could be 'all', 'testing', 'training', 'val'

    ###--- Create Data ---###

    T_sample_input_cd = np.mean(lyapunov_time_arr)#50.1*dt_rnn
    T_sample_output_cd = 3*np.mean(lyapunov_time_arr)
    T_offset_cd = T_sample_input_cd

    skip_intermediate_cd = 1/4

    all_data = np.reshape(all_data, (all_data.shape[0], -1))

    rnn_res_dict = create_data_for_RNN(
        all_data,
        dt_rnn,
        T_sample_input_cd,
        T_sample_output_cd,
        T_offset_cd,
        None,
        boundary_idx_arr,
        delta_t,
        params=params,
        return_numsamples=True,
        normalize_dataset=normalizeforae_flag,
        stddev_multiplier=3.0,
        skip_intermediate=skip_intermediate_cd,
        return_OrgDataIdxArr=False,
        normalization_arr_external=normalization_constant_arr_aedata.reshape(2, -1),
        normalization_type='stddev')

    data_rnn_input = rnn_res_dict['data_rnn_input']
    data_rnn_output = rnn_res_dict['data_rnn_output']
    org_data_idx_arr_input = rnn_res_dict['org_data_idx_arr_input']
    org_data_idx_arr_output = rnn_res_dict['org_data_idx_arr_output']
    num_samples = rnn_res_dict['num_samples']
    normalization_arr = rnn_res_dict['normalization_arr']
    rnn_data_boundary_idx_arr = rnn_res_dict['rnn_data_boundary_idx_arr']

    temp = np.divide(all_data-normalization_arr[0], normalization_arr[1])
    time_stddev = np.std(temp, axis=0)
    timeMeanofSpaceRMS = np.mean(np.mean(temp**2, axis=1)**0.5)
    del(org_data_idx_arr_input)
    del(org_data_idx_arr_output)
    del(temp)

    all_data = np.reshape(all_data, (all_data.shape[0],)+tuple(all_data_shape_og[-3:]))
    data_rnn_input = np.reshape(data_rnn_input, tuple(data_rnn_input.shape[0:2])+tuple(all_data_shape_og[-3:]))
    data_rnn_output = np.reshape(data_rnn_output, tuple(data_rnn_output.shape[0:2])+tuple(all_data_shape_og[-3:]))

    if data_to_consider != 'all':
        cum_samples = rnn_data_boundary_idx_arr[-1]
        num_train = 0
        num_val = 0
        begin_idx = 0
        for i in range(len(rnn_data_boundary_idx_arr)):
            num_samples = rnn_data_boundary_idx_arr[i] - begin_idx
            num_train += int( np.round(train_split*num_samples) )
            num_val += int( np.round(val_split*num_samples) )
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

        shape_to_use = eval(data_to_consider+'_input_shape')
        rnn_data_idx = np.empty(shape=shape_to_use[0], dtype=np.int32)
        
        begin_idx = 0
        training_data_rolling_count = 0
        val_data_rolling_count = 0
        testing_data_rolling_count = 0
        for i in range(len(rnn_data_boundary_idx_arr)):
            num_samples = rnn_data_boundary_idx_arr[i] - begin_idx
            num_train = int( np.round(train_split*num_samples) )
            num_val = int( np.round(val_split*num_samples) )
            num_test = num_samples-num_train-num_val+1

            if data_to_consider == 'training':
                rnn_data_idx[training_data_rolling_count:training_data_rolling_count+num_train] = np.arange(begin_idx, begin_idx+num_train)
            elif data_to_consider == 'val':
                rnn_data_idx[val_data_rolling_count:val_data_rolling_count+num_val] = np.arange(begin_idx+num_train, begin_idx+num_train+num_val)
            elif data_to_consider == 'testing':
                rnn_data_idx[testing_data_rolling_count:testing_data_rolling_count+num_test] = np.arange(begin_idx+num_train+num_val, rnn_data_boundary_idx_arr[i])

            training_data_rolling_count += num_train
            val_data_rolling_count += num_val
            testing_data_rolling_count += num_test

            begin_idx = rnn_data_boundary_idx_arr[i]

        # shuffling
        np.random.shuffle(rnn_data_idx)
        data_rnn_input = data_rnn_input[rnn_data_idx]
        data_rnn_output = data_rnn_output[rnn_data_idx]
        del(rnn_data_idx)

    print(' data_rnn_input.shape :', data_rnn_input.shape)
    print('data_rnn_output.shape :', data_rnn_output.shape)

    ###--- Prediction Horizon ---###

    for kk in range(num_outsteps.shape[0]+1):    
        total_s_len = 80
        
        if kk == 0:
            num_outsteps_kk = 'ZERO'
            load_file_rnn = dir_name_rnn + '/final_net/final_net_class_dict.txt'
            wt_file_rnn = dir_name_rnn+'/final_net/final_net_gru_weights.h5'

            load_file_ae = dir_name_ae+'/final_net/final_net_class_dict.txt'
            wt_file_ae = dir_name_ae+'/final_net/final_net_ae_weights.h5'
        else:
            num_outsteps_kk = num_outsteps[kk-1]
            load_file_rnn = None
            wt_file_rnn = None

            load_file_ae = None
            wt_file_ae = None
        
        
        sep_lr_s = ' num_outsteps : {} '.format(num_outsteps_kk)
        
        sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'>' + sep_lr_s
        sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'<'
        print('\n\n' + '*'*len(sep_lr_s))
        print('' + sep_lr_s+'')
        print('*'*len(sep_lr_s) + '\n\n')

        prediction_horizons(
            num_outsteps=num_outsteps_kk,
            dir_name_AR_AErnn=dir_name_AR_AErnn,
            Autoencoder=Autoencoder,
            data_rnn_input=data_rnn_input,
            data_rnn_output=data_rnn_output,
            AR_RNN=AR_RNN,
            T_sample_input_rnn=T_sample_input_cd,
            T_sample_output_rnn=T_sample_output_cd,
            AR_AERNN=AR_AERNN,
            normalization_constant_arr_rnn=normalization_arr_rnn,
            normalization_constant_arr_aedata=normalization_constant_arr_aedata,
            time_stddev_ogdata=time_stddev_ogdata,
            time_mean_ogdata=time_mean_ogdata,
            batch_size=1,
            num_runs=100,
            error_threshold=0.5,
            rnn_data_boundary_idx_arr=rnn_data_boundary_idx_arr,
            lyapunov_time_arr=lyapunov_time_arr,
            savefig_fname='post-ARtraining'+'_'+data_to_consider+'data',
            data_to_consider=data_to_consider,
            bin_width=0.05,
            bin_begin=0.0,
            rnn_wt_extension='h5', # 'h5' for tf saved rnns, 'hdf5' for my ESNs
            rnn_load_file=load_file_rnn,
            rnn_wt_file=wt_file_rnn,
            ae_load_file=load_file_ae,
            ae_wt_file=wt_file_ae,
            use_ae_data=use_ae_data,
            plot_histogram_and_save=plot_histogram_and_save,
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('AR_rnn_idx', type=int)
    parser.add_argument('-r', '--arrnn', type=str, nargs='+', default=['GRU_AR_v1', 'AR_RNN_GRU'])
    parser.add_argument('-a', '--araernn', type=str, nargs='+', default=['AEGRU_AR_v1', 'AR_AERNN_GRU'])

    args = parser.parse_args()
    
    print('AR_rnn_idx : {}'.format(args.AR_rnn_idx))
    print('araernn : {}, arrnn : {}'.format(args.araernn, args.arrnn))

    main(args.AR_rnn_idx, args.araernn, args.arrnn)