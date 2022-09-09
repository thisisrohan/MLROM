import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import h5py
import os

############################ Lorenz System Functions ###########################

def RK4_integrator(fX, X0, delta_t, **kwargs):
    '''
    fX = callable function
    X0 = initial condition
    delta_t = time interval
    
    This integrator is specific for an 'fX' that
    doesn't require 't' in its argument.
    '''
    
    k1 = fX(X0, **kwargs)
    k2 = fX(X0 + 0.5 * k1 * delta_t, **kwargs)
    k3 = fX(X0 + 0.5 * k2 * delta_t, **kwargs)
    k4 = fX(X0 + k3 * delta_t, **kwargs)
    X_next = X0 + delta_t * (k1 + 2*k2 + 2*k3 + k4) / 6
    return X_next


def Lorenz_time_der(X, params):
    '''
    X = [[x, y, z], ] - (N, 3) arr
    params = [sigma, rho, beta] - (3,) arr
    '''
    time_der = np.empty_like(X)
    time_der[:, 0] = params[0] * (X[:, 1] - X[:, 0])
    time_der[:, 1] = X[:, 0] * (params[1] - X[:, 2]) - X[:, 1]
    time_der[:, 2] = X[:, 0] * X[:, 1] - params[2] * X[:, 2]
    return time_der


def create_Lorenz_data(
        T, t0, delta_t,
        rho_arr, sigma_arr, beta_arr,
        x0, y0, z0, return_params_arr=False,
        normalize=False):

    N = int(((T-t0) + 0.5*delta_t) // delta_t)
    all_data = np.empty(
        shape=(
            len(rho_arr)*len(sigma_arr)*len(beta_arr)*(N+1),
            6
        ),
        dtype=np.float32
    )

    boundary_idx_arr = np.empty(
        shape=len(rho_arr)*len(sigma_arr)*len(beta_arr),
        dtype=np.int64
    )

    if return_params_arr == True:
        params_arr = np.empty(shape=(boundary_idx_arr.shape[0], 3))
    else:
        params_arr = None
    if normalize == True:
        normalization_constant_arr = np.empty(shape=(boundary_idx_arr.shape[0]))
    else:
        normalization_constant_arr = None

    counter = 0
    for i in range(len(rho_arr)):
        for j in range(len(sigma_arr)):
            for k in range(len(beta_arr)):
                rho = rho_arr[i]
                sigma = sigma_arr[j]
                beta = beta_arr[k]

                # setting up internal vectors and parameters
                params = np.array([sigma, rho, beta])
                if return_params_arr == True:
                    params_arr[counter, :] = params[:]

                X = np.empty(shape=(N+1, 3), dtype=np.float64)
                X[0, :] = [x0, y0, z0]

                # integrating
                kwargs = {'params':params}
                for ii in range(1, N+1):
                    X0 = X[ii-1, :].reshape((1, 3))
                    X_next = RK4_integrator(Lorenz_time_der, X0, delta_t, **kwargs)
                    X[ii, :] = X_next

                # storing data
                idx = counter # = len(beta_arr)*len(sigma_arr)*i + len(beta_arr)*j + k
                all_data[idx*(N+1):(idx+1)*(N+1), 0:3] = X[:, :]
                if normalize == True:
                    normalization_constant = (2*beta*(rho-1) + (rho-1)**2)**0.5
                    normalization_constant_arr[counter] = normalization_constant
                    all_data[idx*(N+1):(idx+1)*(N+1), 0:3] /= normalization_constant
                # for jj in range(idx*(N+1), (idx+1)*(N+1)):
                #     all_data[jj, :] = params
                all_data[idx*(N+1):(idx+1)*(N+1), 3:] = params[:]

                boundary_idx_arr[counter] = (idx+1)*(N+1)
                counter += 1

    res_dict = {
        'all_data':all_data,
        'N':N,
        'boundary_idx_arr':boundary_idx_arr,
        'params_arr':params_arr,
        'normalization_constant_arr':normalization_constant_arr
    }

    return res_dict

################################################################################

def create_data_for_RNN(
        data,
        dt_rnn,
        T_input,
        T_output,
        T_offset,
        N,
        boundary_idx_arr,
        delta_t,
        params=None,
        return_numsamples=False):
    '''
    Creates training/testing data for the RNN.
    `data` : numpy array containing all the data
             dimensions = [N*num_cases, data_dimension]
    `dt_rnn` : delta t at which the RNN operates
    `T_input` : time span of the input
    `T_output` : time span of the output
    `T_offset` : time difference between the input sequence and output sequence
    `N` : total number of timesteps in one case
    `boundary_idx_arr` : array of indices separating the cases in `data`
                         num_cases = len(boundary_idx_arr)
    `delta_t` : delta t of the simulation from which `data` is pulled
    `params` : if not None, this must be a 2D array of dimension [num_cases, num_params]
               listing the parameters of each case. In this case the RNN data will be concatenated
               with their respective parameters.
    '''

    ### VERY IMPORTANT ###
    N += 1
    ######################

    num_sample_input = int((T_input+0.25*dt_rnn) // dt_rnn)
    num_sample_output = int((T_output+0.25*dt_rnn) // dt_rnn)
    idx_offset = int((T_offset+0.25*dt_rnn) // dt_rnn)

    idx_to_skip = int((dt_rnn+0.25*delta_t) // delta_t)
    # num_samples = (int(N // idx_to_skip) - num_sample_output) // num_sample_input # needs correction?
    num_samples = int(N - (idx_offset + num_sample_output - 1)*idx_to_skip)# -1

    if params is not None:
        RNN_data_dim = data.shape[1]+params.shape[1]
    else:
        RNN_data_dim = data.shape[1]
        
    num_cases = len(boundary_idx_arr)

    data_rnn_input = np.empty(shape=(num_cases*num_samples, num_sample_input, RNN_data_dim))
    data_rnn_output = np.empty(shape=(num_cases*num_samples, num_sample_output, RNN_data_dim))

    org_data_idx_arr_input = np.empty(shape=(num_cases*num_samples, num_sample_input), dtype=np.int32)
    org_data_idx_arr_output = np.empty(shape=(num_cases*num_samples, num_sample_output), dtype=np.int32)

    for i in range(len(boundary_idx_arr)):
        for j in range(num_samples):
            data_rnn_input[i*num_samples+j, :, 0:data.shape[1]] = data[i*N+j:i*N+j + idx_to_skip*num_sample_input:idx_to_skip]
            data_rnn_output[i*num_samples+j, :, 0:data.shape[1]] = data[i*N+j + idx_to_skip*idx_offset:i*N+j + idx_to_skip*(idx_offset+num_sample_output):idx_to_skip]
            
            if params is not None:
                for k in range(params.shape[1]):
                    data_rnn_input[i*num_samples+j, :, data.shape[1]+k] = params[i, k]
                    data_rnn_output[i*num_samples+j, :, data.shape[1]+k] = params[i, k]

            org_data_idx_arr_input[i*num_samples+j, :] = np.arange(i*N+j, i*N+j + idx_to_skip*num_sample_input, idx_to_skip)
            org_data_idx_arr_output[i*num_samples+j, :] = np.arange(i*N+j + idx_to_skip*idx_offset, i*N+j + idx_to_skip*(idx_offset+num_sample_output), idx_to_skip)
            
    if return_numsamples is True:
        return data_rnn_input, data_rnn_output, org_data_idx_arr_input, org_data_idx_arr_output, num_samples
    else:
        return data_rnn_input, data_rnn_output, org_data_idx_arr_input, org_data_idx_arr_output

################################################################################

def plot_latent_states(
        boundary_idx_arr,
        latent_states_all,
        all_data,
        xlim=None,
        ylim=None,
        max_rows=10,
        legend_bbox_to_anchor=[1, 1],
        legend_loc='upper left',
        legend_markerscale=10,
        markersize=1,
        cmap_name='viridis'):

    # plotting latent states
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, len(boundary_idx_arr))]
    
    prev_idx = 0
    for i in range(len(boundary_idx_arr)):
        next_idx = boundary_idx_arr[i]
        ax.scatter(
            latent_states_all[prev_idx:next_idx, 0],
            latent_states_all[prev_idx:next_idx, 1],
            s=markersize,
            # linewidth=0,
            marker='.',
            color=colors[i],
            # markersize=1,
            label=r'[$\sigma$, $\rho$, $\beta$] = ' + np.array2string(all_data[next_idx-1, 3:], precision=2, separator=', ')
        )
        prev_idx = next_idx

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    # max_rows = 10
    max_rows = float(max_rows)
    ncols = int(np.ceil(len(boundary_idx_arr) / max_rows))
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=ncols, markerscale=legend_markerscale)
    # plt.savefig(dir_name_ae+'/plots/latent_space.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return fig, ax

################################################################################

def plot_losses(
        training_loss,
        val_loss,
        lr_change,
        learning_rate_list=None,
        lr_y_placement_axes_coord=0.5,
        lr_x_offset_axes_coord=0.06):

    epoch_count = range(1, len(training_loss) + 1)

    fig, ax = plt.subplots()

    ax.semilogy(epoch_count, training_loss, 'r--')
    ax.semilogy(epoch_count, val_loss, 'b-', linewidth=0.8)
    ax.legend(['Training Loss', 'Validation Loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.set_axisbelow(True)

    if learning_rate_list is not None:
        for i in range(len(lr_change)-1):
            ax.axvline(lr_change[i], color='m', linestyle='-.', linewidth=0.8)
            ax.text(
                lr_change[i]+ax.transData.inverted().transform(ax.transAxes.transform([lr_x_offset_axes_coord, 0.]))[0],
                ax.transData.inverted().transform(ax.transAxes.transform([0., lr_y_placement_axes_coord]))[1],
                'lr={}'.format(learning_rate_list[i]),
                rotation=90,
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(facecolor='yellow', alpha=0.25, boxstyle='square,pad=0.2')
            )

    return fig, ax

################################################################################

def plot_reconstructed_data(
        boundary_idx_arr,
        dir_name_ae,
        all_data,
        reconstructed_data,
        save_figs=False):

    # saving reconstructed data
    n = len(boundary_idx_arr)
    num_cols = 2
    if save_figs == True:
        num_rows = 1
    else:
        num_rows = n

    if save_figs == True:
        recon_data_dir = dir_name_ae+'/plots/reconstructed_data'
        if not os.path.isdir(recon_data_dir):
            os.makedirs(recon_data_dir)
    else:
        fig = plt.figure(figsize=(7.5*num_cols, 7.5*num_rows))
    
    num_digits_n = int(np.log10(n)+1)
    
    prev_idx = 0
    for i in range(n):
        if save_figs == True:
            fig = plt.figure(figsize=(7.5*num_cols, 7.5*num_rows))
            subplot1 = 1
        else:
            subplot1 = 2*i+1
        subplot2 = subplot1 + 1

        next_idx = boundary_idx_arr[i]

        ax_orig = fig.add_subplot(num_rows, num_cols, subplot1, projection ='3d')
        ax_orig.plot(all_data[prev_idx:next_idx, 0], all_data[prev_idx:next_idx, 1], all_data[prev_idx:next_idx, 2])
        ax_orig.title.set_text(r'Actual Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(all_data[next_idx-1, 3:], precision=2, separator=', '))
        ax_orig.set_xlabel('x')
        ax_orig.set_ylabel('y')
        ax_orig.set_zlabel('z')
        
        ax_predict = fig.add_subplot(num_rows, num_cols, subplot2, projection ='3d')
        ax_predict.plot(reconstructed_data[prev_idx:next_idx, 0], reconstructed_data[prev_idx:next_idx, 1], reconstructed_data[prev_idx:next_idx, 2])
        ax_predict.title.set_text(r'NN Reconstructed Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(all_data[next_idx-1, 3:], precision=2, separator=', ')        )
        ax_predict.set_xlabel('x')
        ax_predict.set_ylabel('y')
        ax_predict.set_zlabel('z')
    
        prev_idx = next_idx

        if save_figs == True:
            fig.savefig(recon_data_dir+'/reconstructed_'+str(i+1).zfill(num_digits_n)+'.png', dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close()
        
    if save_figs == True:
        return
    else:
        return fig

################################################################################

def readAndReturnLossHistories(
        dir_name_ae,
        dir_sep,
        epochs,
        learning_rate_list,
        return_earlystopping_wait=False):

    with h5py.File(dir_name_ae+'{ds}checkpoints{ds}LossHistoriesCheckpoint.hdf5'.format(ds=dir_sep), 'r') as f:
        val_loss_arr_fromckpt = np.array(f['val_loss_arr'])
        train_loss_arr_fromckpt = np.array(f['train_loss_arr'])
    
    val_loss_hist = []
    train_loss_hist = []
    lr_change=[0]
    
    num_epochs_left = epochs
    starting_lr_idx = 0
    for i in range(len(learning_rate_list)):
        nan_flags = np.isnan(val_loss_arr_fromckpt[epochs*i:epochs*(i+1)])
        if nan_flags[0] == True:
            # this and all further learning rates were not reached in previous training session
            break
        else:
            temp_ = np.where(nan_flags == False)[0]
            num_epochs_left = epochs - len(temp_)
            starting_lr_idx = i
    
            val_loss_hist.extend(val_loss_arr_fromckpt[epochs*i:epochs*(i+1)][temp_])
            train_loss_hist.extend(train_loss_arr_fromckpt[epochs*i:epochs*(i+1)][temp_])
            lr_change.append(lr_change[i]+len(temp_))

    if return_earlystopping_wait == True:
        min_val_loss = np.min(val_loss_hist)
        idx = np.where(val_loss_hist == min_val_loss)[0][-1]
        earlystopping_wait = len(val_loss_hist) - idx - 1
        return val_loss_hist, train_loss_hist, lr_change, starting_lr_idx, num_epochs_left, val_loss_arr_fromckpt, train_loss_arr_fromckpt, earlystopping_wait
    else:
        return val_loss_hist, train_loss_hist, lr_change, starting_lr_idx, num_epochs_left, val_loss_arr_fromckpt, train_loss_arr_fromckpt

################################################################################

# time callback for each epoch
class mytimecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.total_time = 0
        self.start_time = time.time()
    
    def on_epoch_end(self,epoch,logs = {}):
        self.total_time = time.time() - self.start_time
        print(' - tot_time: {:.0f}h {:.0f}m {:.1f}s'.format(self.total_time//3600, (self.total_time//60)%60, self.total_time%60))



class SaveLosses(tf.keras.callbacks.Callback):
    def __init__(self, filepath, total_epochs, val_loss_arr=None, train_loss_arr=None, lr_idx=0, period=1):
        if val_loss_arr is not None:
            self.val_loss_arr = val_loss_arr
        else:
            self.val_loss_arr = np.NaN*np.ones(shape=total_epochs)
        if train_loss_arr is not None:
            self.train_loss_arr = train_loss_arr
        else:
            self.train_loss_arr = np.NaN*np.ones(shape=total_epochs)
        self.filepath = filepath
        self.total_epochs = total_epochs
        self.lr_idx = lr_idx
        self.period = period
        self.offset = 0
    
    def on_epoch_end(self, epoch, logs = {}):
        self.val_loss_arr[self.total_epochs*self.lr_idx+epoch+self.offset] = logs['val_loss']
        self.train_loss_arr[self.total_epochs*self.lr_idx+epoch+self.offset] = logs['loss']
        if (epoch+1) % self.period == 0:
            with h5py.File(self.filepath+".hdf5", "w") as f:
                dset1 = f.create_dataset('val_loss_arr', data=self.val_loss_arr)
                dset2 = f.create_dataset('train_loss_arr', data=self.train_loss_arr)
                dset3 = f.create_dataset('total_epochs', data=self.total_epochs)
            print(' - saving loss histories at '+self.filepath)

    def update_lr_idx(self, lr_idx):
        self.lr_idx = lr_idx

    def update_offset(self, offset):
        self.offset = offset
        

def sigmoidWarmupAndDecay(
        epoch, eta_begin=0.0001, eta_high=0.001, eta_low=0.00001,
        warmup=20, expected_epochs=200, g_star=0.999, f_star=0.001):

    sqrt_eta_high = eta_high**0.5
    # warmup lr
    g = 1/g_star - 1
    g = g**(2*epoch/warmup - 1)
    g = eta_begin + (eta_high - eta_begin)/(1 + g)
    g /= sqrt_eta_high

    # decay_lr
    f = 1/f_star - 1
    f = f**(2*(epoch-warmup)/expected_epochs - 1)
    f = eta_low + (eta_high - eta_low)/(1 + f)
    f /= sqrt_eta_high

    final_lr = f*g

    return final_lr


class sigmoidWarmupAndDecayLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, eta_begin=0.0001, eta_high=0.001, eta_low=0.00001,
            warmup=20, expected_epochs=200, g_star=0.999, f_star=0.001):
        
        self.eta_begin = eta_begin
        self.eta_high = eta_high
        self.eta_low = eta_low
        self.warmup = warmup
        self.expected_epochs = expected_epochs
        self.g_star = g_star
        self.f_star = f_star
        self.offset = 0

    def on_epoch_end(self, epoch, logs = {}):
        epoch = epoch + self.offset
        sqrt_eta_high = self.eta_high**0.5
        # warmup lr
        g = 1/self.g_star - 1
        g = g**(2*epoch/self.warmup - 1)
        g = self.eta_begin + (self.eta_high - self.eta_begin)/(1 + g)
        g /= sqrt_eta_high
    
        # decay_lr
        f = 1/self.f_star - 1
        f = f**(2*(epoch-self.warmup)/self.expected_epochs - 1)
        f = self.eta_low + (self.eta_high - self.eta_low)/(1 + f)
        f /= sqrt_eta_high
    
        scheduled_lr = f*g
        
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

    def update_offset(self, offset):
        self.offset = offset
################################################################################