import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import h5py
import os
from scipy.fft import fft, ifft, fftfreq
import scipy.linalg as splnalg

FTYPE = np.float32
ITYPE = np.int32

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
    
def CDV_time_der(X, params):
    '''
    X = [[x1, x2, x3, x4, x5, x6], ] - (N, 6) arr
    params = [x1*, x4*, C, beta, gamma, b] - (6,) arr
    '''
    time_der = np.empty_like(X)
    
    x1star = params[0]
    x4star = params[1]
    C = params[2]
    beta = params[3]
    gamma = params[4]
    b = params[5]
    
    rt2 = 2**0.5
    epsilon = 16*rt2/(5*np.pi)
    
    alpha_fn = lambda m : 8*rt2*m*m*(b*b+m*m-1)/(np.pi*(4*m*m-1)*(b*b+m*m))
    beta_fn = lambda m : beta*b*b/(b*b+m*m)
    delta_fn = lambda m : 64*rt2*(b*b-m*m+1)/(15*np.pi*(b*b+m*m))
    gammastar_fn = lambda m : gamma*4*rt2*m*b/(np.pi*(4*m*m-1))
    gamma_fn = lambda m : gamma*4*rt2*m*m*m*b/(np.pi*(4*m*m-1)*(b*b+m*m))

    alpha_1 = alpha_fn(1)
    alpha_2 = alpha_fn(2)
    beta_1 = beta_fn(1)
    beta_2 = beta_fn(2)
    delta_1 = delta_fn(1)
    delta_2 = delta_fn(2)
    gammastar_1 = gammastar_fn(1)
    gammastar_2 = gammastar_fn(2)
    gamma_1 = gamma_fn(1)
    gamma_2 = gamma_fn(2)

    time_der[:, 0] = gammastar_1*X[:, 2] - C*(X[:, 0] - x1star)
    time_der[:, 1] = -(alpha_1*X[:, 0]-beta_1)*X[:, 2] - C*X[:, 1] - delta_1*X[:, 3]*X[:, 5]
    time_der[:, 2] = (alpha_1*X[:, 0]-beta_1)*X[:, 1] - gamma_1*X[:, 0] - C*X[:, 2] + delta_1*X[:, 3]*X[:, 4]
    time_der[:, 3] = gammastar_2*X[:, 5] - C*(X[:, 3] - x4star) + epsilon*(X[:, 1]*X[:, 5] - X[:, 2]*X[:, 4])
    time_der[:, 4] = -(alpha_2*X[:, 0]-beta_2)*X[:, 5] - C*X[:, 4] - delta_2*X[:, 3]*X[:, 2]
    time_der[:, 5] = (alpha_2*X[:, 0]-beta_2)*X[:, 4] - gamma_2*X[:, 3] - C*X[:, 5] + delta_2*X[:, 3]*X[:, 1]
    
    return time_der


def create_Lorenz_data(
        T, t0, delta_t,
        rho_arr, sigma_arr, beta_arr,
        x0, y0, z0, return_params_arr=False,
        normalize=False, FTYPE=FTYPE, ITYPE=ITYPE):

    N = int(((T-t0) + 0.5*delta_t) // delta_t)
    all_data = np.empty(
        shape=(
            len(rho_arr)*len(sigma_arr)*len(beta_arr)*(N+1),
            6
        ),
        dtype=FTYPE
    )

    boundary_idx_arr = np.empty(
        shape=len(rho_arr)*len(sigma_arr)*len(beta_arr),
        dtype=ITYPE
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

                X = np.empty(shape=(N+1, 3), dtype=FTYPE)
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


def create_CDV_data(
        T, t0, delta_t,
        params_mat,
        init_state_mat, return_params_arr=False,
        normalize=False,
        stddev_multiplier_for_norm=None, FTYPE=FTYPE, ITYPE=ITYPE):

    if len(params_mat.shape) == 1:
        params_mat = params_mat.reshape((1, params_mat.shape[0]))
    N = int(((T-t0) + 0.5*delta_t) // delta_t)
    all_data = np.empty(
        shape=(
            params_mat.shape[0]*(N+1),
            6+6
        ),
        dtype=FTYPE
    )

    boundary_idx_arr = np.empty(
        shape=params_mat.shape[0],
        dtype=ITYPE
    )

    if return_params_arr == True:
        params_arr = np.empty(shape=(boundary_idx_arr.shape[0], 6), dtype=FTYPE)
    else:
        params_arr = None

    counter = 0
    for ii in range(params_mat.shape[0]):
        # setting up internal vectors and parameters
        params = params_mat[ii, :]
        if return_params_arr == True:
            params_arr[counter, :] = params[:]

        X = np.empty(shape=(N+1, 6), dtype=FTYPE)
        X[0, :] = init_state

        # integrating
        kwargs = {'params':params}
        for jj in range(1, N+1):
            X0 = X[jj-1, :].reshape((1, 6))
            X_next = RK4_integrator(CDV_time_der, X0, delta_t, **kwargs)
            X[jj, :] = X_next

        # storing data
        idx = counter # = len(beta_arr)*len(sigma_arr)*i + len(beta_arr)*j + k
        all_data[idx*(N+1):(idx+1)*(N+1), 0:6] = X[:, :]
        # if normalize == True:
        #     normalization_constant = (2*beta*(rho-1) + (rho-1)**2)**0.5
        #     normalization_constant_arr[counter] = normalization_constant
        #     all_data[idx*(N+1):(idx+1)*(N+1), 0:3] /= normalization_constant
        # for jj in range(idx*(N+1), (idx+1)*(N+1)):
        #     all_data[jj, :] = params
        all_data[idx*(N+1):(idx+1)*(N+1), 6:] = params[:]

        boundary_idx_arr[counter] = (idx+1)*(N+1)
        counter += 1

    normalization_constant_arr = None
    if normalize == True:
        normalization_constant_arr = np.empty(shape=(2, 6), dtype=FTYPE)
        if stddev_multiplier_for_norm is None:
            stddev_multiplier_for_norm = 1.414213
        for i in range(6):
            sample_mean = np.mean(all_data[:, i])
            sample_std = np.std(all_data[:, i])
            all_data[:, i] = (all_data[:, i] - sample_mean)/(stddev_multiplier_for_norm*sample_std)
            normalization_constant_arr[0, i] = sample_mean
            normalization_constant_arr[1, i] = stddev_multiplier_for_norm*sample_std

    res_dict = {
        'all_data':all_data,
        'N':N,
        'boundary_idx_arr':boundary_idx_arr,
        'params_arr':params_arr,
        'normalization_constant_arr':normalization_constant_arr
    }

    return res_dict


def create_KS_data(
        T, t0, delta_t, xgrid,
        init_state_mat, params_mat=np.array([[1, 1, 1]]),
        return_params_arr=False,
        normalize=False, M_Cauchy=32, alldata_withparams=False,
        stddev_multiplier_for_norm=None, FTYPE=FTYPE, ITYPE=ITYPE):
    '''
    simulating the KS equation
    u_t = -nu1*u*u_x - nu2*u_xx - nu3*u_xxxx
    params_mat[0] = nu1
    params_mat[1] = nu2
    params_mat[2] = nu3
    '''
    
    if len(params_mat.shape) == 1:
        params_mat = params_mat.reshape((1, params_mat.shape[0]))
        if len(init_state_mat.shape) == 1:
            init_state_mat = init_state_mat.reshape((1, init_state_mat.shape[0]))
    else:
        if len(init_state_mat.shape) == 1:
            init_state_mat = np.tile(init_state_mat, (params_mat.shape[0], 1))

    N = int(((T-t0) + 0.5*delta_t) // delta_t)
    num_modes = init_state_mat.shape[1]
    num_params = params_mat.shape[1]
    if alldata_withparams == True:
        all_data = np.empty(
            shape=(
                params_mat.shape[0]*(N+1),
                num_modes + num_params,
            ),
            dtype=FTYPE
        )
    else:
        all_data = np.empty(
            shape=(
                params_mat.shape[0]*(N+1),
                num_modes,
            ),
            dtype=FTYPE
        )

    boundary_idx_arr = np.empty(
        shape=params_mat.shape[0],
        dtype=ITYPE
    )

    length = xgrid[-1]
    M = xgrid.shape[0]

    params_arr = None
    if return_params_arr == True:
        params_arr = params_mat.copy()
    
    # scalars for ETDRK4
    h = delta_t
    k = fftfreq(M) * M * 2*np.pi/length

    for ii in range(params_mat.shape[0]):
        L = params_mat[ii, 1]*(k**2) - params_mat[ii, 2]*(k**4)
        E = np.exp(h*L)
        E_2 = np.exp(h*L/2)
        Q = 0
        phi1 = 0.0
        phi2 = 0.0
        phi3 = 0.0
    
        for j in range(1,M_Cauchy+1):
            arg = h*L + np.ones(L.shape[0]) * np.exp(2j*np.pi*(j-0.5)/M_Cauchy)
    
            phi1 += 1.0/arg * (np.exp(arg) - np.ones(L.shape[0]))
            phi2 += 1.0/arg**2 * (np.exp(arg) - arg - np.ones(L.shape[0]))
            phi3 += 1.0/arg**3 * (np.exp(arg) - 0.5*arg**2 - arg - np.ones(L.shape[0]))
            Q += 2.0/arg * (np.exp(0.5*arg) - np.ones(L.shape[0]))
    
        phi1 = np.real(phi1/M_Cauchy)
        phi2 = np.real(phi2/M_Cauchy)
        phi3 = np.real(phi3/M_Cauchy)
        Q = np.real(Q/M_Cauchy)
    
        f1 = phi1 - 3*phi2 + 4*phi3 #-4 - L * h + E * (4 - 3 * L * h + (L * h)*(L * h))
        f2 = 2*phi2 - 4*phi3 #2 + L * h + E * (-2 + L * h)
        f3 = -phi2 + 4*phi3 #-4 - 3 * L * h - (L*h)*(L*h) + E * (4 - L*h)
    
        # main loop
        init_state = init_state_mat[ii, :]
        v = fft(init_state)
        all_data[ii*(N+1) + 0, 0:num_modes] = init_state[:]
        if alldata_withparams == True:
            all_data[ii*(N+1):(ii+1)*(N+1), num_modes:] = params_mat[ii, :]
        for i in range(1, N+1):
    
            Nv = -params_mat[ii, 0]*0.5j*k * fft(np.real(ifft(v))**2)
            a = E_2 * v + h/2 * Q * Nv
            Na = -params_mat[ii, 0]*0.5j*k * fft(np.real(ifft(a))**2)
            b = E_2 * v + h/2 * Q * Na
            Nb = -params_mat[ii, 0]*0.5j*k * fft(np.real(ifft(b))**2)
            c = E_2 * a + h/2 * Q * (2 * Nb - Nv)
            Nc = -params_mat[ii, 0]*0.5j*k * fft(np.real(ifft(c))**2)
            #update rule
            v = E * v + h*f1*Nv + h*f2*(Na+Nb) + h*f3*Nc
    
            #save data
            all_data[ii*(N+1) + i, 0:num_modes] = np.real(ifft(v))
            
        boundary_idx_arr[ii] = (ii+1)*(N+1)
    

    normalization_constant_arr = None
    if normalize == True:
        normalization_constant_arr = np.empty(shape=(2, all_data.shape[1]), dtype=FTYPE)
        if stddev_multiplier_for_norm is None:
            stddev_multiplier_for_norm = 1.414213
        for i in range(num_modes):
            sample_mean = np.mean(all_data[:, i])
            sample_std = np.std(all_data[:, i])
            all_data[:, i] = (all_data[:, i] - sample_mean)/(stddev_multiplier_for_norm*sample_std)
            normalization_constant_arr[0, i] = sample_mean
            normalization_constant_arr[1, i] = stddev_multiplier_for_norm*sample_std

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
        return_numsamples=False,
        normalize_dataset=False,
        normalization_arr_external=None,
        stddev_multiplier=None,
        FTYPE=FTYPE, ITYPE=ITYPE,
        skip_intermediate=1,
        return_OrgDataIdxArr=True):
    '''
    Creates training/testing data for the RNN.
    `data` : numpy array containing all the data
             dimensions = [N*num_cases, data_dimension]
    `dt_rnn` : delta t at which the RNN operates
    `T_input` : time span of the input
    `T_output` : time span of the output
    `T_offset` : time difference between the input sequence and output sequence
    `N` : total number of timesteps in one case [obsolete, only here for backwards compatibility]
    `boundary_idx_arr` : array of indices separating the cases in `data`
                         num_cases = len(boundary_idx_arr)
    `delta_t` : delta t of the simulation from which `data` is pulled
    `params` : if not None, this must be a 2D array of dimension [num_cases, num_params]
               listing the parameters of each case. In this case the RNN data will be concatenated
               with their respective parameters.
    '''

    # ### VERY IMPORTANT ###
    # N += 1
    # ######################

    num_sample_input = int((T_input+0.5*dt_rnn) // dt_rnn)
    num_sample_output = int((T_output+0.5*dt_rnn) // dt_rnn)
    idx_offset = int((T_offset+0.5*dt_rnn) // dt_rnn)

    idx_to_skip = int((dt_rnn+0.5*delta_t) // delta_t)
    # num_samples = (int(N // idx_to_skip) - num_sample_output) // num_sample_input # needs correction?
#     num_samples = int(N - (idx_offset + num_sample_output - 1)*idx_to_skip)# -1

    if skip_intermediate == 'full sample':
        skip_intermediate = (num_sample_output+idx_offset)*idx_to_skip
    elif skip_intermediate < 1.0:
        skip_intermediate = int((num_sample_output+idx_offset)*idx_to_skip*skip_intermediate)
    else:
        skip_intermediate = int(skip_intermediate)


    if params is not None:
        RNN_data_dim = data.shape[1]+params.shape[1]
    else:
        RNN_data_dim = data.shape[1]
        
    num_cases = len(boundary_idx_arr)

    begin_idx = 0
    total_num_samples = 0
    s = (idx_offset + num_sample_output - 1)*idx_to_skip + 1
    for i in range(len(boundary_idx_arr)):
        N = boundary_idx_arr[i] - begin_idx
        # if skip_intermediate == False:
        #     s = (idx_offset + num_sample_output - 1)*idx_to_skip + 1
        #     total_num_samples += int(N - s)
        # else:
        #     s = (idx_offset + num_sample_output -0)*idx_to_skip
        #     total_num_samples += int(N//s)
        total_num_samples += int((N-s-1)//skip_intermediate + 1)
        begin_idx = boundary_idx_arr[i]

    data_rnn_input = np.empty(shape=(total_num_samples, num_sample_input, RNN_data_dim), dtype=FTYPE)
    data_rnn_output = np.empty(shape=(total_num_samples, num_sample_output, RNN_data_dim), dtype=FTYPE)

    org_data_idx_arr_input = None
    org_data_idx_arr_output = None
    if return_OrgDataIdxArr == True:
        org_data_idx_arr_input = np.empty(shape=(total_num_samples, num_sample_input), dtype=ITYPE)
        org_data_idx_arr_output = np.empty(shape=(total_num_samples, num_sample_output), dtype=ITYPE)

    begin_idx = 0
    cum_samples = 0
    rnn_data_boundary_idx_arr = np.empty_like(boundary_idx_arr)
    
    # skip = 1
    # if skip_intermediate == True:
    #     skip = (idx_offset + num_sample_output - 0)*idx_to_skip
    skip = skip_intermediate
    for i in range(len(boundary_idx_arr)):
        N = boundary_idx_arr[i] - begin_idx
        # if skip_intermediate == False:
        #     num_samples = int(N - (idx_offset + num_sample_output - 0)*idx_to_skip)
        # else:
        #     num_samples = int(N//((idx_offset + num_sample_output - 0)*idx_to_skip))
        num_samples = int((N-s-1)//skip_intermediate + 1)
        for j in range(0, num_samples):
#             data_rnn_input[i*num_samples+j, :, 0:data.shape[1]] = data[i*N+j:i*N+j + idx_to_skip*num_sample_input:idx_to_skip]
#             data_rnn_output[i*num_samples+j, :, 0:data.shape[1]] = data[i*N+j + idx_to_skip*idx_offset:i*N+j + idx_to_skip*(idx_offset+num_sample_output):idx_to_skip]
            data_rnn_input[cum_samples+j, :, 0:data.shape[1]] = data[begin_idx+j*skip:begin_idx+j*skip + idx_to_skip*num_sample_input:idx_to_skip]
            data_rnn_output[cum_samples+j, :, 0:data.shape[1]] = data[begin_idx+j*skip + idx_to_skip*idx_offset:begin_idx+j*skip + idx_to_skip*(idx_offset+num_sample_output):idx_to_skip]

            if params is not None:
                for k in range(params.shape[1]):
                    data_rnn_input[cum_samples+j, :, data.shape[1]+k] = params[i, k]
                    data_rnn_output[cum_samples+j, :, data.shape[1]+k] = params[i, k]

            if return_OrgDataIdxArr == True:
                org_data_idx_arr_input[cum_samples+j, :] = np.arange(begin_idx+j*skip, begin_idx+j*skip + idx_to_skip*num_sample_input, idx_to_skip)
                org_data_idx_arr_output[cum_samples+j, :] = np.arange(begin_idx+j*skip + idx_to_skip*idx_offset, begin_idx+j*skip + idx_to_skip*(idx_offset+num_sample_output), idx_to_skip)
        cum_samples += num_samples
        rnn_data_boundary_idx_arr[i] = cum_samples
        begin_idx = boundary_idx_arr[i]

    normalization_arr = None
    if normalize_dataset == True:
        if normalization_arr_external is None:
            if stddev_multiplier is None:
                stddev_multiplier = 1.414213
            normalization_arr = np.empty(shape=(2, data.shape[1]), dtype=FTYPE)
            for i in range(data.shape[1]):
                sample_mean = np.mean(data[:, i])
                sample_std = np.std(data[:, i])
                normalization_arr[0, i] = sample_mean
                normalization_arr[1, i] = stddev_multiplier*sample_std
        else:
            normalization_arr = normalization_arr_external.copy()
        for i in range(data.shape[1]):
            data_rnn_input[:, :, i] -= normalization_arr[0, i]
            data_rnn_input[:, :, i] /= normalization_arr[1, i]
            data_rnn_output[:, :, i] -= normalization_arr[0, i]
            data_rnn_output[:, :, i] /= normalization_arr[1, i]
            

    # if return_numsamples is True:
    #     return data_rnn_input, data_rnn_output, org_data_idx_arr_input, org_data_idx_arr_output, num_samples
    # else:
    #     return data_rnn_input, data_rnn_output, org_data_idx_arr_input, org_data_idx_arr_output
    res_dict = {
        'data_rnn_input':data_rnn_input,
        'data_rnn_output':data_rnn_output,
        'org_data_idx_arr_input':org_data_idx_arr_input,
        'org_data_idx_arr_output':org_data_idx_arr_output,
        'num_samples':num_samples,
        'normalization_arr':normalization_arr,
        'rnn_data_boundary_idx_arr':rnn_data_boundary_idx_arr
    }
    return res_dict
    

################################################################################

def plot_latent_states_cdv(
        boundary_idx_arr,
        latent_states_all,
        all_data,
        delta_t,
        params_mat,
        xlim=None,
        ylim=None,
        max_rows=10,
        legend_bbox_to_anchor=[1.1,0.85],
        legend_loc='upper left',
        legend_markerscale=10,
        markersize=1,
        cmap_name='viridis',
        save_config_path=None):

    n = len(boundary_idx_arr)
    num_latent_states = latent_states_all.shape[1]
    N = latent_states_all.shape[0]//n - 1

    num_cols = 1
    num_rows = n*num_latent_states
    
    ax_ylabels = ['$x^*_1$', '$x^*_2$', '$x^*_3$', '$x^*_4$', '$x^*_5$', '$x^*_6$']

    fig, ax = plt.subplots(num_latent_states, 1, sharex=True, figsize=(7.5*num_cols, 2.5*num_rows))
    if num_latent_states == 1:
        ax = [ax]
    input_time = np.arange(0, N+1)*delta_t

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, len(boundary_idx_arr))]

    prev_idx = 0
    
    mpl_ax_artist_list = []
    for j in range(num_latent_states):
        for i in range(len(boundary_idx_arr)):
            obj = ax[j].plot(input_time, latent_states_all[i*(N+1):(i+1)*(N+1), j], linewidth=0.8, color=colors[i], label='Case {}'.format(i+1))
            mpl_ax_artist_list.append(obj[0])
        ax[j].set_ylabel(ax_ylabels[j])
        if xlim is not None:
            ax[j].set_xlim(xlim)
        if ylim is not None:
            ax[j].set_ylim(ylim)
        ax[j].grid(True)
        ax[j].set_axisbelow(True)


    ax[-1].set_xlabel('Time')

    max_rows = 10
    max_rows = float(max_rows)
    ncols = int(np.ceil(len(boundary_idx_arr) / max_rows))
    # plt.figlegend(
    #     handles=mpl_ax_artist_list[0:n],
    #     # bbox_to_anchor=[1.1,0.85],
    #     loc=legend_loc,
    #     bbox_to_anchor=legend_bbox_to_anchor,
    #     ncol=ncols,
    #     markerscale=legend_markerscale
    # )
    ax[0].legend(
        # handles=mpl_ax_artist_list[0:n],
        # bbox_to_anchor=[1.1,0.85],
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=ncols,
        markerscale=legend_markerscale
    )
    # fig.suptitle(r'Latent States', size=12)
    ax[0].set_title(r'Latent States', size=12)

    if save_config_path != None:
        import csv
        header = ['x1star', 'x4star', 'C', 'beta', 'gamma', 'b']
        with open(save_config_path+'/latent_states_legend.csv', 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(header)
            for row in params_mat:
                writer.writerow(row)

    return fig, ax


def plot_latent_states_KS(
        boundary_idx_arr,
        latent_states_all,
        delta_t,
        dir_name_ae,
        xticks_snapto=20,
        num_yticks=11,
        xlim=None,
        ylim=None,
        max_rows=10,
        legend_bbox_to_anchor=[1, 1],
        legend_loc='upper left',
        legend_markerscale=10,
        markersize=1,
        cmap_name='viridis',
        save_figs=False,
        factor=1):

    # plotting latent states
    if save_figs == True:
        ls_dir = dir_name_ae+'/plots/latent_states'
        if not os.path.isdir(ls_dir):
            os.makedirs(ls_dir)

    n = len(boundary_idx_arr)
    num_cols = 1
    num_rows = 1
    
    num_digits_n = int(np.log10(n)+1)
    num_latent_states = latent_states_all.shape[1]

    prev_idx = 0
    for i in range(len(boundary_idx_arr)):
        next_idx = boundary_idx_arr[i]
        fig, ax = plt.subplots(figsize=(factor*7.5*num_cols, factor*5.0*num_rows))
        N = int(next_idx-prev_idx)
        input_time = np.arange(0, N)*delta_t
        im = ax.imshow(latent_states_all[prev_idx:next_idx, :].transpose(), aspect='auto', origin='lower')
        num_xticks = 1 + int((N*delta_t + 0.5*xticks_snapto) // xticks_snapto)
        # xticks = np.linspace(0, N, num_xticks, dtype=ITYPE)
        xticks = np.arange(0, N, int((xticks_snapto+0.5*delta_t)//delta_t))
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(np.round(xticks*delta_t, 1))
        ax.tick_params(axis='x', rotation=270+45)
    
        yticks = np.linspace(0, num_latent_states-1, num_yticks, dtype=ITYPE)
        yticklabels = yticks+1
    
        ax.set_yticks(ticks=yticks)
        ax.set_yticklabels(yticklabels)
    
        ax.set_xlabel('Time')
        ax.set_ylabel(r'Latent State Index')
        ax.title.set_text(r'Latent States')
    
        plt.colorbar(im)
        
        prev_idx = next_idx
        
        # saving the figures
        if save_figs == True:
            fig.savefig(ls_dir+'/latent_states_'+str(i+1).zfill(num_digits_n)+'.png', dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close()
        else:
            plt.show()
            print('')

    return


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


def plot_reconstructed_data_KS(
        boundary_idx_arr,
        dir_name,
        all_data,
        reconstructed_data, delta_t, xgrid,
        xticks_snapto=20,
        num_yticks=11,
        save_figs=False,
        normalization_constant_arr=None,
        xlabel=r'Time',
        ylabel=r'$x$',
        ax1_title=r'Actual Data',
        ax2_title=r'NN Reconstructed Data'):

    # saving reconstructed data
    n = len(boundary_idx_arr)
    num_cols = 3
    num_rows = 1
    
    num_modes = xgrid.shape[0]
    
    if save_figs == True:
        recon_data_dir = dir_name+'/plots/reconstructed_data'
        if not os.path.isdir(recon_data_dir):
            os.makedirs(recon_data_dir)
    
    num_digits_n = int(np.log10(n)+1)
    
    prev_idx = 0
    for i in range(n):
        fig = plt.figure(figsize=(7.5*(num_cols+0), 5.0*num_rows))
        subplot1 = 1
        subplot2 = subplot1 + 1
    
        next_idx = boundary_idx_arr[i]
        N = next_idx - prev_idx
    
        rescaled_orig_data = all_data[prev_idx:next_idx, 0:num_modes]
        rescaled_predicted_data = reconstructed_data[prev_idx:next_idx, 0:num_modes]
        if normalization_constant_arr is not None:
            rescaled_orig_data = invert_normalization(rescaled_orig_data, normalization_constant_arr)
            rescaled_predicted_data = invert_normalization(rescaled_predicted_data, normalization_constant_arr)
    
        vmin = np.min([
            rescaled_orig_data.min(),
            rescaled_predicted_data.min()
        ])
        vmax = np.max([
            rescaled_orig_data.max(),
            rescaled_predicted_data.max()
        ])

        # plotting the original data
        ax_orig = fig.add_subplot(num_rows, num_cols, subplot1)
        im_orig = ax_orig.imshow(rescaled_orig_data.transpose(), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax_orig.title.set_text(ax1_title)
        xticks = np.arange(0, N, int((xticks_snapto+0.5*delta_t)//delta_t))
        ax_orig.set_xticks(ticks=xticks)
        ax_orig.set_xticklabels(np.round(xticks*delta_t, 1))
        ax_orig.tick_params(axis='x', rotation=270+45)
        yticks = np.linspace(0, 1, num_yticks)*(len(xgrid)-1)
        yticklabels = np.round(xgrid[0]+np.linspace(0, 1, yticks.shape[0])*(xgrid[-1]-xgrid[0]), 2)
        ax_orig.set_yticks(ticks=yticks)
        ax_orig.set_yticklabels(yticklabels)
        ax_orig.set_xlabel(xlabel)
        ax_orig.set_ylabel(ylabel)
    
        # plotting the reconstructed data
        ax_predict = fig.add_subplot(num_rows, num_cols, subplot2, sharey=ax_orig, sharex=ax_orig)
        im_predict = ax_predict.imshow(rescaled_predicted_data.transpose(), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax_predict.title.set_text(ax2_title)
        ax_predict.tick_params(axis='x', rotation=270+45)
        ax_predict.set_xlabel(xlabel)
        ax_predict.set_ylabel(ylabel)

        # subplots adjustment to account for colorbars
        fig.subplots_adjust(
            bottom=0.2,
            left=0.1,
        )

        # original data and recon data colorbar
        cb_xbegin = ax_orig.transData.transform([0, 0])
        cb_xbegin = fig.transFigure.inverted().transform(cb_xbegin)[0]
        cb_xend = ax_predict.transData.transform([N, 0])
        cb_xend = fig.transFigure.inverted().transform(cb_xend)[0]

        cb_ax = fig.add_axes([cb_xbegin, 0.0, cb_xend-cb_xbegin, 0.025])
        cbar = fig.colorbar(im_predict, cax=cb_ax, orientation='horizontal')
    
        # computing the normalized error
        subplot3 = subplot2+1
        error = np.abs(rescaled_orig_data-rescaled_predicted_data)
        error_normalizer = rescaled_orig_data**2
        error_normalizer = np.mean(np.mean(0.5*(error_normalizer[0:-1, :]+error_normalizer[1:, :]), axis=0)**0.5)
        error = error / error_normalizer
        # plotting the normalized error
        ax_error = fig.add_subplot(num_rows, num_cols, subplot3, sharey=ax_orig, sharex=ax_orig)
        im_error = ax_error.imshow(error.transpose(), aspect='auto', origin='lower')
        ax_error.title.set_text(r'Normalized Error')
        ax_error.tick_params(axis='x', rotation=270+45)
        ax_error.set_xlabel(xlabel)
        ax_error.set_ylabel(ylabel)

        # error colorbar
        cbe_xbegin = ax_error.transData.transform([0, 0])
        cbe_xbegin = fig.transFigure.inverted().transform(cbe_xbegin)[0]
        cbe_xend = ax_error.transData.transform([N, 0])
        cbe_xend = fig.transFigure.inverted().transform(cbe_xend)[0]
        error_cb_ax = fig.add_axes([cbe_xbegin, 0.0, cbe_xend-cbe_xbegin, 0.025])
        cbar_error = fig.colorbar(im_error, cax=error_cb_ax, orientation='horizontal')
    
        prev_idx = next_idx
        
        # saving the figures
        if save_figs == True:
            fig.savefig(recon_data_dir+'/reconstructed_'+str(i+1).zfill(num_digits_n)+'.png', dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close()
        else:
            plt.show()
            print('')
        
    return
        
        

def plot_reconstructed_data_cdv(
        boundary_idx_arr,
        dir_name_ae,
        all_data,
        reconstructed_data,
        delta_t,
        save_figs=False):

    num_states = 6
    n = len(boundary_idx_arr)
    num_cols = 1
    if save_figs == True:
        num_rows = num_states
    else:
        num_rows = n*num_states

    if save_figs == True:
        recon_data_dir = dir_name_ae+'/plots/reconstructed_data'
        if not os.path.isdir(recon_data_dir):
            os.makedirs(recon_data_dir)
    else:
        fig = plt.figure(figsize=(7.5*num_cols, 2.5*num_rows))
    
    num_digits_n = int(np.log10(n)+1)

    N = all_data.shape[0]//n - 1
    input_time = np.arange(0, N+1)*delta_t

    prev_idx = 0
    for i in range(n):
        if save_figs == True:
            fig = plt.figure(figsize=(7.5*num_cols, 2.5*num_rows))
            subplot_idx = 1
        else:
            subplot_idx = num_states*i + 1

        next_idx = boundary_idx_arr[i]

        for j in range(num_states):
            ax_orig = fig.add_subplot(num_rows, num_cols, subplot_idx)
            ax_orig.plot(input_time, all_data[prev_idx:next_idx, j], label='Actual Data')
            ax_orig.plot(input_time, reconstructed_data[prev_idx:next_idx, j], label='NN Reconstructed Data')
            ax_orig.set_ylabel(r'$x_'+str(j+1)+'$')
            ax_orig.grid(True)
            ax_orig.set_axisbelow(True)
            if j == 0:
                # if save_figs == False or True:
                ax_orig.title.set_text(r'$x_1^*$={:.2f},  $x_4^*$={:.2f},  $C$={:.2f}, $\beta$={:.2f}, $\gamma$={:.2f}, $b$={:.2f}'.format(
                    all_data[i*(N+1), 6],
                    all_data[i*(N+1), 7],
                    all_data[i*(N+1), 8],
                    all_data[i*(N+1), 9],
                    all_data[i*(N+1), 10],
                    all_data[i*(N+1), 11]
                ))
                ax_orig.legend(loc='upper left', bbox_to_anchor=[1, 1])
            subplot_idx += 1
            if j < num_states-1:
                ax_orig.xaxis.set_ticklabels([])

        ax_orig.set_xlabel('Time')
    
        prev_idx = next_idx

        if save_figs == True:
            # fig.suptitle(r'$x_1^*$={:.2f},  $x_4^*$={:.2f},  $C$={:.2f}, $\beta$={:.2f}, $\gamma$={:.2f}, $b$={:.2f}'.format(
            #     all_data[i*(N+1), 6],
            #     all_data[i*(N+1), 7],
            #     all_data[i*(N+1), 8],
            #     all_data[i*(N+1), 9],
            #     all_data[i*(N+1), 10],
            #     all_data[i*(N+1), 11]
            # ))
            fig.savefig(recon_data_dir+'/reconstructed_'+str(i+1).zfill(num_digits_n)+'.png', dpi=300, bbox_inches='tight')
            fig.clear()
            plt.close()


    if save_figs == True:
        return
    else:
        fig.set_tight_layout(True)
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

# LYAPUNOV SPECTRUM COMPUTATION

def compute_lyapunov_spectrum(
        create_data_fn, cdf_kwargs, num_modes, 
        init_state_mat, params_mat, dy_mat,
        zeta=10, delta_completionratio=0.1, num_exp=None,
        print_flag=True, FTYPE=np.float64, ITYPE=np.int64):
    '''
    Computing the Lyapunov spectrum
    '''
    starting_time = time.time()

    # init_state_mat = np.array(init_state_mat, dtype=FTYPE)
    # params_mat = np.array(params_mat, dtype=FTYPE)
    # dy_mat = np.array(dy_mat, dtype=FTYPE)
    
    if len(params_mat.shape) == 1:
        params_mat = params_mat.reshape((1, params_mat.shape[0]))
    if num_exp is None:
        num_exp = num_modes
    elif num_exp > num_modes:
        if print_flag == True:
            print('num_exp > num_modes, being set to num_modes.')
        num_exp = num_modes

    T = cdf_kwargs['T']
    t0 = cdf_kwargs['t0']
    delta_t = cdf_kwargs['delta_t']

    N = int(((T-t0) + 0.5*delta_t) // delta_t)

    num_params = params_mat.shape[1]
    num_cases = params_mat.shape[0]

    xi = int((N+1)//zeta)
    if print_flag == True:
        print('number of evaluation intervals per case: {}\n'.format(xi))
    Rjj_mat = np.ones(shape=(num_cases, xi, num_exp), dtype=FTYPE)

    lyap_coeffs = np.empty(shape=(num_cases, num_exp), dtype=FTYPE)

    for ii in range(num_cases):
        init_state_unptb = init_state_mat[ii, :].copy()
        params = params_mat[ii].copy()
        # dY = np.eye(M)*dy
        dY = np.random.rand(num_modes, num_modes) - 0.5
        for j in range(dY.shape[1]):
            dY[:, j] /= np.linalg.norm(dY[:, j])
        dY, _ = splnalg.qr(dY)
        dY *= dy_mat[ii]

        # main loop

        # initializing the perturbed states        
        init_state_ptb_mat = np.empty(shape=(num_modes, num_modes))
        for j in range(num_modes):
            init_state_ptb_mat[:, j] = init_state_unptb + dY[:, j]
        
        ptb_state_mat = np.empty(shape=(num_modes, num_modes))

        completion_ratio = delta_completionratio
        t0_star = t0
        unptb_dict = cdf_kwargs.copy()
        unptb_dict['params_mat'] = params
        for i in range(xi):
            # for j in range(zeta):
            T_star = t0_star + zeta*delta_t
            
            # evolving the unperturbed state
            unptb_dict['t0'] = t0_star
            unptb_dict['T'] = T_star
            res_dict_unptb = create_data_fn(init_state_mat=init_state_unptb, **unptb_dict)
            all_data_unptb = res_dict_unptb['all_data']

            # evolving the perturbed states
            ptb_dict_j = unptb_dict.copy()
            for j in range(num_modes):
                init_state_ptb_j = init_state_ptb_mat[:, j].copy()
                res_dict_ptb_j = create_data_fn(init_state_mat=init_state_ptb_j, **ptb_dict_j)
                all_data_ptb_j = res_dict_ptb_j['all_data']
                ptb_state_mat[:, j] = all_data_ptb_j[-1, :]
                dY[:, j] = ptb_state_mat[:, j] - all_data_unptb[-1, :]

            # computing the lyapunov spectrum
            Q_qrdecomp, R_qrdecomp = splnalg.qr( dY/dy_mat[ii] )
            for j in range(num_modes):
                fac = 1
                temp = R_qrdecomp[j, j]
                if temp < 0:
                    fac = -1
                if j < num_exp:
                    Rjj_mat[ii, i, j] = fac*temp
                dY[:, j] = dy_mat[ii]*fac*Q_qrdecomp[:, j]

            # updating variables
            t0_star += zeta*delta_t
            init_state_unptb[:] = all_data_unptb[-1, :]
            for j in range(num_modes):
                init_state_ptb_mat[:, j] = init_state_unptb + dY[:, j]

            if i == int(completion_ratio * xi):
                if print_flag == True:
                    print(
                        'case {} completion_ratio : {}, elapsed_time : {}s, global_completion : {}%'.format(
                            ii+1,
                            np.round(completion_ratio, 3),
                            np.round(time.time()-starting_time, 3),
                            np.round(100*(completion_ratio+ii)/num_cases, 3)
                        )
                    )
                completion_ratio += delta_completionratio

        for j in range(num_exp):
            lyap_coeffs[ii, j] = np.sum(np.log(Rjj_mat[ii, :, j])) / (zeta*xi*delta_t)
        if print_flag == True:
            print('case {} MLE : {}\n'.format(ii+1, np.round(lyap_coeffs[ii, :].max(), 6)))

    return lyap_coeffs, Rjj_mat
 
################################################################################
