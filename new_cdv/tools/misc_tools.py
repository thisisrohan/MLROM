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
        params_mat, init_state_mat,
        return_params_arr=False,
        normalize=False, alldata_withparams=True,
        FTYPE=FTYPE, ITYPE=ITYPE):
    '''
    params_mat = [sigma, rho, beta]
    '''

    if len(params_mat.shape) == 1:
        params_mat = params_mat.reshape((1, params_mat.shape[0]))
        if len(init_state_mat.shape) == 1:
            init_state_mat = init_state_mat.reshape((1, init_state_mat.shape[0]))
    else:
        if len(init_state_mat.shape) == 1:
            init_state_mat = np.tile(init_state_mat, (params_mat.shape[0], 1))
    
    N = int(((T-t0) + 0.5*delta_t) // delta_t)
    cols = 3
    if alldata_withparams == True:
        cols += 3
    all_data = np.empty(
        shape=(
            params_mat.shape[0]*(N+1),
            cols
        ),
        dtype=FTYPE
    )

    boundary_idx_arr = np.empty(
        shape=params_mat.shape[0],
        dtype=ITYPE
    )

    params_arr = None
    if return_params_arr == True:
        params_arr = np.empty(shape=(boundary_idx_arr.shape[0], 3))

    normalization_constant_arr = None
    if normalize == True:
        normalization_constant_arr = np.empty(shape=boundary_idx_arr.shape[0])

    counter = 0
    X = np.empty(shape=(N+1, 3), dtype=FTYPE)
    for jj in range(params_mat.shape[0]):
        sigma = params_mat[jj, 0]
        rho = params_mat[jj, 1]
        beta = params_mat[jj, 2]

        # setting up internal vectors and parameters
        params = np.array([sigma, rho, beta])
        if return_params_arr == True:
            params_arr[counter, :] = params[:]

        X[0, :] = init_state_mat[jj, :]

        # integrating
        kwargs = {'params':params}
        for ii in range(1, N+1):
            X0 = X[ii-1:ii, :]#.reshape((1, 3))
            X_next = RK4_integrator(Lorenz_time_der, X0, delta_t, **kwargs)
            X[ii, :] = X_next

        # storing data
        idx = counter # = len(beta_arr)*len(sigma_arr)*i + len(beta_arr)*j + k
        all_data[idx*(N+1):(idx+1)*(N+1), 0:3] = X[:, :]
        if normalize == True:
            normalization_constant = (2*beta*(rho-1) + (rho-1)**2)**0.5
            normalization_constant_arr[counter] = normalization_constant
            all_data[idx*(N+1):(idx+1)*(N+1), 0:3] /= normalization_constant
        if alldata_withparams == True:
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
        init_state = init_state_mat[ii, :]
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
        return_OrgDataIdxArr=True,
        normalization_type='stddev'):
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
            def minmax(data, **kwargs):
                norm_arr_upto_that_point = kwargs.pop('normalization_arr', None)
                normalization_arr = np.empty(shape=(2, data.shape[1]), dtype=FTYPE)
                for i in range(data.shape[1]):
                    sample_min = np.min(data[:, i])
                    sample_max = np.max(data[:, i])
                    if type(norm_arr_upto_that_point) != type(None):
                        sample_min = (sample_min - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                        sample_max = (sample_max - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                    if sample_max - sample_min != 0:
                        normalization_arr[0, i] = sample_min
                        normalization_arr[1, i] = sample_max - sample_min
                    else:
                        normalization_arr[0, i] = sample_min-0.5
                        normalization_arr[1, i] = 1.0
                return data, normalization_arr

            def minmax2(data, **kwargs):
                norm_arr_upto_that_point = kwargs.pop('normalization_arr', None)
                normalization_arr = np.empty(shape=(2, data.shape[1]), dtype=FTYPE)
                for i in range(data.shape[1]):
                    sample_min = np.min(data[:, i])
                    sample_max = np.max(data[:, i])
                    if type(norm_arr_upto_that_point) != type(None):
                        sample_min = (sample_min - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                        sample_max = (sample_max - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                    if sample_max - sample_min != 0:
                        normalization_arr[0, i] = 0.5*(sample_min + sample_max)
                        normalization_arr[1, i] = 0.5*(sample_max - sample_min)
                    else:
                        normalization_arr[0, i] = sample_min
                        normalization_arr[1, i] = 1.0
                return normalization_arr

            def stddev(data, **kwargs):
                stddev_multiplier = kwargs.pop('stddev_multiplier', None)
                norm_arr_upto_that_point = kwargs.pop('normalization_arr', None)
                normalization_arr = np.empty(shape=(2, data.shape[1]), dtype=FTYPE)
                if stddev_multiplier is None:
                    stddev_multiplier = 1.414213
                for i in range(data.shape[1]):
                    sample_mean = np.mean(data[:, i])
                    sample_std = np.std(data[:, i])
                    if type(norm_arr_upto_that_point) != type(None):
                        sample_mean = (sample_mean - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                        sample_std = sample_std / norm_arr_upto_that_point[1, i]
                    normalization_arr[0, i] = sample_mean
                    normalization_arr[1, i] = stddev_multiplier*sample_std
                return normalization_arr
                
            def global_stddev(data, **kwargs):
                stddev_multiplier = kwargs.pop('stddev_multiplier', None)
                norm_arr_upto_that_point = kwargs.pop('normalization_arr', None)
                normalization_arr = np.empty(shape=(2, data.shape[1]), dtype=FTYPE)
                if stddev_multiplier is None:
                    stddev_multiplier = 1.414213
                sample_std_all = np.mean(np.std(data, axis=0))
                for i in range(data.shape[1]):
                    sample_mean = np.mean(data[:, i])
                    sample_std = sample_std_all
                    if type(norm_arr_upto_that_point) != type(None):
                        sample_mean = (sample_mean - norm_arr_upto_that_point[0, i])/norm_arr_upto_that_point[1, i]
                        sample_std = sample_std / norm_arr_upto_that_point[1, i]
                    normalization_arr[0, i] = sample_mean
                    normalization_arr[1, i] = stddev_multiplier*sample_std
                return normalization_arr

            def update_normalizationarr(norm_arr_upto_that_point, new_norm_arr):
                if type(norm_arr_upto_that_point) == type(None):
                    norm_arr_upto_that_point = np.ones_like(new_norm_arr)
                    norm_arr_upto_that_point[0, :] = 0.0
                norm_arr_upto_that_point[0, :] += new_norm_arr[0, :]*norm_arr_upto_that_point[1, :]
                norm_arr_upto_that_point[1, :] *= new_norm_arr[1, :]
                return norm_arr_upto_that_point

            tfunc_dict = {
                'minmax' : minmax,
                'minmax2' : minmax2,
                'stddev' : stddev,
                'global_stddev' : global_stddev,
            }

            normalization_arr = None
            if type(normalization_type) == type([]):
                # it's a list of sequential transformations
                for i in range(len(normalization_type)):
                    transformation = normalization_type[i]
                    tfunc = tfunc_dict[transformation]
                    kwargs = {
                        'stddev_multiplier':stddev_multiplier,
                        'normalization_arr':normalization_arr
                    }
                    this_tfunc_norm_arr = tfunc(data, **kwargs)
                    normalization_arr = update_normalizationarr(normalization_arr, this_tfunc_norm_arr)
            else:
                transformation = normalization_type
                tfunc = tfunc_dict[transformation]
                kwargs = {
                    'stddev_multiplier':stddev_multiplier,
                    'normalization_arr':normalization_arr
                }
                this_tfunc_norm_arr = tfunc(data, **kwargs)
                normalization_arr = update_normalizationarr(normalization_arr, this_tfunc_norm_arr)
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
        save_config_path=None,
        xlabel_kwargs={},
        ylabel_kwargs={},
        legend_kwargs={},
        title_kwargs={'fontsize':12},
    ):

    n = len(boundary_idx_arr)
    num_latent_states = latent_states_all.shape[1]
    N = latent_states_all.shape[0]//n - 1

    num_cols = 1
    num_rows = n*num_latent_states
    
    ax_ylabels = ['$x^*_{}$'.format(i) for i in np.arange(1, 7)]

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
        ax[j].set_ylabel(ax_ylabels[j], **ylabel_kwargs)
        if xlim is not None:
            ax[j].set_xlim(xlim)
        if ylim is not None:
            ax[j].set_ylim(ylim)
        ax[j].grid(True)
        ax[j].set_axisbelow(True)


    ax[-1].set_xlabel('Time', **xlabel_kwargs)

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
        markerscale=legend_markerscale,
        **legend_kwargs
    )
    # fig.suptitle(r'Latent States', size=12)
    ax[0].set_title(r'Latent States', **title_kwargs)

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
        params_mat,
        xlim=None,
        ylim=None,
        max_rows=10,
        legend_bbox_to_anchor=[1, 1],
        legend_loc='upper left',
        legend_markerscale=10,
        markersize=1,
        cmap_name='viridis',
        save_figs=False,
        dir_name_ae=None,
        xlabel_kwargs={},
        ylabel_kwargs={},
        legend_kwargs={},
    ):

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
            label=r'[$\sigma$, $\rho$, $\beta$] = ' + np.array2string(params_mat[i, :], precision=2, separator=', ')
        )
        prev_idx = next_idx

    if type(xlim) != type(None):
        ax.set_xlim(xlim[0], xlim[1])
    if type(ylim) != type(None):
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel('Latent Dimension 1', **xlabel_kwargs)
    ax.set_ylabel('Latent Dimension 2', **ylabel_kwargs)
    # max_rows = 10
    max_rows = float(max_rows)
    ncols = int(np.ceil(len(boundary_idx_arr) / max_rows))
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=ncols, markerscale=legend_markerscale, **legend_kwargs)
    if save_figs == True:
        plt.savefig(dir_name_ae+'/plots/latent_space.pdf', dpi=300, bbox_inches='tight')
        # plt.show()
        return
    else:
        return fig, ax

################################################################################

def plot_losses(
        training_loss,
        val_loss=None,
        more_plot_arrs_lst=[],
        lr_change=[0],
        learning_rate_list=None,
        lr_y_placement_axes_coord=0.5,
        lr_x_offset_axes_coord=0.06,
        legend_list=['Training Loss', 'Validation Loss'],
        xlabel='Epoch',
        ylabel='Loss',
        traininglossplot_args=['r--'],
        traininglossplot_kwargs={},
        vallossplot_args=['b-'],
        vallossplot_kwargs={'linewidth':0.8},
        more_plot_arrs_args=[],
        more_plot_arrs_kwargs=[],
        plot_type='semilogy',
        epoch_count_begin=None,
        epoch_count_end=None,
        xlabel_kwargs={},
        ylabel_kwargs={},
        legend_kwargs={},
        ):

    if epoch_count_begin == None:
        epoch_count_begin = 1
    if epoch_count_end == None:
        epoch_count_end = len(training_loss)
    epoch_count = range(epoch_count_begin, epoch_count_end + 1)

    fig, ax = plt.subplots()

    eval("ax."+plot_type+"(epoch_count, training_loss, *traininglossplot_args, **traininglossplot_kwargs)")
    if type(val_loss) != type(None):
        eval("ax."+plot_type+"(epoch_count, val_loss, *vallossplot_args, **vallossplot_kwargs)")
    if len(more_plot_arrs_lst) > 0:
        for i in range(len(more_plot_arrs_lst)):
            if len(more_plot_arrs_args) >= i+1:
                args = more_plot_arrs_args[i]
                kwargs = more_plot_arrs_kwargs[i]
            else:
                args = []
                kwargs = {}
            eval("ax."+plot_type+"(epoch_count, more_plot_arrs_lst[i], *args, **kwargs)")
    ax.legend(legend_list, **legend_kwargs)
    ax.set_xlabel(xlabel, **xlabel_kwargs)
    ax.set_ylabel(ylabel, **ylabel_kwargs)
    ax.grid(True, which='major', axis='x')
    ax.grid(True, which='both', axis='y')
    ax.set_axisbelow(True)

    if learning_rate_list is not None:
        for i in range(len(lr_change)-1):
            ax.axvline(lr_change[i]+1, color='m', linestyle='-.', linewidth=0.8)
            lr_txt = "{:.2E}".format(learning_rate_list[i])
            idx = lr_txt.find('E')
            lr_txt = "lr = $" + lr_txt[0:idx] + " \\times 10^{" + lr_txt[idx+1:] + "}$"
            
            ax.text(
                lr_change[i]+ax.transData.inverted().transform(ax.transAxes.transform([lr_x_offset_axes_coord, 0.]))[0],
                ax.transData.inverted().transform(ax.transAxes.transform([0., lr_y_placement_axes_coord]))[1],
                lr_txt,# 'lr={:.2E}'.format(learning_rate_list[i]),
                rotation=90,
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(facecolor='yellow', alpha=0.25, boxstyle='square,pad=0.2')
            )

    return fig, ax


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
    nbins = max(1, int(np.round(bin_end/bin_width)))

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


################################################################################

def plot_reconstructed_data(
        boundary_idx_arr,
        dir_name_ae,
        all_data,
        reconstructed_data,
        params_mat,
        save_figs=False,
        xlabel_kwargs={},
        ylabel_kwargs={},
        zlabel_kwargs={},
        title_kwargs={},
    ):

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
        # ax_orig.title.set_text(r'Actual Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(params_mat[i], precision=2, separator=', '), **title_kwargs)
        ax_orig.set_title(r'Actual Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(params_mat[i], precision=2, separator=', '), **title_kwargs)
        ax_orig.set_xlabel('$x_1$', **xlabel_kwargs)
        ax_orig.set_ylabel('$x_2$', **ylabel_kwargs)
        ax_orig.set_zlabel('$x_3$', **zlabel_kwargs)
        
        ax_predict = fig.add_subplot(num_rows, num_cols, subplot2, projection ='3d')
        ax_predict.plot(reconstructed_data[prev_idx:next_idx, 0], reconstructed_data[prev_idx:next_idx, 1], reconstructed_data[prev_idx:next_idx, 2])
        # ax_predict.title.set_text(r'NN Reconstructed Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(params_mat[i], precision=2, separator=', '), **title_kwargs)
        ax_predict.set_title(r'NN Reconstructed Data - [$\sigma$, $\rho$, $\beta$] = ' + np.array2string(params_mat[i], precision=2, separator=', '), **title_kwargs)
        ax_predict.set_xlabel('$x_1$', **xlabel_kwargs)
        ax_predict.set_ylabel('$x_2$', **ylabel_kwargs)
        ax_predict.set_zlabel('$x_3$', **zlabel_kwargs)
    
        prev_idx = next_idx

        if save_figs == True:
            fig.savefig(recon_data_dir+'/reconstructed_'+str(i+1).zfill(num_digits_n)+'.pdf', dpi=300, bbox_inches='tight')
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
        ax2_title=r'NN Reconstructed Data',
    ):

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
        params_mat,
        delta_t,
        save_figs=False,
        xlabel_kwargs={},
        ylabel_kwargs={},
        title_kwargs={},
        legend_kwargs={},
        save_extension='png',
    ):

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
            ax_orig.set_ylabel(r'$x_'+str(j+1)+'$', **ylabel_kwargs)
            ax_orig.grid(True)
            ax_orig.set_axisbelow(True)
            if j == 0:
                # if save_figs == False or True:
                ax_orig.set_title(
                    r'$x_1^*$={:.2f},  $x_4^*$={:.2f},  $C$={:.2f}, $\beta$={:.2f}, $\gamma$={:.2f}, $b$={:.2f}'.format(
                        params_mat[i, 0],
                        params_mat[i, 1],
                        params_mat[i, 2],
                        params_mat[i, 3],
                        params_mat[i, 4],
                        params_mat[i, 5]
                    ),
                    **title_kwargs,
                )
                ax_orig.legend(loc='upper left', bbox_to_anchor=[1, 1], **legend_kwargs)
            subplot_idx += 1
            if j < num_states-1:
                ax_orig.xaxis.set_ticklabels([])

        ax_orig.set_xlabel('Time', **xlabel_kwargs)
    
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
            fig.savefig(recon_data_dir+'/reconstructed_'+str(i+1).zfill(num_digits_n)+'.'+save_extension, dpi=300, bbox_inches='tight')
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
        return_earlystopping_wait=False,
        fname='LossHistoriesCheckpoint.hdf5'):

    with h5py.File(dir_name_ae+'{ds}checkpoints{ds}'.format(ds=dir_sep)+fname, 'r') as f:
        val_loss_arr_fromckpt = np.array(f['val_loss_arr'])
        train_loss_arr_fromckpt = np.array(f['train_loss_arr'])
    
    val_loss_hist = []
    train_loss_hist = []
    lr_change=[0]
    
    if type(epochs) != type([]):
        epochs = [epochs]*len(learning_rate_list)
        epochs_og_list_flag = False
    else:
        epochs_og_list_flag = True
    # num_epochs_left = epochs
    num_epochs_left = [ep for ep in epochs]
    starting_lr_idx = 0
    epochs_covered = 0
    # for j in range(len(epochs)):
    #     epochs_j = epochs[j]
    for i in range(len(learning_rate_list)):
        epochs_i = epochs[i]
        nan_flags = np.isnan(val_loss_arr_fromckpt[epochs_covered + epochs_i*i:epochs_i*(i+1)])
        if nan_flags[0] == True:
            # this and all further learning rates were not reached in previous training session
            break
        else:
            temp_ = np.where(nan_flags == False)[0]
            num_epochs_left[i] = epochs_i - len(temp_)
            starting_lr_idx = i
    
            val_loss_hist.extend(val_loss_arr_fromckpt[epochs_covered + epochs_i*i:epochs_i*(i+1)][temp_])
            train_loss_hist.extend(train_loss_arr_fromckpt[epochs_covered + epochs_i*i:epochs_i*(i+1)][temp_])
            lr_change.append(lr_change[i]+len(temp_))
            
        epochs_covered += epochs_i

    if epochs_og_list_flag == False:
        num_epochs_left = num_epochs_left[0]

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
# everything in this section is taken from `numpy-hilbert-curve`.              #
# https://github.com/PrincetonLIPS/numpy-hilbert-curve/                        #
################################################################################

def right_shift(binary, k=1, axis=-1):
    ''' Right shift an array of binary values.
    Parameters:
    -----------
    binary: An ndarray of binary values.
    k: The number of bits to shift. Default 1.
    axis: The axis along which to shift.  Default -1.
    Returns:
    --------
    Returns an ndarray with zero prepended and the ends truncated, along
    whatever axis was specified.
    '''

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return np.zeros_like(binary)

    # Determine the padding pattern.
    padding = [(0,0)] * len(binary.shape)
    padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)

    shifted = np.pad(binary[tuple(slicing)], padding,
                   mode='constant', constant_values=0)

    return shifted


def binary2gray(binary, axis=-1):
    ''' Convert an array of binary values into Gray codes.
    This uses the classic X ^ (X >> 1) trick to compute the Gray code.
    Parameters:
    -----------
    binary: An ndarray of binary values.
    axis: The axis along which to compute the gray code. Default=-1.
    Returns:
    --------
    Returns an ndarray of Gray codes.
    '''
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = np.logical_xor(binary, shifted)

    return gray

def gray2binary(gray, axis=-1):
    ''' Convert an array of Gray codes back into binary values.
    Parameters:
    -----------
    gray: An ndarray of gray codes.
    axis: The axis along which to perform Gray decoding. Default=-1.
    Returns:
    --------
    Returns an ndarray of binary values.
    '''

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2**(int(np.ceil(np.log2(gray.shape[axis])))-1)
    while shift > 0:
        gray = np.logical_xor( gray, right_shift(gray, shift) )
        shift //= 2

    return gray

def right_shift(binary, k=1, axis=-1):
    ''' Right shift an array of binary values.
    Parameters:
    -----------
    binary: An ndarray of binary values.
    k: The number of bits to shift. Default 1.
    axis: The axis along which to shift.  Default -1.
    Returns:
    --------
    Returns an ndarray with zero prepended and the ends truncated, along
    whatever axis was specified.
    '''

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return np.zeros_like(binary)

    # Determine the padding pattern.
    padding = [(0,0)] * len(binary.shape)
    padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)

    shifted = np.pad(binary[tuple(slicing)], padding,
                   mode='constant', constant_values=0)

    return shifted


def binary2gray(binary, axis=-1):
    ''' Convert an array of binary values into Gray codes.
    This uses the classic X ^ (X >> 1) trick to compute the Gray code.
    Parameters:
    -----------
    binary: An ndarray of binary values.
    axis: The axis along which to compute the gray code. Default=-1.
    Returns:
    --------
    Returns an ndarray of Gray codes.
    '''
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = np.logical_xor(binary, shifted)

    return gray

def gray2binary(gray, axis=-1):
    ''' Convert an array of Gray codes back into binary values.
    Parameters:
    -----------
    gray: An ndarray of gray codes.
    axis: The axis along which to perform Gray decoding. Default=-1.
    Returns:
    --------
    Returns an ndarray of binary values.
    '''

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2**(int(np.ceil(np.log2(gray.shape[axis])))-1)
    while shift > 0:
        gray = np.logical_xor( gray, right_shift(gray, shift) )
        shift //= 2

    return gray

def encode(locs, num_dims, num_bits):
    ''' Decode an array of locations in a hypercube into a Hilbert integer.
    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:
    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.
    Params:
    -------
    locs - An ndarray of locations in a hypercube of num_dims dimensions, in
          which each dimension runs from 0 to 2**num_bits-1.  The shape can
          be arbitrary, as long as the last dimension of the same has size
          num_dims.
    num_dims - The dimensionality of the hypercube. Integer.
    num_bits - The number of bits for each dimension. Integer.
    Returns:
    --------
    The output is an ndarray of uint64 integers with the same shape as the
    input, excluding the last dimension, which needs to be num_dims.
    '''

    # Keep around the original shape for later.
    orig_shape = locs.shape

    if orig_shape[-1] != num_dims:
        raise ValueError(
            '''
            The shape of locs was surprising in that the last dimension was of size
            %d, but num_dims=%d.  These need to be equal.
            ''' % (orig_shape[-1], num_dims)
        )

    if num_dims*num_bits > 64:
        raise ValueError(
            '''
            num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
            into a uint64.  Are you sure you need that many points on your Hilbert
            curve?
            ''' % (num_dims, num_bits)
        )

    # TODO: check that the locations are valid.

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    locs_uint8 = np.reshape(locs.astype('>u8').view(np.uint8), (-1, num_dims, 8))

    # Now turn these into bits and truncate to num_bits.
    gray = np.unpackbits(locs_uint8, axis=-1)[...,-num_bits:]

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):

        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):

            # Identify which ones have this bit active.
            mask = gray[:,dim,bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], mask[:,np.newaxis])

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = np.logical_and(
                np.logical_not(mask[:,np.newaxis]),
                np.logical_xor(gray[:,0,bit+1:], gray[:,dim,bit+1:])
            )
            gray[:,dim,bit+1:] = np.logical_xor(gray[:,dim,bit+1:], to_flip)
            gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], to_flip)

    # Now flatten out.
    gray = np.reshape(
        np.swapaxes(gray, axis1=1, axis2=2),
        (-1, num_bits*num_dims)
    )

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits*num_dims
    padded = np.pad(hh_bin, ((0, 0), (extra_dims, 0)),
                  mode='constant', constant_values=0)

    # Convert binary values into uint8s.
    hh_uint8 = np.squeeze(np.packbits(np.reshape(padded[:,::-1], (-1, 8, 8)),
                                    bitorder='little', axis=2))

    # Convert uint8s into uint64s.
    hh_uint64 = np.squeeze(hh_uint8.view(np.uint64))

    return hh_uint64

def decode(hilberts, num_dims, num_bits):
    ''' Decode an array of Hilbert integers into locations in a hypercube.
    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:
    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.
    Params:
    -------
    hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
              cannot have fewer bits than num_dims * num_bits.
    num_dims - The dimensionality of the hypercube. Integer.
    num_bits - The number of bits for each dimension. Integer.
    Returns:
    --------
    The output is an ndarray of unsigned integers with the same shape as hilberts
    but with an additional dimension of size num_dims.
    '''

    if num_dims*num_bits > 64:
        raise ValueError(
            '''
            num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
            into a uint64.  Are you sure you need that many points on your Hilbert
            curve?
            ''' % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    hilberts = np.atleast_1d(hilberts)

    # Keep around the shape for later.
    orig_shape = hilberts.shape

    # Treat each of the hilberts as a sequence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    hh_uint8 = np.reshape(hilberts.ravel().astype('>u8').view(np.uint8), (-1, 8))

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    hh_bits = np.unpackbits(hh_uint8, axis=1)[:,-num_dims*num_bits:]

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    gray = np.swapaxes(
        np.reshape(gray, (-1, num_bits, num_dims)),
        axis1=1, axis2=2,
    )

    # Iterate backwards through the bits.
    for bit in range(num_bits-1, -1, -1):

        # Iterate backwards through the dimensions.
        for dim in range(num_dims-1, -1, -1):

            # Identify which ones have this bit active.
            mask = gray[:,dim,bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], mask[:,np.newaxis])

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = np.logical_and(
                np.logical_not(mask[:,np.newaxis]),
                np.logical_xor(gray[:,0,bit+1:], gray[:,dim,bit+1:])
            )
            gray[:,dim,bit+1:] = np.logical_xor(gray[:,dim,bit+1:], to_flip)
            gray[:,0,bit+1:] = np.logical_xor(gray[:,0,bit+1:], to_flip)

    # Pad back out to 64 bits.
    extra_dims = 64 - num_bits
    padded = np.pad(gray, ((0,0), (0,0), (extra_dims,0)),
                  mode='constant', constant_values=0)

    # Now chop these up into blocks of 8.
    locs_chopped = np.reshape(padded[:,:,::-1], (-1, num_dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    locs_uint8 = np.squeeze(np.packbits(locs_chopped, bitorder='little', axis=3))

    # Finally, treat these as uint64s.
    flat_locs = locs_uint8.view(np.uint64)

    # Return them in the expected shape.
    return np.reshape(flat_locs, (*orig_shape, num_dims))

################################################################################

def return_hilbert_x0(num_points, dimensions, order):

    hindex = np.linspace(0, 2**(dimensions*order), num_points+1, dtype=np.int64)[0:-1]
    locs = decode(hindex, dimensions, order)
    
    return locs

################################################################################
