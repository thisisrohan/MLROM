################################################################################
# Convolutional ESN with uniform/normal noise added to every input.            #
################################################################################

import os
import numpy as np
from scipy import linalg
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import time as time

import h5py

import tensorflow as tf
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

################################################################################
#################################### LSTM V4 ###################################

class single_weights(layers.Layer):
    def __init__(self, w_regularizer=None, **kwargs):
        super(single_weights, self).__init__()
        self._weights_regularizer = w_regularizer
        
    def build(self, input_shape):
        self.individual_weights = self.add_weight(
            name='individual_weights',
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.33),
            regularizer=self._weights_regularizer,
            trainable=True
        )

    def call(self, x):
        return x * self.individual_weights


class ESN_Cell(layers.Layer):
    def __init__(
            self, omega_in, kernel_size, rho_res, res_channels, alpha=1.0,
            usebias_Win=False, prng_seed=42, activation='tanh', **kwargs):

        name = kwargs.pop('name', None)
        if name != None:
            super().__init__(name=name)
        else:
            super().__init__()

        self.omega_in = omega_in
        self.kernel_size = kernel_size
        self.rho_res = rho_res
        self.alpha = alpha
        self.usebias_Win = usebias_Win
        self.prng = np.random.default_rng(seed=prng_seed)
        self.activation = eval('tf.keras.activations.'+activation)
        
        # self.Win_np = None
        # self.Win_bias_np = None
        # self.Wres_np = self._build_Wres()
        
        self.input_channels = None
        self.input_ylen = None
        self.input_xlen = None
        self.res_ylen = None
        self.res_xlen = None
        self.og_spectral_rad_Wres = None
        self.res_channels = res_channels
        
        elp = int(0.5*(self.kernel_size-1))
        erp = self.kernel_size-1 - elp
        self.input_periodic_padding = periodic_padding(elp, erp, name='pp_input')
        self.Win = layers.Conv2D(
            filters=self.res_channels,
            kernel_size=self.kernel_size,
            data_format='channels_first',
            padding='valid',
            use_bias=self.usebias_Win,
            dtype='float32',
            name=self.name+'/Win',
        )
        self.res_periodic_padding = periodic_padding(elp, erp, name='pp_res')
        self.Wres = layers.Conv2D(
            filters=self.res_channels,
            kernel_size=self.kernel_size,
            data_format='channels_first',
            padding='valid',
            use_bias=False,
            dtype='float32',
            # name='Wres',
        )
        
        return

    def _build_Win(self):
        
        shape = (self.kernel_size, self.kernel_size, self.input_channels, self.res_channels)
        Win_kernel_np = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=shape)
        
        if self.usebias_Win == True:
            Win_bias = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=self.res_channels)
            return_tuple = (Win_kernel_np.astype('float32'), Win_bias.astype('float32'))
        else:
            return_tuple = (Win_kernel_np.astype('float32'), None)
        
        return return_tuple

    def _build_Wres_conv_kernel(self):
    
        start_time = time.time()
        padded_xlen = self.res_xlen + self.kernel_size - 1
        padded_ylen = self.res_ylen + self.kernel_size - 1
        shape = [self.res_xlen*self.res_ylen*self.res_channels, padded_xlen*padded_ylen*self.res_channels]
        kernel_ = self.prng.uniform(low=-1, high=1, size=[self.kernel_size, self.kernel_size, self.res_channels, self.res_channels])

        Wres = np.zeros(shape=shape, dtype=np.float32) 
        temp = np.zeros(shape=(padded_ylen, padded_xlen), dtype=np.float32)
        # print(Wres.shape)
        row_ind = []
        col_ind = []
        data = []
        for i0 in range(self.res_channels):
            # i0 corresponds to channels of the output (rows-side of Wres)
            i0_row_begin = i0*self.res_xlen*self.res_ylen
            for i1 in range(self.res_channels):
                # i1 corresponds to channels of the input (columns-side of Wres)
                i1_col_begin = i1*padded_xlen*padded_ylen
                i1_col_end = (i1+1)*padded_xlen*padded_ylen
                kernel_i0_i1 = kernel_[:, :, i1, i0]
                for j0 in range(self.res_ylen):
                    j0_row_begin = i0_row_begin + j0*self.res_xlen
                    for j1 in range(self.res_xlen):
                        j1_row = j0_row_begin + j1
                        temp[:, :] = 0.
                        temp[j0:j0+self.kernel_size, j1:j1+self.kernel_size] = kernel_i0_i1
                        # data_ = temp.reshape(-1,)
                        # idx = np.where(data_ > -10)[0]
                        # row_ind.extend([j1_row]*len(idx))
                        # col_ind.extend(idx+i1_col_begin)
                        # data.extend(np.reshape(kernel_i0_i1, (-1,)))
                        Wres[j1_row, i1_col_begin:i1_col_end] = np.reshape(temp, (padded_xlen*padded_ylen))
        # print(Wres.shape)
        # Wres = sp.csc_array(
        #     (
        #         data,
        #         (row_ind, col_ind)
        #     ),
        #     shape=shape
        # )
        # Wres.eliminate_zeros()
        # Wres.sort_indices()
        print('init Wres made : {} s.'.format(time.time()-start_time))

        # DEALING WITH PERIODIC PADDING
        elp = int(0.5*(self.kernel_size - 1)) # left and top padding
        erp = self.kernel_size - 1 - elp # right and bottom padding
        selection_arr = np.ones(shape=Wres.shape[1], dtype=np.bool_) # used to select which columns to keep
        for i1 in range(self.res_channels):
            # i1 corresponds to channels of the input (columns-side of Wres)
            begin_col = i1*padded_xlen*padded_ylen
            end_col = begin_col + padded_xlen*padded_ylen
            
            # dealing with top padding
            # for j0 in range(elp*padded_xlen):
            #     col_ind_pp = begin_col + j0
            #     col_ind_toset = end_col-erp*padded_xlen-elp*padded_xlen + j0
            #     ind1ptr_pp = Wres.indptr[col_ind_pp]
            #     ind2ptr_pp = Wres.indptr[col_ind_pp+1]
            #     ind1ptr_toset = Wres.indptr[col_ind_toset]
            #     ind2ptr_toset = Wres.indptr[col_ind_toset+1]
            #     pp_rows = Wres.indices[ind1ptr_pp:ind2ptr_pp]
            #     toset_rows = Wres.indices[ind1ptr_toset:ind2ptr_toset]
            #     for j1_ind in len(pp_rows):
            #         j1 = pp_rows[j1_ind]
            #         idx = np.where(toset_rows == j1)
            #         if len(idx) != 0:
            #             Wres.data[ind1ptr_toset + idx[0]] += Wres.data[ind1ptr_pp + j1_ind]
                
            Wres[:, end_col-erp*padded_xlen-elp*padded_xlen:end_col-erp*padded_xlen] += Wres[:, begin_col:begin_col+elp*padded_xlen]
            selection_arr[begin_col:begin_col+elp*padded_xlen] = False
            
            # dealing with bottom padding
            Wres[:, begin_col+elp*padded_xlen:begin_col+elp*padded_xlen+erp*padded_xlen] += Wres[:, end_col-erp*padded_xlen:end_col]
            selection_arr[end_col-erp*padded_xlen:end_col] = False

            for j0 in range(elp, elp+self.res_ylen):
                begin_j0 = begin_col + j0*padded_xlen
                end_j0 = begin_j0 + padded_xlen
                
                # dealing with left padding
                Wres[:, end_j0-erp-elp:end_j0-erp] += Wres[:, begin_j0:begin_j0+elp]
                selection_arr[begin_j0:begin_j0+elp] = False
                
                # dealing with right padding
                Wres[:, begin_j0+elp:begin_j0+elp+erp] += Wres[:, end_j0-erp:end_j0]
                selection_arr[end_j0-erp:end_j0] = False
        print('Wres pp dealt with : {} s.'.format(time.time()-start_time))

        # print(Wres.shape)
        Wres = Wres[:, selection_arr]
        Wres = sp.csr_array(Wres)
        print('sparse Wres made : {} s.'.format(time.time()-start_time))
        # print(Wres.shape)
        # self.og_spectral_rad_Wres = np.max(np.abs(np.linalg.eigvals(Wres)))
        self.og_spectral_rad_Wres = np.abs(sp_la.eigs(Wres, k=1, which='LM', return_eigenvectors=False))[0]
        print('spectral radius found : {} s.'.format(time.time()-start_time))
        fac = self.rho_res / self.og_spectral_rad_Wres
        kernel_ *= fac
        
        return kernel_.astype('float32')

    def build(self, input_shape):
    
        super().build(input_shape)
        self.input_channels = input_shape[-3]
        self.input_ylen = input_shape[-2]
        self.input_xlen = input_shape[-1]
        
        self.res_ylen = self.input_ylen
        self.res_xlen = self.input_xlen

        # Win layer
        Win_kernel_np, Win_bias_np = self._build_Win()
        padded_xlen = self.input_xlen + self.kernel_size - 1
        padded_ylen = self.input_ylen + self.kernel_size - 1
        Win_input_shape = list(input_shape)
        Win_input_shape[-2] = padded_ylen
        Win_input_shape[-1] = padded_xlen
        self.Win.build(input_shape=Win_input_shape)
        K.set_value(self.Win.kernel, Win_kernel_np)
        if self.usebias_Win == True:
            K.set_value(self.Win.bias, Win_bias_np)
        self.Win.trainable = False


        Wres_kernel = self._build_Wres_conv_kernel()
        padded_xlen = self.res_xlen + self.kernel_size - 1
        padded_ylen = self.res_ylen + self.kernel_size - 1
        res_input_shape = list(input_shape)
        res_input_shape[-3] = self.res_channels
        res_input_shape[-2] = padded_ylen
        res_input_shape[-1] = padded_xlen
        self.Wres.build(input_shape=res_input_shape)
        K.set_value(self.Wres.kernel, Wres_kernel)
        self.Wres.trainable = False

        elp = int(0.5*(self.kernel_size - 1)) # left and top padding
        erp = self.kernel_size - 1 - elp # right and bottom padding
        self.res_periodic_padding.build(
            input_shape=(
                input_shape[0],
                self.res_channels,
                self.res_ylen,
                self.res_xlen,
            )
        )
        self.input_periodic_padding.build(
            input_shape=input_shape
        )

    @tf.function
    def call(self, inputs, states):

        states = states[0]
        # print(inputs.shape)
        # print(states.shape)
        padded_states = self.res_periodic_padding(states)
        padded_inputs = self.input_periodic_padding(inputs)
        # print(padded_inputs.shape)
        # print(padded_states.shape)

        t1 = self.Wres(padded_states)
        t2 = self.Win(padded_inputs)
        # print(t1.shape, t2.shape)
        candidate_states = self.activation(t1 + t2)
        
        # computing new_state as a weighted average of old state and candidate state
        new_states = states + self.alpha * (candidate_states - states)
        # print(new_states.shape)

        return new_states, (new_states,)

    @property
    def state_size(self):
        return tf.TensorShape([self.res_channels, self.res_ylen, self.res_xlen])

    @property
    def output_size(self):
        return tf.TensorShape([self.res_channels, self.res_ylen, self.res_xlen])

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     batch_size = inputs.shape[0] if inputs.shape[0] != None else 1
    #     return tf.zeros(shape=(batch_size, inputs.shape[-1]), dtype='float32')

        
class periodic_padding(layers.Layer):
    # only operates on square images
    def __init__(self, elp, erp, **kwargs):
        name = kwargs.pop('name', None)
        if name != None:
            super().__init__(name=name)
        else:
            super().__init__()
        self.elp = elp
        self.erp = erp
        
    def build(self, input_shape):
        super().build(input_shape)
        input_xlen = input_shape[-1]
        input_ylen = input_shape[-2] # must be the same as `input_xlen`
        M_mat = np.zeros(shape=(input_ylen+self.elp+self.erp, input_ylen), dtype=np.float32)
        for i in range(1, self.elp + 1):
            M_mat[self.elp - i, (input_ylen - i) % input_ylen] = 1.
        for i in range(self.elp, self.elp + input_ylen):
            M_mat[i, i-self.elp] = 1.
        for i in range(self.elp + input_ylen, M_mat.shape[0]):
            M_mat[i, (i - (self.elp + input_ylen)) % input_ylen] = 1.
        # B_mat = np.empty(shape=(A_mat.shape[1], A_mat.shape[0]), dtype=A_mat.dtype)
        N_mat = np.transpose(M_mat)
        
        self.M_mat = tf.Variable(M_mat, trainable=False, name=self.name+'/M_mat')
        self.N_mat = tf.Variable(N_mat, trainable=False, name=self.name+'/N_mat')

    def call(self, x, training=None):
        '''
        x has shape --> [batch_size, channels, y_len, x_len]
        '''
        return tf.linalg.matmul(tf.linalg.matmul(self.M_mat, x), self.N_mat)

    def compute_output_shape(self, input_shape):
        # input_shape : [batch_size?, time_len?, in_channels, ylen, xlen]
        output_shape = (input_shape[-3], input_shape[-2]+self.elp+self.erp, input_shape[-1]+self.elp+self.erp)
        ndims = len(input_shape)
        if ndims >= 4:
            output_shape = tuple(input_shape[0:ndims-3]) + output_shape
        return output_shape


class ESN(Model):
    """
    Single-step ESN network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self, data_dim=None,
            dt_rnn=None,
            lambda_reg=0.0,
            kernel_size=3,
            res_channels=200,
            load_file=None,
            stddev=0.0,
            mean=0.0,
            noise_type='uniform',
            stateful=True,
            omega_in=0.5,
            rho_res=0.6,
            usebias_Win=False,
            alpha=1.,
            ESN_cell_activations='tanh',
            prng_seed=42,
            usebias_Wout=False,
            batch_size=1,
            **kwargs,
            ):
        
        super(ESN, self).__init__()

        self.load_file = load_file
        self.data_dim = data_dim
        self.dt_rnn = dt_rnn
        self.lambda_reg = lambda_reg
        self.kernel_size = kernel_size
        self.res_channels = res_channels
        self.mean = mean
        self.stddev = stddev
        self.noise_type = noise_type
        self.stateful = stateful
        self.omega_in = omega_in
        self.rho_res = rho_res
        self.usebias_Win = usebias_Win
        self.prng_seed = prng_seed
        self.alpha = alpha
        self.ESN_cell_activations = ESN_cell_activations
        self.usebias_Wout = usebias_Wout
        self.batch_size = batch_size
        if self.load_file is not None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            if 'data_dim' in load_dict.keys():
                self.data_dim = load_dict['data_dim']
            if 'dt_rnn' in load_dict.keys():
                self.dt_rnn = load_dict['dt_rnn']
            if 'lambda_reg' in load_dict.keys():
                self.lambda_reg = load_dict['lambda_reg']
            if 'kernel_size' in load_dict.keys():
                self.kernel_size = load_dict['kernel_size']
            if 'res_channels' in load_dict.keys():
                self.res_channels = load_dict['res_channels']
            if 'mean' in load_dict.keys():
                self.mean = load_dict['mean']
            if 'stddev' in load_dict.keys():
                self.stddev = load_dict['stddev']
            if 'noise_type' in load_dict.keys():
                self.noise_type = load_dict['noise_type']
            if 'stateful' in load_dict.keys():
                self.stateful = load_dict['stateful']
            if 'omega_in' in load_dict.keys():
                self.omega_in = load_dict['omega_in']
            if 'rho_res' in load_dict.keys():
                self.rho_res = load_dict['rho_res']
            if 'usebias_Win' in load_dict.keys():
                self.usebias_Win = load_dict['usebias_Win']
            if 'alpha' in load_dict.keys():
                self.alpha = load_dict['alpha']
            if 'prng_seed' in load_dict.keys():
                self.prng_seed = load_dict['prng_seed']
            if 'ESN_cell_activations' in load_dict.keys():
                self.ESN_cell_activations = load_dict['ESN_cell_activations']
            if 'usebias_Wout' in load_dict.keys():
                self.usebias_Wout = load_dict['usebias_Wout']

        self.noise_gen = eval('tf.random.'+self.noise_type)
        if self.noise_type == 'uniform':
            self.noise_kwargs = {
                'minval':self.mean-1.732051*self.stddev,
                'maxval':self.mean+1.732051*self.stddev
            }
        elif self.noise_type == 'normal':
            self.noise_kwargs = {
                'mean':self.mean,
                'stddev':self.stddev
            }


        ### the ESN network
        self.rnn_list = [
            layers.RNN(
                cell=ESN_Cell(
                    omega_in=self.omega_in,
                    res_channels=self.res_channels,
                    rho_res=self.rho_res,
                    kernel_size=self.kernel_size,
                    alpha=self.alpha,
                    usebias_Win=self.usebias_Win,
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations,
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            )
        ]
        
        elp = int(0.5*(self.kernel_size-1))
        erp = self.kernel_size - 1 - elp
        self.post_res_periodic_padding = periodic_padding(elp, erp, name='pp_post_res')
        
        ### adding the Wout layer
        reg = lambda x : None
        if self.lambda_reg != None:
            if self.lambda_reg > 0.0:
                reg = lambda x : tf.keras.regularizers.L2(x)
        self.Wout = layers.Conv2D(
            filters=self.data_dim[-3],
            kernel_size=self.kernel_size,
            data_format='channels_first',
            padding='valid',
            use_bias=self.usebias_Wout,
            dtype='float32',
            name='Wout',
            kernel_regularizer=reg(self.lambda_reg),
            bias_regularizer=reg(self.lambda_reg),
        )
        
        self.power_arr = np.ones(shape=(1, 1, self.res_channels, 1, 1))
        self.power_arr[0, 0, 0::2, 0, 0] = 2

        # self.global_Hb = np.zeros(shape=[shape[0]]*2, dtype='float32')
        # self.global_Yb = np.zeros(shape=[self.data_dim, shape[0]], dtype='float32')


        ### initializing weights
        # temp = tf.ones(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = Input(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = self(temp)

        return

    # def build(self, input_shape):

    #     if self.stateful:
    #         input_shape_ESN = (None, )
    #     else:
    #         input_shape_ESN = (self.batch_size, )
    #     input_shape_ESN = input_shape_ESN + tuple(input_shape[1:])

    #     for rnnlayer in self.rnn_list:
    #         rnnlayer.build(input_shape_ESN)
    #         input_shape_ESN = input_shape_ESN[0:-1] + (rnnlayer.cell.state_num_units, )

    #     self.Wout.build(input_shape_ESN)
    #     # super(ESN, self).build(input_shape)
    #     if self.use_weights_post_dense == True:
    #         self.postWout.build(input_shape=[None, input_shape[-1]])

    #     self.built = True

    @tf.function
    def _callhelperfn(self, x):
        for i in range(len(self.rnn_list)):
            x = self.rnn_list[i](x)
        return x

    @tf.function
    def call(self, inputs, training=False, manual_training=False):

        # inputs shape : (None, time_steps, data_dim)
        out_steps = inputs.shape[1]

        ### Running through the ESN layers
        # x = tf.Variable(inputs)
        x = inputs
        if training == True or manual_training == True:
            # add noise to the inputs during training
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        x = self._callhelperfn(x)

        ### applying the quadratic transformation from Pathak et. al.
        # x = K.get_value(x)
        # for k in range(0, x.shape[-1], 2):
        #     # if k % 2 == 0:
        #         # K.set_value(x[:, :, k], tf.math.square(K.get_value(x[:, :, k])))
        #     x[:, :, k] = x[:, :, k]**2

        x = tf.math.pow(x, self.power_arr)

        x = self.post_res_periodic_padding(x)

        if manual_training == True:
            # simply return the hidden states, these are used for computing Wout
            output = x
        else:
            output = layers.TimeDistributed(self.Wout)(x) # DO NOT USE layers.TimeDistributed, dense layer takes care of it

        return output


    def save_model_weights(self, file_name, H5=True):

        # file_name = file_dir + '/' + file_name
        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        file_name += '.hdf5'
        # self.save_weights(file_name)
        
        f = h5py.File(file_name, 'w')
        ESN_cell_Win = f.create_group('ESN_cell_Win')
        ESN_cell_Wres = f.create_group('ESN_cell_Wres')
        ESN_net_Wout = f.create_group('ESN_net_Wout')

        ### saving Win
        usebias_Win = self.usebias_Win
        
        Win = self.rnn_list[0].cell.Win
        Win_kernel = Win.kernel

        Win_this_cell = ESN_cell_Win.create_group("cell_{}".format(0))
        
        Win_this_cell.create_dataset("kernel", data=Win_kernel.numpy())
        if usebias_Win == True:
            Win_bias = Win.bias
            Win_this_cell.create_dataset("bias", data=Win_bias.numpy())
            
        ### saving Wres
        Wres = self.rnn_list[0].cell.Wres
        Wres_kernel = Wres.kernel

        Wres_this_cell = ESN_cell_Wres.create_group("cell_{}".format(0))
        
        Wres_this_cell.create_dataset("kernel", data=Wres_kernel.numpy())
        Wres_this_cell.create_dataset("og_spectral_rad_Wres", data=self.rnn_list[0].cell.og_spectral_rad_Wres)

        ### saving Wout
        ESN_net_Wout.create_dataset("kernel", data=self.Wout.kernel.numpy())
        if self.usebias_Wout == True:
            ESN_net_Wout.create_dataset("bias", data=self.Wout.bias.numpy())

        f.close()

        return
    
    def save_class_dict(self, file_name):
        
        class_dict = {
            'kernel_size':self.kernel_size,
            'res_channels':self.res_channels,
            'data_dim':self.data_dim,
            'dt_rnn':self.dt_rnn,
            'lambda_reg':self.lambda_reg,
            'mean':self.mean,
            'stddev':self.stddev,
            'noise_type':self.noise_type,
            'stateful':self.stateful,
            'omega_in':self.omega_in,
            'rho_res':self.rho_res,
            'usebias_Win':self.usebias_Win,
            'prng_seed':self.prng_seed,
            'alpha':self.alpha,
            'ESN_cell_activations':self.ESN_cell_activations,
            'usebias_Wout':self.usebias_Wout,
        }
        with open(file_name, 'w') as f:
            f.write(str(class_dict))
            # s = '{\n'
            # for entry in class_dict.keys():
            #     s += '    '+str(entry)+':'+str(class_dict[entry])+',\n'
            # s += '}'
            # f.write(s)

    def save_everything(self, file_name, H5=True):

        ### saving class attributes
        self.save_class_dict(file_name+'_class_dict.txt')

        ### saving weights
        self.save_model_weights(file_name+'_ESN_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        # self.load_weights(file_name)

        if self.built == False:
            self.build((self.batch_size, None) + tuple(self.data_dim))
            
        f = h5py.File(file_name,'r')
        ESN_cell_Win = f['ESN_cell_Win']
        ESN_cell_Wres = f['ESN_cell_Wres']
        ESN_net_Wout = f['ESN_net_Wout']

        input_dim = self.data_dim
        ### reading Win
        usebias_Win = self.usebias_Win
        
        Win_this_cell = ESN_cell_Win["cell_{}".format(0)]

        Win_kernel = np.array(Win_this_cell["kernel"])

        Win = self.rnn_list[0].cell.Win
        K.set_value(Win.kernel, Win_kernel)
        if usebias_Win == True:
            Win_bias = np.array(Win_this_cell["bias"])
            K.set_value(Win.bias, Win_bias)
            
        ### reading Wres
        Wres_this_cell = ESN_cell_Wres["cell_{}".format(0)]

        Wres_kernel = np.array(Wres_this_cell["kernel"])

        Wres = self.rnn_list[0].cell.Wres
        K.set_value(Wres.kernel, Wres_kernel)
        
        self.rnn_list[0].cell.og_spectral_rad_Wres = np.float32(np.array(Wres_this_cell["og_spectral_rad_Wres"]))
        

        ### reading Wout
        K.set_value(self.Wout.kernel, np.array(ESN_net_Wout['kernel']))
        if self.usebias_Wout == True:
            K.set_value(self.Wout.bias, np.array(ESN_net_Wout['bias']))
        
        f.close()
        
        return


################################################################################
