################################################################################
# RK Methods inspired residual ESN with skip connections, with uniform/normal  #
# noise added to every input and learnable initial states. [STATEFUL]          #
# Using sparse tensors to make storage easier and more efficient.              #
#------------------------------------------------------------------------------#
#                        Basic Network Architecture                            #
#------------------------------------------------------------------------------#
#                                                                              #
#                         ______________________________________               #
#                        /                  _\ _________________\          z1  #
#                       /                  /  \        z1        \    +a13*d1  #
#                      /            z1    /    \  +a12*d1         \   +a23*d2  #
#         __   z1   __/  d1    +a11*d1 __/  d2  \ +a22*d2 __   d3  \  +a33*d3  #
# u----->|__|----->|__|----->[+]----->|__|----->[+]----->|__|----->[+]----->   #
#           \________________/                  /                  /           #
#            \_________________________________/                  /            #
#             \__________________________________________________/             #
#                                                                              #
# (a1, a2 and a3 are scalars that determine a weighted average and sum to `dt`)#
#                                                                              #
# Note here that you can only specify the number of layers and the number of   #
# units in a layer, not the number of units in each layer individually. Also,  #
# a single layer network is the same as a regular GRU.                         #
#                                                                              #
# The RNN weights are shared amongst the 2nd, 3rd,... networks. Need to        #
# provide `dt` to the class, so the learned scalars can sum to `dt` at each    #
# layer (in the case that one is learning them, as opposed to providing them   #
# outright).                                                                   #
#                                                                              #
# Wres & Win are used as a sparse tensor (as opposed to a dense layer).        #
#                                                                              #
# ESN_v6 in og KS dir                                                          #
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

    def compute_output_shape(self, input_shape):
        return input_shape

class ESN_Cell(layers.Layer):
    def __init__(
            self, omega_in, sparsity, rho_res, state_size, alpha=1.0,
            usebias_Win=False, prng_seed=42, activation='tanh', **kwargs):

        super(ESN_Cell, self).__init__()

        self.omega_in = omega_in
        self.sparsity = sparsity
        self.rho_res = rho_res
        self.input_size = None
        self.state_num_units = state_size
        self.alpha = alpha
        self.usebias_Win = usebias_Win
        self.prng = np.random.default_rng(seed=prng_seed)
        self.activation = eval('tf.keras.activations.'+activation)
        
        self.Win_sp = None
        self.Win_bias = None
        self.Wres_sp = self._build_Wres()
        
        return

    def _build_Win(self):
        
        shape = (self.input_size, self.state_num_units)
        
        Win_row_idxs = self.prng.integers(low=0, high=self.input_size, size=self.state_num_units)
        Win_col_idxs = np.arange(0, self.state_num_units)
        Win_data = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=self.state_num_units).astype('float32', copy=False)
        Win_sp = tf.sparse.SparseTensor(
            indices=[elem for elem in zip(Win_row_idxs, Win_col_idxs)],
            values=Win_data,
            dense_shape=shape,
        )
        if self.usebias_Win == True:
            Win_bias = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=self.state_num_units).astype('float32', copy=False)
            return_tuple = (Win_sp, tf.constant(Win_bias))
        else:
            return_tuple = (Win_sp, None)

        return return_tuple

    def _build_Wres(self):
    
        shape = (self.state_num_units, self.state_num_units)
        
        Wres = self.prng.random(size=shape) < 1 - self.sparsity # really just the mask for Wres
        Wres = sp.csr_array(Wres, dtype=np.float32)
        Wres.data = self.prng.uniform(low=-1, high=1, size=Wres.data.shape).astype('float32', copy=False)
        spectral_rad = np.abs(sp_la.eigs(Wres, k=1, which='LM', return_eigenvectors=False))[0]
        
        Wres.data *= self.rho_res/spectral_rad
        Wres = Wres.tocoo()
        Wres = tf.sparse.SparseTensor(
            indices=[elem for elem in zip(Wres.row, Wres.col)],
            values=Wres.data,
            dense_shape=Wres.shape,
        )

        return Wres

    def build(self, input_shape):
    
        self.input_size = input_shape[-1]

        self.Win_sp, self.Win_bias = self._build_Win()
        if self.usebias_Win == True:
            self.Win_bias = tf.reshape(self.Win_bias, (1, self.Win_bias.shape[0]))
        
        super(ESN_Cell, self).build(input_shape)

    @tf.function
    def call(self, inputs, states):

        state = states[0]
        
        candidate_state = tf.sparse.sparse_dense_matmul(inputs, self.Win_sp)
        if self.usebias_Win == True:
            candidate_state = candidate_state + self.Win_bias
        candidate_state = candidate_state + tf.sparse.sparse_dense_matmul(state, self.Wres_sp)
        candidate_state = self.activation(candidate_state)
        
        # computing new_state as a weighted average of old state and candidate state
        new_state = state + self.alpha * (candidate_state - state)

        return new_state, [new_state]

    @property
    def state_size(self):
        return [self.state_num_units]

    @property
    def output_size(self):
        return self.state_num_units

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     batch_size = inputs.shape[0] if inputs.shape[0] != None else 1
    #     return tf.zeros(shape=(batch_size, inputs.shape[-1]), dtype='float32')

        
        


class ESN(Model):
    """
    Single-step ESN network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self, data_dim=None,
            dt_rnn=None,
            lambda_reg=0.0,
            ESN_layers_units=[1000],
            load_file=None,
            stddev=0.0,
            mean=0.0,
            noise_type='uniform',
            stateful=True,
            omega_in=[0.5],
            sparsity=[0.995],
            rho_res=[0.6],
            usebias_Win=[False],
            alpha=[1.],
            ESN_cell_activations=['tanh'],
            prng_seed=42,
            usebias_Wout=False,
            batch_size=1,
            activation_post_Wout='linear',
            use_weights_post_dense=False,
            scalar_weights=None,
            **kwargs,
            ):
        
        super(ESN, self).__init__()

        self.load_file = load_file
        self.data_dim = data_dim
        self.dt_rnn = dt_rnn
        self.lambda_reg = lambda_reg
        self.ESN_layers_units = ESN_layers_units
        self.mean = mean
        self.stddev = stddev
        self.noise_type = noise_type
        self.stateful = stateful
        self.omega_in = omega_in
        self.sparsity = sparsity
        self.rho_res = rho_res
        self.usebias_Win = usebias_Win
        self.prng_seed = prng_seed
        self.alpha = alpha
        self.ESN_cell_activations = ESN_cell_activations
        self.usebias_Wout = usebias_Wout
        self.batch_size = batch_size
        self.activation_post_Wout = activation_post_Wout
        self.use_weights_post_dense = use_weights_post_dense
        self.scalar_weights = scalar_weights
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
            if 'ESN_layers_units' in load_dict.keys():
                self.ESN_layers_units = load_dict['ESN_layers_units']
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
            if 'sparsity' in load_dict.keys():
                self.sparsity = load_dict['sparsity']
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
            if 'activation_post_Wout' in load_dict.keys():
                self.activation_post_Wout = load_dict['activation_post_Wout']
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'scalar_weights' in load_dict.keys():
                self.scalar_weights = load_dict['scalar_weights']
            
        self.num_rnn_layers = len(self.ESN_layers_units)
        self.num_skip_connections = self.num_rnn_layers - 1

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
        self.hidden_states_list = []
        self.rnn_list = [
            layers.RNN(
                cell=ESN_Cell(
                    omega_in=self.omega_in[0],
                    sparsity=self.sparsity[0],
                    rho_res=self.rho_res[0],
                    state_size=self.ESN_layers_units[0],
                    alpha=self.alpha[0],
                    usebias_Win=self.usebias_Win[0],
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations[0],
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            )
        ]
        if self.num_skip_connections > 0:
            self.RK_RNNCell = ESN_Cell(
                    omega_in=self.omega_in[1],
                    sparsity=self.sparsity[1],
                    rho_res=self.rho_res[1],
                    state_size=self.ESN_layers_units[1],
                    alpha=self.alpha[1],
                    usebias_Win=self.usebias_Win[1],
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations[1],
                )
            self.rnn_list.extend([
                layers.RNN(
                    self.RK_RNNCell,
                    return_sequences=True,
                    stateful=self.stateful,
                    batch_size=self.batch_size if self.stateful else None,
                ) for i in range(self.num_skip_connections)
            ])
        
        
        ### adding the Wout layer
        reg = lambda x : None
        if self.lambda_reg != None:
            if self.lambda_reg > 0.0:
                reg = lambda x : tf.keras.regularizers.L2(x)
        self.Wout = layers.Dense(
            self.data_dim,
            use_bias=self.usebias_Wout,
            trainable=True,
            dtype='float32',
            name='Wout',
            kernel_regularizer=reg(self.lambda_reg),
            bias_regularizer=reg(self.lambda_reg),
        )
        self.post_Wout_activation = tf.keras.activations.get(self.activation_post_Wout)
        
        if self.use_weights_post_dense == True:
            self.postWout = single_weights(w_regularizer=reg(self.lambda_reg))

        self.power_arr = np.ones(shape=self.ESN_layers_units[-1])
        self.power_arr[0::2] = 2

        
        if self.num_skip_connections > 0:
            if type(self.scalar_weights) == type(None):
                self.scalar_multiplier_pre_list = []
                for i in range(self.num_skip_connections):
                    for j in range(i+1):
                        self.scalar_multiplier_pre_list.append(
                            tf.Variable(
                                initial_value=1.0,
                                dtype='float32',
                                trainable=True,
                                name='a_{}{}'.format(j+1, i+1) # this naming convention follows the one in the diagram at the top of this script
                            )
                        )
                # self.scalar_multiplier_pre_list = [
                #     tf.Variable(
                #         initial_value=1.0,
                #         dtype='float32',
                #         trainable=True,
                #     ) for i in range(int(0.5*self.num_skip_connections*(self.num_skip_connections+1)))
                # ]
            else:
                self.scalar_weights = np.array(self.scalar_weights, dtype='float32')


        self.ESN_layers = self.rnn_list
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
    def _callhelperfn(self, x, scalar_multiplier_list):
        intermediate_outputs_lst = []
        x = self.rnn_list[0](
            x,
            # training=training,
        )
        intermediate_outputs_lst.append(x)
        for i in range(self.num_skip_connections):
            prediction = self.rnn_list[i+1](
                x,
                # training=training,
            )
            intermediate_outputs_lst.append(prediction)
            x = intermediate_outputs_lst[0]
            for j in range(i+1):
                x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]
        return x

    @tf.function
    def call(self, inputs, training=False, manual_training=False):

        scalar_multiplier_list = self.scalar_weights
        # inputs shape : (None, time_steps, data_dim)
        out_steps = inputs.shape[1]

        ### Running through the ESN layers
        x = inputs
        if training == True or manual_training == True:
            # add noise to the inputs during training
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        x = self._callhelperfn(x, scalar_multiplier_list)


        x = tf.math.pow(x, self.power_arr)

        if manual_training == True:
            # simply return the hidden states, these are used for computing Wout
            output = x
        else:
            output = self.Wout(x) # DO NOT USE layers.TimeDistributed, dense layer takes care of it
            output = self.post_Wout_activation(output)
            if self.use_weights_post_dense == True:
                output = self.postWout(output) # DO NOT USE layers.TimeDistributed, single_weights layer takes care of it

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

        for i in range(len(self.rnn_list)):
            ### saving Win in a sparse manner
            usebias_Win = self.usebias_Win[i]

            Win_kernel_sparse = self.rnn_list[i].cell.Win_sp

            Win_this_cell = ESN_cell_Win.create_group("cell_{}".format(i))
            
            Win_this_cell.create_dataset("kernel_values", data=Win_kernel_sparse.values.numpy())
            Win_this_cell.create_dataset("kernel_indices", data=Win_kernel_sparse.indices.numpy())
            if usebias_Win == True:
                Win_bias = self.rnn_list[i].cell.Win_bias
                Win_this_cell.create_dataset("bias", data=Win_bias.numpy())
                
            ### saving Wres in a sparse manner
            Wres_kernel_sparse = self.rnn_list[i].cell.Wres_sp

            Wres_this_cell = ESN_cell_Wres.create_group("cell_{}".format(i))
            
            Wres_this_cell.create_dataset("kernel_values", data=Wres_kernel_sparse.values.numpy())
            Wres_this_cell.create_dataset("kernel_indices", data=Wres_kernel_sparse.indices.numpy())

        ### saving Wout
        ESN_net_Wout.create_dataset("kernel", data=self.Wout.kernel.numpy())
        if self.usebias_Wout == True:
            ESN_net_Wout.create_dataset("bias", data=self.Wout.bias.numpy())

        if self.use_weights_post_dense == True:
            ESN_net_postWout = f.create_group('ESN_net_postWout') 
            ESN_net_postWout.create_dataset("individual_weights", data=self.postWout.individual_weights.numpy())
        
        f.close()

        return
    
    def save_class_dict(self, file_name):
        
        class_dict = {
            'data_dim':self.data_dim,
            'dt_rnn':self.dt_rnn,
            'lambda_reg':self.lambda_reg,
            'ESN_layers_units':list(self.ESN_layers_units),
            'mean':self.mean,
            'stddev':self.stddev,
            'noise_type':self.noise_type,
            'stateful':self.stateful,
            'omega_in':list(self.omega_in),
            'sparsity':list(self.sparsity),
            'rho_res':list(self.rho_res),
            'usebias_Win':list(self.usebias_Win),
            'prng_seed':self.prng_seed,
            'alpha':list(self.alpha),
            'ESN_cell_activations':list(self.ESN_cell_activations),
            'usebias_Wout':self.usebias_Wout,
            'activation_post_Wout':self.activation_post_Wout,
            'use_weights_post_dense':self.use_weights_post_dense,
            'scalar_weights':list(self.scalar_weights),
        }
        with open(file_name, 'w') as f:
            f.write(str(class_dict))
            # s = '{\n'
            # for entry in class_dict.keys():
            #     s += '    '+str(entry)+':'+str(class_dict[entry])+',\n'
            # s += '}'
            # f.write(s)

    def save_everything(self, file_name, suffix='', H5=True):

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
            self.build((self.batch_size, None, self.data_dim))
            
        f = h5py.File(file_name,'r')
        ESN_cell_Win = f['ESN_cell_Win']
        ESN_cell_Wres = f['ESN_cell_Wres']
        ESN_net_Wout = f['ESN_net_Wout']

        input_dim = self.data_dim
        for i in range(len(self.rnn_list)):
            ### reading Win, stored in a sparse manner
            usebias_Win = self.usebias_Win[i]
            
            Win_this_cell = ESN_cell_Win["cell_{}".format(i)]

            Win_kernel_sparse_values = np.array(Win_this_cell["kernel_values"])
            Win_kernel_sparse_indices = np.array(Win_this_cell["kernel_indices"])
            self.rnn_list[i].cell.Win_sp = tf.sparse.SparseTensor(
                indices=Win_kernel_sparse_indices,
                values=Win_kernel_sparse_values,
                dense_shape=[input_dim, self.ESN_layers_units[i]])
            if usebias_Win == True:
                self.rnn_list[i].cell.Win_bias =  tf.constant(np.array(Win_this_cell["bias"]))
                
            ### reading Wres, stored in a sparse manner
            Wres_this_cell = ESN_cell_Wres["cell_{}".format(i)]
            input_dim =  self.ESN_layers_units[i]

            Wres_kernel_sparse_values = np.array(Wres_this_cell["kernel_values"])
            Wres_kernel_sparse_indices = np.array(Wres_this_cell["kernel_indices"])
            self.rnn_list[i].cell.Wres_sp = tf.sparse.SparseTensor(
                indices=Wres_kernel_sparse_indices,
                values=Wres_kernel_sparse_values,
                dense_shape=[input_dim, input_dim])

        ### reading Wout
        K.set_value(self.Wout.kernel, np.array(ESN_net_Wout['kernel']))
        if self.usebias_Wout == True:
            K.set_value(self.Wout.bias, np.array(ESN_net_Wout['bias']))

        if self.use_weights_post_dense == True:
            ESN_net_postWout = f["ESN_net_postWout"]
            K.set_value(self.postWout.individual_weights, np.array(ESN_net_postWout["individual_weights"]))
        
        f.close()
        
        return


################################################################################
