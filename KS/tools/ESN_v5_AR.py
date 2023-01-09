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
################################################################################

import os
import numpy as np
from scipy import linalg

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

class ESN_Cell_trainableparams(layers.Layer):
    def __init__(
            self, omega_in, sparsity, rho_res, state_size, alpha=1.0,
            usebias_Win=False, prng_seed=42, activation='tanh', train_alpha=True,
            train_rho_res=True, train_omega_in=True, **kwargs):

        super(ESN_Cell_trainableparams, self).__init__()

        self.train_omega_in = train_omega_in
        self.omega_in_original = omega_in
        self.omega_in = tf.Variable(
            initial_value=omega_in,
            trainable=self.train_omega_in,
            name='omega_in',
            dtype='float32',
        )

        self.sparsity = sparsity
        
        self.train_rho_res = train_rho_res
        self.rho_res_original = rho_res
        self.rho_res = tf.Variable(
            initial_value=rho_res,
            trainable=self.train_rho_res,
            name='rho_res',
            dtype='float32',
            constraint=lambda t : tf.clip_by_value(t, 1e-3, 1.-1e-3)
        )
        self.input_size = None
        self.state_num_units = state_size
        
        self.train_alpha = train_alpha
        self.alpha_original = alpha
        self.alpha = tf.Variable(
            initial_value=alpha,
            trainable=self.train_alpha,
            name='alpha',
            dtype='float32',
            constraint=lambda t : tf.clip_by_value(t, 1e-3, 1.)
        )
        self.usebias_Win = usebias_Win
        self.prng = np.random.default_rng(seed=prng_seed)
        self.activation = eval('tf.keras.activations.'+activation)
        
        self.Win_np = None
        self.Win_bias_np = None
        self.Wres_np = self._build_Wres()
        
        return

    def _build_Win(self):
        
        shape = (self.input_size, self.state_num_units)
        # if self.usebias_Win == True:
        #     # last row of the matrix Win corresponds to the bias
        #     shape[0] += 1

        ### this is a sparse Win, with only one element in each column having a non-zero value
        Win = np.zeros(shape=shape, dtype='float32')
        for i in range(shape[1]):
            Win[self.prng.integers(low=0, high=self.input_size, size=1), i] = self.prng.uniform(low=-1, high=1)
        if self.usebias_Win == True:
            Win_bias = self.prng.uniform(low=-1, high=1, size=self.state_num_units)
            return_tuple = (Win.astype('float32'), Win_bias.astype('float32'))
        else:
            return_tuple = (Win.astype('float32'), None)
        ### this is a dense Win
        # Win = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=shape)

        return return_tuple

    def _build_Wres(self):
    
        shape = (self.state_num_units, self.state_num_units)
        
        Wres = self.prng.uniform(low=-1, high=1, size=shape)
        Wres *= (self.prng.random(size=shape) < 1 - self.sparsity)
        
        fac = np.max(np.abs(np.linalg.eigvals(Wres)))
        Wres /= fac
        
        return Wres.astype('float32')

    def build(self, input_shape):
    
        self.input_size = input_shape[-1]

        self.Win_np, self.Win_bias_np = self._build_Win()
        self.Win = layers.Dense(
            self.state_num_units,
            use_bias=self.usebias_Win,
            dtype='float32',
            name='Win',
            trainable=False,
        )
        self.Win.build(input_shape=(input_shape[0], self.input_size))
        K.set_value(self.Win.kernel, self.Win_np)
        if self.usebias_Win == True:
            K.set_value(self.Win.bias, self.Win_bias_np)
        self.Win.trainable = False
        
        
        self.Wres = layers.Dense(
            self.state_num_units,
            use_bias=False,
            dtype='float32',
            name='Wres',
            trainable=False,
        )
        self.Wres.build(input_shape=(input_shape[0], self.state_num_units))
        K.set_value(self.Wres.kernel, self.Wres_np)
        self.Wres.trainable = False
        
        super(ESN_Cell_trainableparams, self).build(input_shape)

    # @tf.function
    def call(self, inputs, states):

        # print(type(states), len(states), type(states[0]), len(states[0]), states[0])
        state = states[0]
        
        candidate_state = self.activation(self.rho_res * self.Wres(state) + self.omega_in * self.Win(inputs))
        
        # computing new_state as a weighted average of old state and candidate state
        new_state = state + self.alpha * (candidate_state - state)
        
        ### applying the quadratic transformation from Pathak et. al.
        # r = new_state[:, 0::2] + new_state[:, 1::2]

        # output = tf.linalg.matmul(r, self.Wout[0:self.output_size])
        # if usebias_Wout == True:
        #     output = output + tf.tile(self.Wout[-1], [inputs.shape[0], 1])

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
            stateful=False,
            omega_in=[0.5],
            sparsity=[0.995],
            rho_res=[0.6],
            usebias_Win=[False],
            alpha=[1.],
            ESN_cell_activations=['tanh'],
            prng_seed=42,
            usebias_Wout=False,
            batch_size=1,
            T_input=None,
            T_output=None,
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
        self.T_input = T_input
        self.T_output = T_output
        self.activation_post_Wout = activation_post_Wout
        self.use_weights_post_dense = use_weights_post_dense
        self.train_alpha = kwargs.pop('train_alpha', None)
        self.train_omega_in = kwargs.pop('train_omega_in', None)
        self.train_rho_res = kwargs.pop('train_rho_res', None)
        self.scalar_weights = scalar_weights
        if self.load_file is not None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            if 'data_dim' in load_dict.keys():
                self.data_dim = load_dict['data_dim']
            if 'dt_rnn' in load_dict.keys():
                self.dt_rnn = load_dict['dt_rnn']
            # if 'lambda_reg' in load_dict.keys():
            #     self.lambda_reg = load_dict['lambda_reg']
            if 'ESN_layers_units' in load_dict.keys():
                self.ESN_layers_units = load_dict['ESN_layers_units']
            if 'mean' in load_dict.keys():
                self.mean = load_dict['mean']
            if 'stddev' in load_dict.keys():
                self.stddev = load_dict['stddev']
            if 'noise_type' in load_dict.keys():
                self.noise_type = load_dict['noise_type']
            # if 'stateful' in load_dict.keys():
            #     self.stateful = load_dict['stateful']
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
            if 'out_steps' in load_dict.keys():
                self.out_steps = int(load_dict['out_steps'])
            if 'in_steps' in load_dict.keys():
                self.in_steps = int(load_dict['in_steps'])
            if 'activation_post_Wout' in load_dict.keys():
                self.activation_post_Wout = load_dict['activation_post_Wout']
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'train_alpha' in load_dict.keys():
                self.train_alpha = load_dict['train_alpha']
            if 'train_rho_res' in load_dict.keys():
                self.train_rho_res = load_dict['train_rho_res']
            if 'train_omega_in' in load_dict.keys():
                self.train_omega_in = load_dict['train_omega_in']
            if 'scalar_weights' in load_dict.keys():
                self.scalar_weights = load_dict['scalar_weights']
            
        self.num_rnn_layers = len(self.ESN_layers_units)
        self.num_skip_connections = self.num_rnn_layers - 1
        
        ### time steps
        if T_input is not None:
            self.in_steps = int((self.T_input+0.5*self.dt_rnn)//self.dt_rnn)
        if T_output is not None:
            self.out_steps = int((self.T_output+0.5*self.dt_rnn)//self.dt_rnn)
        if stddev is None:
            if self.stddev is None:
                self.stddev = 0.0
        else:
            self.stddev = stddev
        if mean is None:
            if self.mean is None:
                self.mean = 0.0
        else:
            self.mean = mean
        
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

        if self.train_alpha == None:
            self.train_alpha = [True]*len(self.ESN_layers_units)
        if self.train_omega_in == None:
            self.train_omega_in = [True]*len(self.ESN_layers_units)
        if self.train_rho_res == None:
            self.train_rho_res = [True]*len(self.ESN_layers_units)

        ### the ESN network
        self.hidden_states_list = []
        self.rnn_list = [
            layers.RNN(
                cell=ESN_Cell_trainableparams(
                    omega_in=self.omega_in[0],
                    sparsity=self.sparsity[0],
                    rho_res=self.rho_res[0],
                    state_size=self.ESN_layers_units[0],
                    alpha=self.alpha[0],
                    usebias_Win=self.usebias_Win[0],
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations[0],
                    train_alpha=self.train_alpha[0],
                    train_rho_res=self.train_rho_res[0],
                    train_omega_in=self.train_omega_in[0],
                ),
                return_sequences=True,
                return_state=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            )
        ]
        if self.num_skip_connections > 0:
            self.RK_RNNCell = ESN_Cell_trainableparams(
                    omega_in=self.omega_in[1],
                    sparsity=self.sparsity[1],
                    rho_res=self.rho_res[1],
                    state_size=self.ESN_layers_units[1],
                    alpha=self.alpha[1],
                    usebias_Win=self.usebias_Win[1],
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations[1],
                    train_alpha=self.train_alpha[1],
                    train_rho_res=self.train_rho_res[1],
                    train_omega_in=self.train_omega_in[1],
                )
            self.rnn_list.extend([
                layers.RNN(
                    self.RK_RNNCell,
                    return_sequences=True,
                    return_state=True,
                    stateful=self.stateful,
                    batch_size=self.batch_size if self.stateful else None,
                ) for i in range(self.num_skip_connections)
            ])
        
        
        ### adding the Wout layer
        reg = lambda x : None
        if self.lambda_reg is not None:
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
        
    #     super().build(input_shape)
    #     self.built = True

    def _warmup(
            self,
            inputs,
            training=None,
            usenoiseflag=False,
            # scalar_multiplier_list=None,
            **kwargs):
        ### Initialize the ESN state.
        states_list = []
        intermediate_outputs_lst = []
        scalar_multiplier_list = self.scalar_weights
        
        x = inputs
        if training == True or usenoiseflag == True:
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        
        x, states = self.rnn_list[0](
            x,
            # training=training,
        )
        states_list.append(states)
        intermediate_outputs_lst.append(x)
        for i in range(self.num_skip_connections):
            prediction, states = self.rnn_list[i+1](
                x,
                # training=training,
            )
            states_list.append(states)
            intermediate_outputs_lst.append(prediction)
            x = intermediate_outputs_lst[0]
            for j in range(i+1):
                x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]

        output = x[:, -1:, :]
        output = tf.math.pow(output, self.power_arr)
        output = self.Wout(output, training=training) # DO NOT USE layers.TimeDistributed, dense layer takes care of it
        output = self.post_Wout_activation(output)
        if self.use_weights_post_dense == True:
            output = self.postWout(output) # DO NOT USE layers.TimeDistributed, single_weights layer takes care of it

        return output, states_list, intermediate_outputs_lst, scalar_multiplier_list
        
    def onestep(
            self,
            x,
            training=False,
            intermediate_outputs_lst=[],
            states_list=[],
            scalar_multiplier_list=[],
            **kwargs):
        ### Passing input through the ESN layers
        x, states = self.rnn_list[0](
            x,
            initial_state=states_list[0],
            # training=training,
        )
        states_list[0] = states
        intermediate_outputs_lst[0] = x
        for i in range(self.num_skip_connections):
            prediction, states = self.rnn_list[i+1](
                x,
                initial_state=states_list[i+1],
                # training=training,
            )
            states_list[i+1] = states
            intermediate_outputs_lst[i+1] = prediction
            x = intermediate_outputs_lst[0]
            for j in range(i+1):
                x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]

        output = x[:, -1:, :]
        output = tf.math.pow(output, self.power_arr)
        output = self.Wout(output, training=training) # DO NOT USE layers.TimeDistributed, dense layer takes care of it
        output = self.post_Wout_activation(output)
        if self.use_weights_post_dense == True:
            output = self.postWout(output) # DO NOT USE layers.TimeDistributed, single_weights layer takes care of it

        return output, states_list

    def call(self, inputs, training=False, usenoiseflag=False):

        scalar_multiplier_list = self.scalar_weights
        predictions_list = []

        ### warming up the RNN
        x, states_list, intermediate_outputs_lst, scalar_multiplier_list = self._warmup(
            inputs,
            training=False,
            usenoiseflag=usenoiseflag,
        )
        predictions_list.append(x[:, -1, :])
        # print(type(states_list), len(states_list), type(states_list[0]), states_list[0])


        ### Passing input through the GRU layers
        for tstep in range(1, self.out_steps):
            x, states_list = self.onestep(
                x=x,
                training=training,
                states_list=states_list,
                intermediate_outputs_lst=intermediate_outputs_lst,
                scalar_multiplier_list=scalar_multiplier_list,
            )
            predictions_list.append(x[:, -1, :])
            # print(type(states_list), len(states_list), type(states_list[0]), states_list[0])

        output = tf.stack(predictions_list)
        output = tf.transpose(output, [1, 0, 2])

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
        ESN_cell_rho_res = f.create_group('ESN_cell_rho_res')
        ESN_cell_omega_in = f.create_group('ESN_cell_omega_in')
        ESN_cell_alpha = f.create_group('ESN_cell_alpha')
        ESN_net_Wout = f.create_group('ESN_net_Wout')

        for i in range(len(self.rnn_list)):
            ### saving Win in a sparse manner
            usebias_Win = self.usebias_Win[i]
            
            omega_in = self.rnn_list[i].cell.omega_in.numpy()
            Win = self.rnn_list[i].cell.Win
            Win_kernel_sparse = tf.sparse.from_dense(Win.kernel * omega_in)

            Win_this_cell = ESN_cell_Win.create_group("cell_{}".format(i))
            
            Win_this_cell.create_dataset("kernel_values", data=Win_kernel_sparse.values.numpy())
            Win_this_cell.create_dataset("kernel_indices", data=Win_kernel_sparse.indices.numpy())
            if usebias_Win == True:
                Win_bias = Win.bias * omega_in
                Win_this_cell.create_dataset("bias", data=Win_bias.numpy())
                
            ### saving Wres in a sparse manner
            rho_res = self.rnn_list[i].cell.rho_res.numpy()
            Wres = self.rnn_list[i].cell.Wres
            Wres_kernel_sparse = tf.sparse.from_dense(Wres.kernel * rho_res)

            Wres_this_cell = ESN_cell_Wres.create_group("cell_{}".format(i))
            
            Wres_this_cell.create_dataset("kernel_values", data=Wres_kernel_sparse.values.numpy())
            Wres_this_cell.create_dataset("kernel_indices", data=Wres_kernel_sparse.indices.numpy())
            
            rho_res_this_cell = ESN_cell_rho_res.create_group("cell_{}".format(i))
            rho_res_this_cell.create_dataset("rho_res", data=rho_res)
            
            omega_in_this_cell = ESN_cell_omega_in.create_group("cell_{}".format(i))
            omega_in_this_cell.create_dataset("omega_in", data=omega_in)
            
            alpha_this_cell = ESN_cell_alpha.create_group("cell_{}".format(i))
            alpha_this_cell.create_dataset("alpha", data=self.rnn_list[i].cell.alpha.numpy())

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
        
        self.alpha = np.array([elem.cell.alpha.numpy() for elem in self.rnn_list])
        self.rho_res = np.array([elem.cell.rho_res.numpy() for elem in self.rnn_list])
        self.omega_in = np.array([elem.cell.omega_in.numpy() for elem in self.rnn_list])

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
            'in_steps':self.in_steps,
            'out_steps':self.out_steps,
            'activation_post_Wout':self.activation_post_Wout,
            'use_weights_post_dense':self.use_weights_post_dense,
            'train_alpha':list(self.train_alpha),
            'train_omega_in':list(self.train_omega_in),
            'train_rho_res':list(self.train_rho_res),
            'scalar_weights':list(self.scalar_weights),
        }
        with open(file_name, 'w') as f:
            f.write(str(class_dict))
            # s = '{\n'
            # for entry in class_dict.keys():
            #     s += '    '+str(entry)+':'+str(class_dict[entry])+',\n'
            # s += '}'
            # f.write(s)

    def save_everything(self, file_name, H5=True):

        ### saving weights
        self.save_model_weights(file_name+'_ESN_weights', H5=H5)

        ### saving class attributes
        self.save_class_dict(file_name+'_class_dict.txt')

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
        
        cell_rho_res_flag = False
        if 'ESN_cell_rho_res' in f.keys():
            cell_rho_res_flag = True
        cell_omega_in_flag = False
        if 'ESN_cell_omega_in' in f.keys():
            cell_omega_in_flag = True
        cell_alpha_flag = False
        if 'ESN_cell_alpha' in f.keys():
            cell_alpha_flag = True

        input_dim = self.data_dim
        for i in range(len(self.rnn_list)):
            if cell_rho_res_flag == True:
                rho_res_this_cell = np.array(f["ESN_cell_rho_res"]["cell_{}".format(i)]['rho_res'])
                K.set_value(self.rnn_list[i].cell.rho_res, rho_res_this_cell)
            if cell_omega_in_flag == True:
                omega_in_this_cell = np.array(f["ESN_cell_omega_in"]["cell_{}".format(i)]['omega_in'])
                K.set_value(self.rnn_list[i].cell.omega_in, omega_in_this_cell)
            if cell_alpha_flag == True:
                alpha_this_cell = np.array(f["ESN_cell_alpha"]["cell_{}".format(i)]['alpha'])
                K.set_value(self.rnn_list[i].cell.alpha, alpha_this_cell)
        
            ### reading Win, stored in a sparse manner
            usebias_Win = self.usebias_Win[i]
            
            Win_this_cell = ESN_cell_Win["cell_{}".format(i)]

            Win_kernel_sparse_values = np.array(Win_this_cell["kernel_values"])
            Win_kernel_sparse_indices = np.array(Win_this_cell["kernel_indices"])
            Win_kernel = tf.sparse.SparseTensor(
                indices=Win_kernel_sparse_indices,
                values=Win_kernel_sparse_values,
                dense_shape=[input_dim, self.ESN_layers_units[i]])
            input_dim =  self.ESN_layers_units[i]
            Win_kernel = tf.sparse.to_dense(Win_kernel)
            Win_kernel = Win_kernel.numpy()

            Win = self.rnn_list[i].cell.Win
            K.set_value(Win.kernel, Win_kernel / self.rnn_list[i].cell.omega_in_original)
            if usebias_Win == True:
                Win_bias = np.array(Win_this_cell["bias"])
                K.set_value(Win.bias, Win_bias / self.rnn_list[i].cell.omega_in_original)
                
            ### reading Wres, stored in a sparse manner
            Wres_this_cell = ESN_cell_Wres["cell_{}".format(i)]

            Wres_kernel_sparse_values = np.array(Wres_this_cell["kernel_values"])
            Wres_kernel_sparse_indices = np.array(Wres_this_cell["kernel_indices"])
            Wres_kernel = tf.sparse.SparseTensor(
                indices=Wres_kernel_sparse_indices,
                values=Wres_kernel_sparse_values,
                dense_shape=[input_dim, input_dim])
            Wres_kernel = tf.sparse.to_dense(Wres_kernel)
            Wres_kernel = Wres_kernel.numpy()

            Wres = self.rnn_list[i].cell.Wres
            K.set_value(Wres.kernel, Wres_kernel / self.rnn_list[i].cell.rho_res_original)

        ### reading Wout
        K.set_value(self.Wout.kernel, np.array(ESN_net_Wout['kernel']))
        if self.usebias_Wout == True:
            K.set_value(self.Wout.bias, np.array(ESN_net_Wout['bias']))
        
        ### reading
        if self.use_weights_post_dense == True:
            ESN_net_postWout = f["ESN_net_postWout"]
            K.set_value(self.postWout.individual_weights, np.array(ESN_net_postWout["individual_weights"]))
        
        f.close()
        
        self.alpha = np.array([elem.cell.alpha.numpy() for elem in self.rnn_list])
        self.rho_res = np.array([elem.cell.rho_res.numpy() for elem in self.rnn_list])
        self.omega_in = np.array([elem.cell.omega_in.numpy() for elem in self.rnn_list])
        
        return


################################################################################
