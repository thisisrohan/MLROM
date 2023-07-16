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
# ESN_v5 in og KS dir                                                          #
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
        self.wts_to_be_loaded = kwargs.pop('wts_to_be_loaded', False)

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
            Win[self.prng.integers(low=0, high=self.input_size, size=1), i] = self.prng.uniform(low=-self.omega_in, high=self.omega_in)
        if self.usebias_Win == True:
            Win_bias = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=self.state_num_units)
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
        
        if self.wts_to_be_loaded == False:
            fac = self.rho_res / np.max(np.abs(np.linalg.eigvals(Wres)))
            Wres *= fac
        
        return Wres.astype('float32')

    def build(self, input_shape):
    
        self.input_size = input_shape[-1]

        self.Win_np, self.Win_bias_np = self._build_Win()
        self.Win = layers.Dense(
            self.state_num_units,
            use_bias=self.usebias_Win,
            dtype='float32',
            name='Win',
        )
        self.Win.build(input_shape=(input_shape[0], None, self.input_size))
        K.set_value(self.Win.kernel, self.Win_np)
        if self.usebias_Win == True:
            K.set_value(self.Win.bias, self.Win_bias_np)
        self.Win.trainable = False
        
        
        self.Wres = layers.Dense(
            self.state_num_units,
            use_bias=False,
            dtype='float32',
            name='Wres',
        )
        self.Wres.build(input_shape=(input_shape[0], self.state_num_units, self.state_num_units))
        K.set_value(self.Wres.kernel, self.Wres_np)
        self.Wres.trainable = False
        
        super(ESN_Cell, self).build(input_shape)

    @tf.function
    def call(self, inputs, states):

        state = states[0]
        
        candidate_state = self.activation(self.Wres(state) + self.Win(inputs))
        
        # computing new_state as a weighted average of old state and candidate state
        new_state = state + self.alpha * (candidate_state - state)
        
        return new_state, [new_state]

    @property
    def state_size(self):
        return [self.state_num_units]

    @property
    def output_size(self):
        return self.state_num_units
        
        
def check_if_enumerable(test_obj):
    try:
        for elem in test_obj:
            pass
        return True
    except:
        return False


def convert_to_enumerable(test_obj):
    try:
        for elem in test_obj:
            pass
        return True
    except:
        return False


class ESN_ensemble(Model):
    """
    Single-step ESN network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self, num_outsteps=None, data_dim=None,
            dt_rnn=None,
            lambda_reg=None,
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
        
        super().__init__()

        self.num_outsteps = num_outsteps
        self.load_files_lst = load_file
        self.data_dim = []
        self.dt_rnn = []
        self.lambda_reg = []
        self.ESN_layers_units = []
        self.mean = []
        self.stddev = []
        self.noise_type = []
        self.stateful_lst = []
        self.omega_in = []
        self.sparsity = []
        self.rho_res = []
        self.usebias_Win = []
        self.prng_seed = []
        self.alpha = []
        self.ESN_cell_activations = []
        self.usebias_Wout = []
        self.batch_size = batch_size
        self.activation_post_Wout = []
        self.use_weights_post_dense = []
        self.scalar_weights = []
        self.T_input = kwargs.pop('T_input', None)
        self.T_output = kwargs.pop('T_output', None)
        self.wts_to_be_loaded = kwargs.pop('wts_to_be_loaded', False)
        if self.load_files_lst is not None:
            for load_file in self.load_files_lst:
                with open(load_file, 'r') as f:
                    lines = f.readlines()
                load_dict = eval(''.join(lines))
                if 'data_dim' in load_dict.keys():
                    self.data_dim.append(load_dict['data_dim'])
                if 'dt_rnn' in load_dict.keys():
                    self.dt_rnn.append(load_dict['dt_rnn'])
                if 'lambda_reg' in load_dict.keys():
                    self.lambda_reg.append(load_dict['lambda_reg'])
                if 'ESN_layers_units' in load_dict.keys():
                    self.ESN_layers_units.append(load_dict['ESN_layers_units'])
                if 'mean' in load_dict.keys():
                    self.mean.append(load_dict['mean'])
                if 'stddev' in load_dict.keys():
                    self.stddev.append(load_dict['stddev'])
                if 'noise_type' in load_dict.keys():
                    self.noise_type.append(load_dict['noise_type'])
                if 'stateful' in load_dict.keys():
                    self.stateful_lst.append(load_dict['stateful'])
                if 'omega_in' in load_dict.keys():
                    self.omega_in.append(load_dict['omega_in'])
                if 'sparsity' in load_dict.keys():
                    self.sparsity.append(load_dict['sparsity'])
                if 'rho_res' in load_dict.keys():
                    self.rho_res.append(load_dict['rho_res'])
                if 'usebias_Win' in load_dict.keys():
                    self.usebias_Win.append(load_dict['usebias_Win'])
                if 'alpha' in load_dict.keys():
                    self.alpha.append(load_dict['alpha'])
                if 'prng_seed' in load_dict.keys():
                    self.prng_seed.append(load_dict['prng_seed'])
                if 'ESN_cell_activations' in load_dict.keys():
                    self.ESN_cell_activations.append(load_dict['ESN_cell_activations'])
                if 'usebias_Wout' in load_dict.keys():
                    self.usebias_Wout.append(load_dict['usebias_Wout'])
                if 'activation_post_Wout' in load_dict.keys():
                    self.activation_post_Wout.append(load_dict['activation_post_Wout'])
                if 'use_weights_post_dense' in load_dict.keys():
                    self.use_weights_post_dense.append(load_dict['use_weights_post_dense'])
                if 'scalar_weights' in load_dict.keys():
                    self.scalar_weights.append(load_dict['scalar_weights'])

        self.num_rnn_layers = [len(self.ESN_layers_units[i]) for i in range(len(self.load_files_lst))]
        self.num_skip_connections = [elem - 1 for elem in self.num_rnn_layers]
        
        
        if not isinstance(lambda_reg, type(None)):
            self.lambda_reg = [lambda_reg for elem in self.lambda_reg]
        
        if self.T_input is not None:
            self.in_steps = int((self.T_input+0.5*self.dt_rnn[0])//self.dt_rnn[0])
        if self.T_output is not None:
            self.num_outsteps = int((self.T_output+0.5*self.dt_rnn[0])//self.dt_rnn[0])
        self.out_steps = self.num_outsteps

        self.noise_gen = [eval('tf.random.'+elem) for elem in self.noise_type]
        self.noise_kwargs = []
        for i_en in range(len(self.noise_type)):
            elem = self.noise_type[i_en]
            if elem == 'uniform':
                self.noise_kwargs.append({
                    'minval':self.mean[i_en]-1.732051*self.stddev[i_en],
                    'maxval':self.mean[i_en]+1.732051*self.stddev[i_en],
                })
            elif elem == 'normal':
                self.noise_kwargs.append({
                    'mean':self.mean[i_en],
                    'stddev':self.stddev[i_en],
                })


        ### the ESN network
        self.ensemble_list = []
        self.hidden_states_list = []
        self.rnn_list = []
        self.RK_RNNCell = []
        self.Wout = []
        self.post_Wout_activation = []
        self.postWout = []
        self.power_arr = []
        self.scalar_multiplier_pre_list = []
        self.hidden_states_list = []
        for i in range(len(self.load_files_lst)):
            i_rnn_list = [
                layers.RNN(
                    cell=ESN_Cell(
                        omega_in=self.omega_in[i][0],
                        sparsity=self.sparsity[i][0],
                        rho_res=self.rho_res[i][0],
                        state_size=self.ESN_layers_units[i][0],
                        alpha=self.alpha[i][0],
                        usebias_Win=self.usebias_Win[i][0],
                        prng_seed=self.prng_seed[i],
                        activation=self.ESN_cell_activations[i][0],
                        wts_to_be_loaded=self.wts_to_be_loaded,
                    ),
                    return_sequences=True,
                    return_state=True,
                    stateful=self.stateful_lst[i],
                    batch_size=self.batch_size,# if self.stateful_lst[i] else None,
                )
            ]
            if self.num_skip_connections[i] > 0:
                self.RK_RNNCell.append(ESN_Cell(
                        omega_in=self.omega_in[i][1],
                        sparsity=self.sparsity[i][1],
                        rho_res=self.rho_res[i][1],
                        state_size=self.ESN_layers_units[i][1],
                        alpha=self.alpha[i][1],
                        usebias_Win=self.usebias_Win[i][1],
                        prng_seed=self.prng_seed[i],
                        activation=self.ESN_cell_activations[i][1],
                        wts_to_be_loaded=self.wts_to_be_loaded,
                    ))
                i_rnn_list.extend([
                    layers.RNN(
                        self.RK_RNNCell[i],
                        return_sequences=True,
                        return_state=True,
                        stateful=self.stateful_lst[i],
                        batch_size=self.batch_size,# if self.stateful_lst[i] else None,
                    ) for jj in range(self.num_skip_connections[i])
                ])
            
            
            ### adding the Wout layer
            reg = lambda x : None
            if self.lambda_reg[i] != None:
                if self.lambda_reg[i] > 0.0:
                    reg = lambda x : tf.keras.regularizers.L2(x)
            self.Wout.append(layers.Dense(
                self.data_dim[i],
                use_bias=self.usebias_Wout[i],
                trainable=True,
                dtype='float32',
                name='Wout',
                kernel_regularizer=reg(self.lambda_reg[i]),
                bias_regularizer=reg(self.lambda_reg[i]),
            ))
            self.post_Wout_activation.append(tf.keras.activations.get(self.activation_post_Wout[i]))
            
            if self.use_weights_post_dense[i] == True:
                self.postWout.append(single_weights(w_regularizer=reg(self.lambda_reg[i])))

            power_arr = np.ones(shape=self.ESN_layers_units[i][-1])
            power_arr[0::2] = 2
            self.power_arr.append(power_arr)

            
            if self.num_skip_connections[i] > 0:
                if type(self.scalar_weights[i]) == type(None):
                    scalar_multiplier_pre_list = []
                    for ii in range(self.num_skip_connections[i]):
                        for j in range(ii+1):
                            scalar_multiplier_pre_list.append(
                                tf.Variable(
                                    initial_value=1.0,
                                    dtype='float32',
                                    trainable=True,
                                    name='a_{}{}'.format(j+1, ii+1) # this naming convention follows the one in the diagram at the top of this script
                                )
                            )
                    # self.scalar_multiplier_pre_list = [
                    #     tf.Variable(
                    #         initial_value=1.0,
                    #         dtype='float32',
                    #         trainable=True,
                    #     ) for i in range(int(0.5*self.num_skip_connections*(self.num_skip_connections+1)))
                    # ]
                    self.scalar_multiplier_pre_list.append(scalar_multiplier_pre_list)
                else:
                    self.scalar_weights[i] = np.array(self.scalar_weights[i], dtype='float32')
            
            self.rnn_list.append(i_rnn_list)


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
    def build(self, input_shape):
        for i_en in range(len(self.load_files_lst)):
            ldim = self.data_dim[i_en]
            for i_layer in range(len(self.rnn_list[i_en])):
                layer = self.rnn_list[i_en][i_layer]
                layer.build((self.batch_size, input_shape[-2], ldim))
                ldim = self.ESN_layers_units[i_en][i_layer]
            self.Wout[i_en].build((self.batch_size, ldim))
            if self.use_weights_post_dense[i_en] == True:
                self.postWout[i_en].build((self.batch_size, ldim))
        self.built = True
        return

    def _warmup(
            self,
            inputs,
            training=None,
            usenoiseflag=False,
            # scalar_multiplier_list=None,
            **kwargs):
        ### Initialize the ESN state.
        global_states_list = []
        global_intermediate_outputs_lst = []
        global_scalar_multiplier_list = self.scalar_weights
        
        final_output = 0.
        
        for i_en in range(len(self.load_files_lst)):   
            states_list = []
            intermediate_outputs_lst = []
            scalar_multiplier_list = global_scalar_multiplier_list[i_en]
            x = inputs
            if training == True or usenoiseflag == True:
                x = x + self.noise_gen[i_en](shape=tf.shape(x), **self.noise_kwargs[i_en])
            
            x, states = self.rnn_list[i_en][0](
                x,
                # training=training,
            )
            states_list.append(states)
            intermediate_outputs_lst.append(x)
            for i in range(self.num_skip_connections[i_en]):
                prediction, states = self.rnn_list[i_en][i+1](
                    x,
                    # training=training,
                )
                states_list.append(states)
                intermediate_outputs_lst.append(prediction)
                x = intermediate_outputs_lst[0]
                for j in range(i+1):
                    x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]

            output = x[:, -1:, :]
            output = tf.math.pow(output, self.power_arr[i_en])
            output = self.Wout[i_en](output, training=training) # DO NOT USE layers.TimeDistributed, dense layer takes care of it
            output = self.post_Wout_activation[i_en](output)
            if self.use_weights_post_dense[i_en] == True:
                output = self.postWout[i_en](output) # DO NOT USE layers.TimeDistributed, single_weights layer takes care of it

            global_states_list.append(states_list)
            global_intermediate_outputs_lst.append(intermediate_outputs_lst)
            final_output = final_output + output

        final_output = final_output / len(self.load_files_lst)

        return final_output, global_states_list, global_intermediate_outputs_lst, global_scalar_multiplier_list

    def onestep(
            self,
            x,
            training=False,
            intermediate_outputs_lst=[],
            states_list=[],
            scalar_multiplier_list=[],
            **kwargs):
        
        final_output = 0.
        
        for i_en in range(len(self.load_files_lst)):        
            ### Passing input through the ESN layers
            
            x2, states = self.rnn_list[i_en][0](
                x,
                initial_state=states_list[i_en][0],
                # training=training,
            )
            states_list[i_en][0] = states
            intermediate_outputs_lst[i_en][0] = x2
            for i in range(self.num_skip_connections[i_en]):
                prediction, states = self.rnn_list[i_en][i+1](
                    x2,
                    initial_state=states_list[i_en][i+1],
                    # training=training,
                )
                states_list[i_en][i+1] = states
                intermediate_outputs_lst[i_en][i+1] = prediction
                x2 = intermediate_outputs_lst[i_en][0]
                for j in range(i+1):
                    x2 += scalar_multiplier_list[i_en][int(i*(i+1)/2) + j] * intermediate_outputs_lst[i_en][j+1]

            output = x2[:, -1:, :]
            output = tf.math.pow(output, self.power_arr[i_en])
            output = self.Wout[i_en](output, training=training) # DO NOT USE layers.TimeDistributed, dense layer takes care of it
            output = self.post_Wout_activation[i_en](output)
            if self.use_weights_post_dense[i_en] == True:
                output = self.postWout[i_en](output) # DO NOT USE layers.TimeDistributed, single_weights layer takes care of it

            final_output = final_output + output
            
        final_output = final_output / len(self.load_files_lst)

        return final_output, states_list

    #@tf.function
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
        file_name = file_name[::-1]
        idx = file_name.find('/')
        fn_pt1 = file_name[0:idx]
        fn_pt2 = file_name[idx+1:]
        sep = '/'
        if fn_pt2[0] == '/':
            fn_pt2 = fn_pt2[1:]
            sep = '//'

        for i_file in range(len(self.load_files_lst)):
            fn = fn_pt1 + '_{}'.format(i_file) + sep + fn_pt2
            fn = fn[::-1]
            f = h5py.File(fn, 'w')
            ESN_cell_Win = f.create_group('ESN_cell_Win')
            ESN_cell_Wres = f.create_group('ESN_cell_Wres')
            ESN_net_Wout = f.create_group('ESN_net_Wout')

            for i in range(len(self.rnn_list[i_file])):
                ### saving Win in a sparse manner
                usebias_Win = self.usebias_Win[i_file][i]
            
                Win = self.rnn_list[i_file][i].cell.Win
                Win_kernel_sparse = tf.sparse.from_dense(Win.kernel)

                Win_this_cell = ESN_cell_Win.create_group("cell_{}".format(i))
                
                Win_this_cell.create_dataset("kernel_values", data=Win_kernel_sparse.values.numpy())
                Win_this_cell.create_dataset("kernel_indices", data=Win_kernel_sparse.indices.numpy())
                if usebias_Win == True:
                    Win_bias = Win.bias
                    Win_this_cell.create_dataset("bias", data=Win_bias.numpy())
                    
                ### saving Wres in a sparse manner
                Wres = self.rnn_list[i_file][i].cell.Wres
                Wres_kernel_sparse = tf.sparse.from_dense(Wres.kernel)

                Wres_this_cell = ESN_cell_Wres.create_group("cell_{}".format(i))
                
                Wres_this_cell.create_dataset("kernel_values", data=Wres_kernel_sparse.values.numpy())
                Wres_this_cell.create_dataset("kernel_indices", data=Wres_kernel_sparse.indices.numpy())

            ### saving Wout
            ESN_net_Wout.create_dataset("kernel", data=self.Wout[i_file].kernel.numpy())
            if self.usebias_Wout[i_file] == True:
                ESN_net_Wout.create_dataset("bias", data=self.Wout[i_file].bias.numpy())

            if self.use_weights_post_dense[i_file] == True:
                ESN_net_postWout = f.create_group('ESN_net_postWout') 
                ESN_net_postWout.create_dataset("individual_weights", data=self.postWout[i_file].individual_weights.numpy())
            
            f.close()

        return
    
    def save_class_dict(self, file_name):
        
        file_name = file_name[::-1]
        idx = file_name.find('/')
        fn_pt1 = file_name[0:idx]
        fn_pt2 = file_name[idx+1:]
        sep = '/'
        if fn_pt2[0] == '/':
            fn_pt2 = fn_pt2[1:]
            sep = '//'

        for i_file in range(len(self.load_files_lst)):
            fn = fn_pt1 + '_{}'.format(i_file) + sep + fn_pt2
            fn = fn[::-1]
            
            class_dict = {
                'data_dim':self.data_dim[i_file],
                'dt_rnn':self.dt_rnn[i_file],
                'lambda_reg':self.lambda_reg[i_file],
                'ESN_layers_units':list(self.ESN_layers_units[i_file]),
                'mean':self.mean[i_file],
                'stddev':self.stddev[i_file],
                'noise_type':self.noise_type[i_file],
                'stateful':self.stateful_lst[i_file],
                'omega_in':list(self.omega_in[i_file]),
                'sparsity':list(self.sparsity[i_file]),
                'rho_res':list(self.rho_res[i_file]),
                'usebias_Win':list(self.usebias_Win[i_file]),
                'prng_seed':self.prng_seed[i_file],
                'alpha':list(self.alpha[i_file]),
                'ESN_cell_activations':list(self.ESN_cell_activations[i_file]),
                'usebias_Wout':self.usebias_Wout[i_file],
                'activation_post_Wout':self.activation_post_Wout[i_file],
                'use_weights_post_dense':self.use_weights_post_dense[i_file],
                'scalar_weights':list(self.scalar_weights[i_file]),
            }
            
            with open(fn, 'w') as f:
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

    def load_weights_from_file(self, file_name_lst):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        # self.load_weights(file_name)

        if self.built == False:
            self.build((self.batch_size, self.in_steps, self.data_dim[0]))
            # self.build((self.batch_size, self.in_steps, self.data_dim[0])) # self.data_dim is a list of the ensemble data_dims

        for i_file in range(len(file_name_lst)):
            file_name = file_name_lst[i_file]
            f = h5py.File(file_name,'r')
            ESN_cell_Win = f['ESN_cell_Win']
            ESN_cell_Wres = f['ESN_cell_Wres']
            ESN_net_Wout = f['ESN_net_Wout']

            input_dim = self.data_dim[i_file]
            for i in range(len(self.rnn_list[i_file])):
                ### reading Win, stored in a sparse manner
                usebias_Win = self.usebias_Win[i_file][i]
                
                Win_this_cell = ESN_cell_Win["cell_{}".format(i)]

                Win_kernel_sparse_values = np.array(Win_this_cell["kernel_values"])
                Win_kernel_sparse_indices = np.array(Win_this_cell["kernel_indices"])
                Win_kernel = tf.sparse.SparseTensor(
                    indices=Win_kernel_sparse_indices,
                    values=Win_kernel_sparse_values,
                    dense_shape=[input_dim, self.ESN_layers_units[i_file][i]])
                Win_kernel = tf.sparse.to_dense(Win_kernel, validate_indices=False)
                Win_kernel = Win_kernel.numpy()

                Win = self.rnn_list[i_file][i].cell.Win
                K.set_value(Win.kernel, Win_kernel)
                if usebias_Win == True:
                    Win_bias = np.array(Win_this_cell["bias"])
                    K.set_value(Win.bias, Win_bias)

                    
                ### reading Wres, stored in a sparse manner
                Wres_this_cell = ESN_cell_Wres["cell_{}".format(i)]
                input_dim =  self.ESN_layers_units[i_file][i]

                Wres_kernel_sparse_values = np.array(Wres_this_cell["kernel_values"])
                Wres_kernel_sparse_indices = np.array(Wres_this_cell["kernel_indices"])
                Wres_kernel = tf.sparse.SparseTensor(
                    indices=Wres_kernel_sparse_indices,
                    values=Wres_kernel_sparse_values,
                    dense_shape=[input_dim, input_dim])
                Wres_kernel = tf.sparse.to_dense(Wres_kernel, validate_indices=False)
                Wres_kernel = Wres_kernel.numpy()

            Wres = self.rnn_list[i_file][i].cell.Wres
            K.set_value(Wres.kernel, Wres_kernel)

            ### reading Wout
            K.set_value(self.Wout[i_file].kernel, np.array(ESN_net_Wout['kernel']))
            if self.usebias_Wout[i_file] == True:
                K.set_value(self.Wout[i_file].bias, np.array(ESN_net_Wout['bias']))

            if self.use_weights_post_dense[i_file] == True:
                ESN_net_postWout = f["ESN_net_postWout"]
                K.set_value(self.postWout[i_file].individual_weights, np.array(ESN_net_postWout["individual_weights"]))
            
            f.close()
        
        return


################################################################################
