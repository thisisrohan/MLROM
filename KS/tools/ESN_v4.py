################################################################################
# Regular ESN with uniform/normal noise added to every input.  Using sparse    #
# tensors to make storage easier and more efficient.                           #
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
            name='Win'
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
            name='Wres'
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
            
        self.num_ESN_layers = len(self.ESN_layers_units)

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
        self.ESN_layers = [
            layers.RNN(
                cell=ESN_Cell(
                    omega_in=self.omega_in[i],
                    sparsity=self.sparsity[i],
                    rho_res=self.rho_res[i],
                    state_size=self.ESN_layers_units[i],
                    alpha=self.alpha[i],
                    usebias_Win=self.usebias_Win[i],
                    prng_seed=self.prng_seed,
                    activation=self.ESN_cell_activations[i],
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            ) for i in range(len(self.ESN_layers_units))
        ]
        
        
        ### adding the Wout layer
        reg = None
        if self.lambda_reg > 0.0:
            reg = tf.keras.regularizers.L2(self.lambda_reg)
        self.Wout = layers.Dense(
            self.data_dim,
            use_bias=self.usebias_Wout,
            trainable=True,
            dtype='float32',
            name='Wout',
            kernel_regularizer=reg,
            bias_regularizer=reg,
        )
        self.post_Wout_activation = tf.keras.activations.get(self.activation_post_Wout)
        
        if self.use_weights_post_dense == True:
            self.postWout = single_weights(w_regularizer=reg)

        self.power_arr = np.ones(shape=self.ESN_layers_units[-1])
        self.power_arr[0::2] = 2

        # self.global_Hb = np.zeros(shape=[shape[0]]*2, dtype='float32')
        # self.global_Yb = np.zeros(shape=[self.data_dim, shape[0]], dtype='float32')


        ### initializing weights
        # temp = tf.ones(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = Input(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = self(temp)

        return

    def build(self, input_shape):

        if self.stateful:
            input_shape_ESN = (None, )
        else:
            input_shape_ESN = (self.batch_size, )
        input_shape_ESN = input_shape_ESN + tuple(input_shape[1:])

        for rnnlayer in self.ESN_layers:
            rnnlayer.build(input_shape_ESN)
            input_shape_ESN = input_shape_ESN[0:-1] + (rnnlayer.cell.state_num_units, )

        self.Wout.build(input_shape_ESN)
        # super(ESN, self).build(input_shape)
        if self.use_weights_post_dense == True:
            self.postWout.build(input_shape=[None, input_shape[-1]])

        self.built = True

    @tf.function
    def _callhelperfn(self, x):
        for i in range(len(self.ESN_layers_units)):
            x = self.ESN_layers[i](x)
        return x

    # @tf.function
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

        for i in range(len(self.ESN_layers)):
            ### saving Win in a sparse manner
            usebias_Win = self.usebias_Win[i]
            
            Win = self.ESN_layers[i].cell.Win
            Win_kernel_sparse = tf.sparse.from_dense(Win.kernel)

            Win_this_cell = ESN_cell_Win.create_group("cell_{}".format(i))
            
            Win_this_cell.create_dataset("kernel_values", data=Win_kernel_sparse.values.numpy())
            Win_this_cell.create_dataset("kernel_indices", data=Win_kernel_sparse.indices.numpy())
            if usebias_Win == True:
                Win_bias = Win.bias
                Win_this_cell.create_dataset("bias", data=Win_bias.numpy())
                
            ### saving Wres in a sparse manner
            Wres = self.ESN_layers[i].cell.Wres
            Wres_kernel_sparse = tf.sparse.from_dense(Wres.kernel)

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
            self.build((self.batch_size, None, self.data_dim))
            
        f = h5py.File(file_name,'r')
        ESN_cell_Win = f['ESN_cell_Win']
        ESN_cell_Wres = f['ESN_cell_Wres']
        ESN_net_Wout = f['ESN_net_Wout']

        input_dim = self.data_dim
        for i in range(len(self.ESN_layers)):
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

            Win = self.ESN_layers[i].cell.Win
            K.set_value(Win.kernel, Win_kernel)
            if usebias_Win == True:
                Win_bias = np.array(Win_this_cell["bias"])
                K.set_value(Win.bias, Win_bias)
                
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

            Wres = self.ESN_layers[i].cell.Wres
            K.set_value(Wres.kernel, Wres_kernel)

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
