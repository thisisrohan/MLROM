################################################################################
# Regular ESN with uniform/normal noise added to every input.                  #
################################################################################

import os
import numpy as np
from scipy import linalg

import time as time

import tensorflow as tf
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

################################################################################
#################################### LSTM V4 ###################################

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
        self.Wres_np = self._build_Wres()
        
        return

    def _build_Win(self):
        
        shape = (self.input_size, self.state_num_units)
        if self.usebias_Win == True:
            # last row of the matrix Win corresponds to the bias
            shape[0] += 1

        ### this is a sparse Win, with only one element in each column having a non-zero value
        Win = np.zeros(shape=shape, dtype='float32')
        for i in range(shape[1]):
            Win[self.prng.integers(low=0, high=self.input_size, size=1), i] = self.prng.uniform(low=-self.omega_in, high=self.omega_in)
        if self.usebias_Win == True:
            Win[-1, :] = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=self.state_num_units)

        ### this is a dense Win
        # Win = self.prng.uniform(low=-self.omega_in, high=self.omega_in, size=shape)

        return Win.astype('float32')

    def _build_Wres(self):
    
        shape = (self.state_num_units, self.state_num_units)
        
        Wres = self.prng.uniform(low=-1, high=1, size=shape)
        Wres *= (self.prng.random(size=shape) < 1 - self.sparsity)
        
        fac = self.rho_res / np.max(np.abs(np.linalg.eigvals(Wres)))
        Wres *= fac
        
        return Wres.astype('float32')

    def build(self, input_shape):
    
        self.input_size = input_shape[-1]
        self.Win_np = self._build_Win()
        self.Win = tf.Variable(self.Win_np, trainable=False, dtype='float32', name='Win')
        self.Wres = tf.Variable(self.Wres_np, trainable=False, dtype='float32', name='Wres')

        # shape = [self.state_num_units, self.output_size]
        # if self.usebias_Wout == True:
        #     shape[0] += 1
        
        # self.Wout = tf.add_weight(
        #     shape=shape,
        #     trainable=True,
        #     dtype='float32',
        #     name='Wout',
        #     initializer=tf.keras.initializers.HeUniform(),
        # )
        
        super(ESN_Cell, self).build(input_shape)

    def call(self, inputs, states):

        state = states[0]
        
        candidate_state = tf.linalg.matmul(state, self.Wres)
        candidate_state = candidate_state + tf.linalg.matmul(inputs, self.Win[0:self.input_size])
        if self.usebias_Win == True:
            candidate_state = candidate_state + tf.tile(self.Win[-1], [inputs.shape[0], 1])
        candidate_state = self.activation(candidate_state)
        
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
                    batch_size=batch_size,
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=batch_size,
            ) for i in range(len(self.ESN_layers_units))
        ]
        
        
        ### adding the Wout layer
        shape = [self.ESN_layers_units[-1], self.data_dim]
        if self.usebias_Wout == True:
            shape[0] += 1
        
        self.Wout = self.add_weight(
            shape=shape,
            trainable=True,
            dtype='float32',
            name='Wout',
            initializer=tf.keras.initializers.HeUniform(),
        )

        # self.global_Hb = np.zeros(shape=[shape[0]]*2, dtype='float32')
        # self.global_Yb = np.zeros(shape=[self.data_dim, shape[0]], dtype='float32')


        ### initializing weights
        # temp = tf.ones(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = Input(shape=(1, self.data_dim), batch_size=batch_size)
        # temp = self(temp)

        return

    def build(self, input_shape):

        for rnnlayer in self.ESN_layers:
            rnnlayer.build(input_shape)
        # super(ESN, self).build(input_shape)
        self.built = True

    # @tf.function
    def call(self, inputs, training=None):

        # inputs shape : (None, time_steps, data_dim)
        out_steps = inputs.shape[1]

        ### Running through the ESN layers
        x = tf.Variable(inputs)
        if training == True:
            # add noise to the inputs during training
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        for i in range(len(self.ESN_layers_units)):
            x = self.ESN_layers[i](x)

        ### applying the quadratic transformation from Pathak et. al.
        x = K.get_value(x)
        for k in range(x.shape[-1]):
            if k % 2 == 0:
                # K.set_value(x[:, :, k], tf.math.square(K.get_value(x[:, :, k])))
                x[:, :, k] = x[:, :, k]**2

        if self.usebias_Wout == True:
            x = tf.concat([x, tf.ones(shape=(inputs.shape[0], inputs.shape[1], 1))], axis=-1)

        if training == True:
            # simply return the hidden states, these are used for specific matrix computations
            output = x
        else:
            output = tf.linalg.matmul(x, self.Wout)

        return output


    def save_model_weights(self, file_name, H5=True):

        # file_name = file_dir + '/' + file_name
        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        if H5 == True:
            file_name += '.h5'
        self.save_weights(file_name)
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

        self.load_weights(file_name)
        return


################################################################################
