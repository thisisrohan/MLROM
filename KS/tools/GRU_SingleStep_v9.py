################################################################################
# RK Methods inspired residual GRU with skip connections, with uniform/normal  #
# noise added to every input.                                                  #
#------------------------------------------------------------------------------#
#                        Basic Network Architecture                            #
#------------------------------------------------------------------------------#
#                                                                              #
#                        ________________________________________              #
#                       /                  ______________________\    z1+a1*d1 #
#                      /                  /                       \     +a2*d2 #
#         __   z1   __/  d1     z1+d1  __/  d2     z1+d2  __   d3  \    +a3*d3 #
# u----->|__|----->|__|----->[+]----->|__|----->[+]----->|__|----->[+]----->   #
#           \________________/                  /                  /           #
#            \_________________________________/                  /            #
#             \__________________________________________________/             #
#                                                                              #
# (a1, a2 and a3 are scalars that determine a weighted average and sum to 1)   #
#                                                                              #
# Note here that you can only specify the number of layers and the number of   #
# units in a layer, not the number of units in each layer individually. Also, a#
# single layer network is the same as a regular GRU.                           #
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

class learnable_state(layers.Layer):
    def __init__(self, hidden_shape, b_regularizer=None, **kwargs):
        super(learnable_state, self).__init__()
        self.learnable_variable = self.add_weight(
            name='learnable_variable',
            shape=[1, hidden_shape],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            regularizer=b_regularizer,
            trainable=True
        )

    def call(self, x):
        batch_size = x.shape[0]
        if batch_size == None:
            batch_size = 1
        return tf.tile(self.learnable_variable, [batch_size, 1])


class scalar_multipliers(layers.Layer):
    def __init__(self, num_skip_connections, **kwargs):
        super(scalar_multipliers, self).__init__()
        self.scalars = self.add_weight(
            name='final scalar multipliers',
            shape=[num_skip_connections],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True
        )

    def call(self, x, idx=0):
        multiplier = tf.math.exp(self.scalars[idx])/tf.math.sum(tf.math.exp(self.scalars))
        return multiplier*x


class uniform_noise(layers.Layer):
    def __init__(self, mean=0.0, stddev=1e-4, **kwargs):
        super(uniform_noise, self).__init__()
        self.stddev = stddev
        self.mean = mean

    def call(self, x):
        x = x + tf.random.uniform(shape=tf.shape(x), minval=self.mean-1.732051*self.stddev, maxval=self.mean+1.732051*self.stddev)
        return x


class RNN_GRU(Model):
    """
    Single-step GRU network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self, data_dim=None,
            dt_rnn=None,
            lambda_reg=0.0,
            reg_name='L2',
            rnn_layers_units=[3, 3, 3],
            dense_layer_act_func='linear',
            load_file=None,
            stddev=0.0,
            mean=0.0,
            noise_type='uniform',
            dense_dim=None):
        
        super(RNN_GRU, self).__init__()

        self.load_file = load_file
        self.data_dim = data_dim
        self.dt_rnn = dt_rnn
        self.lambda_reg = lambda_reg
        self.reg_name = reg_name
        self.rnn_layers_units = rnn_layers_units
        self.dense_layer_act_func = dense_layer_act_func
        self.mean = mean
        self.stddev = stddev
        self.noise_type = noise_type
        self.dense_dim = dense_dim
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
            if 'reg_name' in load_dict.keys():
                self.reg_name = load_dict['reg_name']
            if 'rnn_layers_units' in load_dict.keys():
                self.rnn_layers_units = load_dict['rnn_layers_units']
            if 'dense_layer_act_func' in load_dict.keys():
                self.dense_layer_act_func = load_dict['dense_layer_act_func']
            if 'mean' in load_dict.keys():
                self.mean = load_dict['mean']
            if 'stddev' in load_dict.keys():
                self.stddev = load_dict['stddev']
            if 'noise_type' in load_dict.keys():
                self.noise_type = load_dict['noise_type']
            if 'dense_dim' in load_dict.keys():
                self.dense_dim = load_dict['dense_dim']
        self.num_rnn_layers = len(self.rnn_layers_units)

        if isinstance(self.dense_layer_act_func, list) == False:
            self.dense_layer_act_func = [self.dense_layer_act_func]

        if self.dense_dim is None:
            self.dense_dim = [self.data_dim]*len(self.dense_layer_act_func)


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
        self.num_skip_connections = self.num_rnn_layers - 1
        
        if self.num_skip_connections > 0:
            self.final_scalar_multipliers = scalar_multipliers(self.num_skip_connections)

        ### the GRU network
        if self.reg_name != None and self.lambda_reg != None and self.lambda_reg != 0:
            reg = eval('tf.keras.regularizers.'+self.reg_name)
            self.rnn_cells_list = [
                layers.GRUCell(
                    units=units,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg)
                ) for units in self.rnn_layers_units
            ]
            
            self.hidden_states_list = [learnable_state(hidden_shape=units, b_regularizer=reg(self.lambda_reg)) for units in self.rnn_layers_units]

            self.dense = [
                layers.Dense(
                    self.dense_dim[i],
                    activation=self.dense_layer_act_func[i],
                    # kernel_initializer=tf.initializers.zeros(),
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg)
                ) for i in range(len(self.dense_layer_act_func))
            ]
        else:
            self.rnn_cells_list = [
                layers.GRUCell(units=units) for units in self.rnn_layers_units
            ]
            
            self.hidden_states_list = [learnable_state(hidden_shape=units) for units in self.rnn_layers_units]

            self.dense = [
                layers.Dense(
                    self.dense_dim[i],
                    activation=self.dense_layer_act_func[i],
                    # kernel_initializer=tf.initializers.zeros(),
                ) for i in range(len(self.dense_layer_act_func))
            ]


        ### initializing weights
        temp = tf.ones(shape=[1, 1, self.data_dim])
        temp = self.predict(temp)

        return

    @tf.function
    def call(self, inputs, training=None):

        # inputs shape : (None, time_steps, data_dim)
        out_steps = inputs.shape[1]

        predictions_list = []
        intermediate_outputs_lst = []
        
        ### Initialize the GRU state.
        states_list = []
        # first step
        x = inputs[:, 0, :]
        x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        state1 = self.hidden_states_list[0](x, training=training)
        prediction, *states = self.rnn_cells_list[j](
            x,
            states=state1,
            training=training,
        )
        states_list.append(states[0])
        intermediate_outputs_lst.append(prediction)
        x = prediction
        for j in range(1, self.num_rnn_layers):
            state1 = self.hidden_states_list[j](x, training=training)
            prediction, *states = self.rnn_cells_list[j](
                x,
                states=state1,
                training=training,
            )
            states_list.append(states[0])
            intermediate_outputs_lst.append(prediction)
            x = intermediate_outputs_lst[0] + prediction
        x = intermediate_outputs_lst[0]
        for j in range(self.num_skip_connections)
            x = x + self.final_scalar_multipliers(intermediate_outputs_list[j+1], idx=j)
        for j in range(len(self.dense_layer_act_func)):
            prediction = self.dense[j](prediction, training=training)
        predictions_list.append(prediction)

        ### Remaining number of time-steps
        for i in range(1, out_steps):
            x = inputs[:, i, :]
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
            state1 = states_list[0]
            prediction, *states = self.rnn_cells_list[j](
                x,
                states=state1,
                training=training,
            )
            states_list[0] = states[0]
            intermediate_outputs_lst[0] = prediction
            x = prediction
            for j in range(1, self.num_rnn_layers):
                state1 = states_list[0]
                prediction, *states = self.rnn_cells_list[j](
                    x,
                    states=state1,
                    training=training,
                )
                states_list[j] = states[0]
                intermediate_outputs_lst[j] = prediction
                x = intermediate_outputs_lst[0] + prediction
            x = intermediate_outputs_lst[0]
            for j in range(self.num_skip_connections)
                x = x + self.final_scalar_multipliers(intermediate_outputs_list[j+1], idx=j)
            for j in range(len(self.dense_layer_act_func)):
                prediction = self.dense[j](prediction, training=training)
            predictions_list.append(prediction)

        # predictions_list.shape => (time, batch, features)
        predictions = tf.stack(predictions_list)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


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
            'reg_name':self.reg_name,
            'rnn_layers_units':list(self.rnn_layers_units),
            'dense_layer_act_func':list(self.dense_layer_act_func),
            'load_file':self.load_file,
            'mean':self.mean,
            'stddev':self.stddev,
            'noise_type':self.noise_type,
            'module':self.__module__,
            'dense_dim':list(self.dense_dim),
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
        self.save_model_weights(file_name+'_gru_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name)
        return


################################################################################
