################################################################################
# Regular residual GRU with skip connections.                                  #
#------------------------------------------------------------------------------#
#                        Basic Network Architecture                            #
#------------------------------------------------------------------------------#
#                                                                     z1+d1    #
#                                                  z1+d1                +d2    #
#         __   z1   __   d1     z1+d1  __   d2       +d2  __   d3       +d3    #
# u----->|__|----->|__|----->[+]----->|__|----->[+]----->|__|----->[+]----->   #
#           \________________/ \________________/ \________________/           #
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
        self.hidden_shape = hidden_shape
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
        # return tf.matmul(tf.ones(shape=[batch_size, 1]), self.learnable_variable)
        return tf.tile(self.learnable_variable, [batch_size, 1])


class AR_RNN_GRU(Model):
    """
    Single-step GRU network that advances (in time) the latent space representation
    """
    def __init__(
            self, data_dim=None,
            T_input=None,
            T_output=None,
            dt_rnn=None,
            lambda_reg=0.0,
            reg_name='L2',
            rnn_layers_units=[3, 3, 3],
            dense_layer_act_func='linear',
            load_file=None,
            stddev=None,
            mean=None,
            noise_type='uniform'):
        
        super(AR_RNN_GRU, self).__init__()

        self.mean = mean
        self.stddev = stddev
        self.noise_type = noise_type

        self.load_file = load_file
        if self.load_file == None:
            self.data_dim = data_dim
            self.T_input = T_input
            self.T_output = T_output
            self.dt_rnn = dt_rnn
            self.lambda_reg = lambda_reg
            self.reg_name = reg_name
            self.rnn_layers_units = rnn_layers_units
            self.dense_layer_act_func = dense_layer_act_func
            self.noise_type = noise_type
        else:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            self.data_dim = load_dict['data_dim']
            # self.T_input = load_dict['T_input']
            # self.T_output = load_dict['T_output']
            self.T_input = T_input
            self.T_output = T_output
            self.dt_rnn = load_dict['dt_rnn']
            self.lambda_reg = load_dict['lambda_reg']
            self.reg_name = load_dict['reg_name']
            self.rnn_layers_units = load_dict['rnn_layers_units']
            self.dense_layer_act_func = load_dict['dense_layer_act_func']
            if 'out_steps' in load_dict.keys():
                self.out_steps = int(load_dict['out_steps'])
            if 'in_steps' in load_dict.keys():
                self.in_steps = int(load_dict['in_steps'])
            if 'stddev' in load_dict.keys():
                self.stddev = int(load_dict['stddev'])
            if 'mean' in load_dict.keys():
                self.mean = int(load_dict['mean'])
            if 'noise_type' in load_dict.keys():
                self.noise_type = load_dict['noise_type']
        self.num_rnn_layers = len(self.rnn_layers_units)

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
            
            self.hidden_states_list = [
    		learnable_state(hidden_shape=units, b_regularizer=reg(self.lambda_reg)) for units in self.rnn_layers_units
            ]

            self.dense = layers.Dense(
                self.data_dim,
                activation=self.dense_layer_act_func,
                # kernel_initializer=tf.initializers.zeros(),
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            )
        else:
            self.rnn_cells_list = [
                layers.GRUCell(units=units) for units in self.rnn_layers_units
            ]
            
            self.hidden_states_list = [
                learnable_state(hidden_shape=units) for units in self.rnn_layers_units
            ]

            self.dense = layers.Dense(
                self.data_dim,
                activation=self.dense_layer_act_func,
                # kernel_initializer=tf.initializers.zeros(),
            )


        ### initializing weights
        temp = tf.ones(shape=[1, self.in_steps, self.data_dim])
        temp = self.predict(temp)

        return

    @tf.function
    def _warmup(self, inputs, training=None):
        ### Initialize the GRU state.
        states_list = []
        # first step
        x = inputs[:, 0, :]
        x = x + tself.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        state1 = self.hidden_states_list[0](x, training=training)
        prediction, *states = self.rnn_cells_list[0](
            x,
            states=state1,
            training=training,
        )
        states_list.append(states[0])
        x = prediction
        for j in range(1, self.num_rnn_layers):
            state1 = self.hidden_states_list[j](x, training=training)
            prediction, *states = self.rnn_cells_list[j](
                x,
                states=state1,
                training=training,
            )
            states_list.append(states[0])
            x = prediction + x
        # prediction = self.dense(x, training=training)

        ### Remaining number of time-steps
        for i in range(1, self.in_steps):
            x = inputs[:, i, :]
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
            state1 = states_list[0]
            prediction, *states = self.rnn_cells_list[0](
                x,
                states=state1,
                training=training,
            )
            states_list[0] = states[0]
            x = prediction
            for j in range(1, self.num_rnn_layers):
                state1 = states_list[j]
                prediction, *states = self.rnn_cells_list[j](
                    x,
                    states=state1,
                    training=training,
                )
                states_list[j] = states[0]
                x = prediction + x
        prediction = self.dense(x, training=training)

        return prediction, states_list
        

    @tf.function
    def call(self, inputs, training=None):
        predictions_list = []
        prediction, states_list = self._warmup(inputs, training=False)

        # first prediction
        predictions_list.append(prediction)

        ### Run the rest of the prediction steps.
        for i in range(1, self.out_steps):
            x = prediction
            state1 = states_list[0]
            prediction, *states = self.rnn_cells_list[0](
                x,
                states=state1,
                training=training,
            )
            states_list[0] = states[0]
            x = prediction
            for j in range(1, self.num_rnn_layers):
                state1 = states_list[j]
                prediction, *states = self.rnn_cells_list[j](
                    x,
                    states=state1,
                    training=training,
                )
                states_list[j] = states[0]
                x = prediction + x
            prediction = self.dense(x, training=training)
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
            'dense_layer_act_func':self.dense_layer_act_func,
            'load_file':self.load_file,
            'in_steps':self.in_steps,
            'out_steps':self.out_steps,
            'stddev':self.stddev,
            'mean':self.mean,
            'noise_type':self.noise_type,
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

    def update_Tinput(self, new_T_input):
        self.T_input = new_T_input
        self.in_steps = int((self.T_input+0.5*self.dt_rnn)//self.dt_rnn)

    def update_Toutput(self, new_T_output):
        self.T_output = new_T_output
        self.out_steps = int((self.T_output+0.5*self.dt_rnn)//self.dt_rnn)

################################################################################
