################################################################################
# Gated Recurrent Unit RNN                                                     #
# For Comparing Hidden States between the Teacher Forced and autoregressive    #
# models.                                                                      #
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


class TF_RNN_GRU_CHS(Model):
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
            load_file=None):
        
        super(TF_RNN_GRU_CHS, self).__init__()

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
        self.num_rnn_layers = len(self.rnn_layers_units)

        ### time steps
        self.in_steps = int((self.T_input+0.5*self.dt_rnn)//self.dt_rnn)
        self.out_steps = int((self.T_output+0.5*self.dt_rnn)//self.dt_rnn)


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
                # kernel_initializer=tf.initializers.zeros(),
            )


        ### initializing weights
        temp = tf.ones(shape=[1, self.in_steps, self.data_dim])
        temp = self.predict(temp)

        return

    @tf.function
    def _warmup(self, inputs, training=None):
        ### Initialize the GRU state.
        states_list = [None]*self.num_rnn_layers
        # first step
        prediction = inputs[:, 0, :]
        for j in range(self.num_rnn_layers):
            prediction, *states = self.rnn_cells_list[j](
                prediction,
                states=self.hidden_states_list[j](prediction, training=training),
                training=training
            )
            states_list[j] = states[0]

        # remaining number of input steps
        # warmup_steps = inputs.shape[1]
        # if warmup_steps == None:
        #     warmup_steps = 1
        for i in range(1, self.in_steps-self.out_steps):
            prediction = inputs[:, i, :]
            for j in range(self.num_rnn_layers):
                prediction, *states = self.rnn_cells_list[j](
                    prediction,
                    states=states_list[j],
                    training=training
                )
                states_list[j] = states[0]

        # pass final RNN output through the dense layer to get final prediction
        prediction = self.dense(prediction, training=training)

        return prediction, states_list
        

    @tf.function
    def call(self, inputs, training=None):
        predictions_TF_list = []
        hidden_states_TF_list = []

        prediction_TF, states_list_TF = self._warmup(inputs, training)

        # first prediction
        predictions_TF_list.append(prediction_TF)
        
        # print(type(states_list))
        hidden_states_TF_list.append(states_list_TF)

        ### Run the rest of the prediction steps -- TF mode
        for i in range(self.in_steps-self.out_steps, self.in_steps - 1):
            states_list_TF_new = [None]*self.num_rnn_layers
            prediction_TF = inputs[:, i, :]
            for j in range(self.num_rnn_layers):
                prediction_TF, *states_TF = self.rnn_cells_list[j](
                    prediction_TF,
                    states=states_list_TF[j],
                    training=training
                )
                states_list_TF_new[j] = states_TF[0]
            states_list_TF = states_list_TF_new
            prediction_TF = self.dense(prediction_TF, training=training)
            predictions_TF_list.append(prediction_TF)
            hidden_states_TF_list.append(states_list_TF)

        # predictions_list.shape => (time, batch, features)
        predictions_TF = tf.stack(predictions_TF_list)
        hidden_states_TF = tf.stack(hidden_states_TF_list)
        # predictions.shape => (batch, time, features)
        predictions_TF = tf.transpose(predictions_TF, [1, 0, 2])
        hidden_states_TF = tf.transpose(hidden_states_TF, [1, 0, 2, 3])

        return predictions_TF, hidden_states_TF

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
            # 'out_steps':self.out_steps,
            'dt_rnn':self.dt_rnn,
            'lambda_reg':self.lambda_reg,
            'reg_name':self.reg_name,
            'rnn_layers_units':list(self.rnn_layers_units),
            'dense_layer_act_func':self.dense_layer_act_func,
            'load_file':self.load_file,
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
        self.save_model_weights(file_name+'_lstm_weights', H5=H5)

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
