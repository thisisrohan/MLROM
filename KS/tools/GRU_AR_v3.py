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


class normalization_layer(layers.Layer):
    def __init__(self, normalization_arr, **kwargs):
        super(normalization_layer, self).__init__()
        self.normalization_arr = normalization_arr
        self.data_shape = self.normalization_arr.shape[1]
        self.mean = self.add_weight(
            name='sample_mean',
            shape=[1, self.data_shape],
            initializer=tf.constant_initializer(self.normalization_arr[0, :]),
            trainable=False
        )
        self.std = self.add_weight(
            name='sample_std',
            shape=[1, self.data_shape],
            initializer=tf.constant_initializer(self.normalization_arr[1, :]),
            trainable=False
        )

    def call(self, x):
        batch_size = x.shape[0]
        if batch_size == None:
            batch_size = 1
        x = tf.math.subtract(x, tf.tile(self.mean, [batch_size, 1]))
        x = tf.math.divide(x, tf.tile(self.std, [batch_size, 1]))
        return x


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
            normalization_arr=None):
        
        super(AR_RNN_GRU, self).__init__()

        self.load_file = load_file
        #if self.load_file == None:
        self.data_dim = data_dim
        self.T_input = T_input
        self.T_output = T_output
        self.dt_rnn = dt_rnn
        self.lambda_reg = lambda_reg
        self.reg_name = reg_name
        self.rnn_layers_units = rnn_layers_units
        self.dense_layer_act_func = dense_layer_act_func
        self.normalization_arr = normalization_arr
        if load_file != None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            try:
                self.data_dim = load_dict['data_dim']
            except:
                pass
            # self.T_input = load_dict['T_input']
            # self.T_output = load_dict['T_output']
            try:
                self.dt_rnn = load_dict['dt_rnn']
            except:
                pass
            try:
                self.lambda_reg = load_dict['lambda_reg']
            except:
                pass
            try:
                self.reg_name = load_dict['reg_name']
            except:
                pass
            try:
                self.rnn_layers_units = load_dict['rnn_layers_units']
            except:
                pass
            try:
                self.dense_layer_act_func = load_dict['dense_layer_act_func']
            except:
                pass
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
            
            self.hidden_states_list = [learnable_state(hidden_shape=units, b_regularizer=reg(self.lambda_reg)) for units in self.rnn_layers_units]

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
            
            self.hidden_states_list = [learnable_state(hidden_shape=units) for units in self.rnn_layers_units]

            self.dense = layers.Dense(
                self.data_dim,
                # kernel_initializer=tf.initializers.zeros(),
            )


        ### normalization layer
        self.norm_layer = normalization_layer(self.normalization_arr)

        ### initializing weights
        temp = tf.ones(shape=[1, self.in_steps, self.data_dim])
        temp = self.predict(temp)

        return

    @tf.function
    def _warmup(self, inputs, training=None):
        ### Initialize the GRU state.
        states_list = []
        # first step
        prediction = inputs[:, 0, :]
        for j in range(self.num_rnn_layers):
            prediction, *states = self.rnn_cells_list[j](
                prediction,
                states=self.hidden_states_list[j](prediction, training=training),
                training=training
            )
            states_list.append(states[0])

        # remaining number of input steps
        for i in range(1, self.in_steps):
            prediction = inputs[:, i, :]
            #for j in range(inputs.shape[2]):
            #    prediction[:, :, j] -= self.normalization_arr[0, j]
            #    prediction[:, :, j] /= self.normalization_arr[1, j]
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
        #norm_layer = normalization_layer(self.normalization_arr)
        predictions_list = []
        prediction, states_list = self._warmup(inputs, training)

        # first prediction
        predictions_list.append(prediction)

        ### Run the rest of the prediction steps.
        for i in range(1, self.out_steps):
            prediction = self.norm_layer(prediction)
            for j in range(self.num_rnn_layers):
                prediction, *states = self.rnn_cells_list[j](
                    prediction,
                    states=states_list[j],
                    training=training
                )
                states_list[j] = states[0]
            prediction = self.dense(prediction, training=training)
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
        self.save_model_weights(file_name+'_gru_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name, by_name=True, skip_mismatch=True)
        return

    def update_Tinput(self, new_T_input):
        self.T_input = new_T_input
        self.in_steps = int((self.T_input+0.5*self.dt_rnn)//self.dt_rnn)

    def update_Toutput(self, new_T_output):
        self.T_output = new_T_output
        self.out_steps = int((self.T_output+0.5*self.dt_rnn)//self.dt_rnn)

################################################################################
