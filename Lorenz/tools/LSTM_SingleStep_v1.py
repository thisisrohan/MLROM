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

class RNN_LSTM(Model):
    """
    Single-step LSTM network that advances (in time) the latent space representation
    """
    def __init__(
            self, data_dim=None,
            # in_steps=None,
            # out_steps=None,
            dt_rnn=None,
            lambda_reg=0.0,
            reg_name='L2',
            rnn_layers_units=[3, 3, 3],
            dense_layer_act_func='linear',
            load_file=None):
        
        super(RNN_LSTM, self).__init__()

        self.load_file = load_file
        if self.load_file == None:
            self.data_dim = data_dim
            # self.out_steps = out_steps
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
            # self.out_steps = load_dict['out_steps']
            self.dt_rnn = load_dict['dt_rnn']
            self.lambda_reg = load_dict['lambda_reg']
            self.reg_name = load_dict['reg_name']
            self.rnn_layers_units = load_dict['rnn_layers_units']
            self.dense_layer_act_func = load_dict['dense_layer_act_func']
        self.num_rnn_layers = len(self.rnn_layers_units)


        ### input vector
        input_shape_t = (None, self.data_dim)  # [in_timesteps, data_dim]
        input_vec_t = Input(shape=input_shape_t)


        ### the LSTM network
        self.rnn_layers_list = []
        if self.reg_name != None and self.lambda_reg != None and self.lambda_reg != 0:
            reg = eval('tf.keras.regularizers.'+self.reg_name)
            self.rnn_cells_list = [
                layers.LSTMCell(
                    units=units,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg)
                ) for units in self.rnn_layers_units
            ]

            self.dense = layers.Dense(
                self.data_dim,
                # kernel_initializer=tf.initializers.zeros(),
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            )
        else:
            self.rnn_cells_list = [
                layers.LSTMCell(units=units) for units in self.rnn_layers_units
            ]
            self.dense = layers.Dense(
                self.data_dim,
                # kernel_initializer=tf.initializers.zeros(),
            )

        self.rnn_layers_list.extend(
            [
                layers.RNN(
                    self.rnn_cells_list[i],
                    return_sequences=True,
                    return_state=False,
                ) for i in range(self.num_rnn_layers)
            ]
        )


        ### defining the LSTM model
        x = self.rnn_layers_list[0](input_vec_t)
        # x, state = self.rnn_layers_list[0](input_vec_t)
        for i in range(1, self.num_rnn_layers):
            x = self.rnn_layers_list[i](x)
            # x, state = self.rnn_layers_list[i](x)
        lstm_output = layers.TimeDistributed(self.dense)(x)
        # x = self.dense(x)

        self.LSTM_net = Model(inputs=input_vec_t, outputs=lstm_output)

        return


    @tf.function
    def call(self, inputs):

        prediction = self.LSTM_net(inputs)
        # prediction = self.rnn_layers_list[0](inputs)
        # for i in range(1, self.num_rnn_layers):
        #     prediction = self.rnn_layers_list[i](prediction)
        # prediction = layers.TimeDistributed(self.dense)(prediction)

        return prediction

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


################################################################################
