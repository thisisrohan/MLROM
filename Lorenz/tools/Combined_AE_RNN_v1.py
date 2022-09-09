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
######################## Combined AE RNN for Training v1 #######################

class Combined_AE_RNN(Model):
    """
    Combined AE and RNN for training together, after having been trained
    separately
    """
    def __init__(
            self, data_dim,
            AE_net,
            RNN_net,
            # input_time,
            load_dir=None):
        
        super(Combined_AE_RNN, self).__init__()

        self.load_dir = load_dir
        if self.load_dir == None:
            self.data_dim = data_dim
            self.AE_net = AE_net
            self.RNN_net = RNN_net
            # self.input_time = input_time
        else:
            pass # for now


        ### input vector
        input_shape_t = (None, self.data_dim)  # [in_timesteps, data_dim]
        input_vec_t = Input(shape=input_shape_t)


        ### Combined network
        x = layers.TimeDistributed(self.AE_net.encoder_net)(input_vec_t)
        x = self.RNN_net(x)
        x = layers.TimeDistributed(self.AE_net.decoder_net)(x)

        self.Combined_model = Model(inputs=input_vec_t, outputs=x)

        return


    @tf.function
    def call(self, inputs):

        prediction = self.Combined_model(inputs)
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

    def save_everything(self, save_dir, dir_sep='/', file_name='combined_net', H5=True):

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        ### saving class attributes
        save_file = save_dir + dir_sep + file_name
        class_dict = {
            'data_dim':self.data_dim,
            'load_dir':self.load_dir,
        }
        with open(save_file+'_class_dict.txt', 'w') as f:
            f.write(str(class_dict))
            # s = '{\n'
            # for entry in class_dict.keys():
            #     s += '    '+str(entry)+':'+str(class_dict[entry])+',\n'
            # s += '}'
            # f.write(s)

        ### saving the autoencoder
        dir_ae = save_dir + dir_sep + 'AE'
        if not os.path.isdir(dir_ae):
            os.makedirs(dir_ae)
        save_file_ae = dir_ae + dir_sep + 'final_net'
        self.AE_net.save_everything(save_file_ae)

        ### saving the RNN
        dir_rnn = save_dir + dir_sep + 'RNN'
        if not os.path.isdir(dir_rnn):
            os.makedirs(dir_rnn)
        save_file_rnn = dir_rnn + dir_sep + 'final_net'
        self.RNN_net.save_everything(save_file_rnn)

        ### saving weights
        self.save_model_weights(save_file + '_combined_ae_rnn_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name)
        return


################################################################################
