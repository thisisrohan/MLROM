import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

################################################################################

class Autoencoder(Model):
    """
    Autoencoder network that finds the latent space representation
    """
    def __init__(
            self, data_dim,
            enc_layers=[16,12,8,8,4,4,2],
            dec_layers=[2,4,4,8,8,12,16],
            latent_space_dim=2,
            lambda_reg=0.0,
            enc_layer_act_func='elu',
            enc_final_layer_act_func='tanh',
            dec_layer_act_func='elu',
            dec_final_layer_act_func='linear'):
        
        super(Autoencoder, self).__init__()
        # the encoder network
        encoder_layers_list = [
            layers.Dense(
                neurons,
                activation=enc_final_layer_act_func,
                kernel_regularizer=L2(lambda_reg),
                bias_regularizer=L2(lambda_reg)
            ) for neurons in enc_layers
        ]
        encoder_layers_list.append(
            layers.Dense(
                latent_space_dim,
                activation=enc_final_layer_act_func,
                kernel_regularizer=L2(lambda_reg),
                bias_regularizer=L2(lambda_reg)
            )
        )
        self.encoder_net = tf.keras.Sequential(encoder_layers_list)

        # the decoder network
        decoder_layers_list = [
            layers.Dense(
                neurons,
                activation=dec_layer_act_func,
                kernel_regularizer=L2(lambda_reg),
                bias_regularizer=L2(lambda_reg)
            ) for neurons in dec_layers
        ]
        decoder_layers_list.append(
            layers.Dense(
                data_dim+3,
                activation=dec_final_layer_act_func,
                kernel_regularizer=L2(lambda_reg),
                bias_regularizer=L2(lambda_reg)
            )
        )
        self.decoder_net = tf.keras.Sequential(decoder_layers_list)

        self.ae_net = tf.keras.Sequential()

        return

    def call(self, x):

        encoded = self.encoder_net(x)
        decoded = self.decoder_net(encoded)

        return decoded

################################################################################