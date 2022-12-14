################################################################################
# RK Methods inspired residual AE-GRU with skip connections, with              #
# uniform/normal noise added to every input and learnable initial states.      #
# [STATEFUL]                                                                   #
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
from keras.engine import data_adapter

array = np.array
float64 = np.float64
float32 = np.float32
int64 = np.int64
int32 = np.int32

################################################################################
#################################### LSTM V4 ###################################

class reverseNormalization_layer(layers.Layer):
    def __init__(self, normalization_arr):
        super(reverseNormalization_layer, self).__init__()
        self.normalization_arr = normalization_arr
        self.alpha = tf.Variable(
            initial_value=self.normalization_arr[0:1, :],
            trainable=False,
            name='alpha',
            dtype='float32',
            shape=(1, self.normalization_arr.shape[1])
        )
        self.beta = tf.Variable(
            initial_value=self.normalization_arr[1:, :],
            trainable=False,
            name='beta',
            dtype='float32',
            shape=(1, self.normalization_arr.shape[1])
        )

    def call(self, inputs, training=None):
        # batch_size = inputs.shape[0]
        # if batch_size == None:
        #     batch_size = 1
        return inputs * self.beta + self.alpha


class normalization_layer(layers.Layer):
    def __init__(self, normalization_arr):
        super(normalization_layer, self).__init__()
        self.normalization_arr = normalization_arr
        self.alpha = tf.Variable(
            initial_value=self.normalization_arr[0:1, :],
            trainable=False,
            name='alpha',
            dtype='float32',
            shape=(1, self.normalization_arr.shape[1])
        )
        self.beta = tf.Variable(
            initial_value=self.normalization_arr[1:, :],
            trainable=False,
            name='beta',
            dtype='float32',
            shape=(1, self.normalization_arr.shape[1])
        )

    def call(self, inputs, training=None):
        # batch_size = inputs.shape[0]
        # if batch_size == None:
        #     batch_size = 1
        return (inputs - self.alpha) / self.beta



class AR_AERNN_GRU(Model):
    """
    Single-step GRU network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self,
            ae_net,
            rnn_net,
            rnn_data_normalization_arr,
            ae_data_normalization_arr,
            covmat_lmda,
            time_stddev_ogdata,
            time_mean_ogdata,
        ):
        
        super(AR_AERNN_GRU, self).__init__()

        self.ae_net = ae_net
        self.rnn_net = rnn_net
        self.normalization_preRNN = normalization_layer(rnn_data_normalization_arr)
        self.reverseNormalization_postRNN = reverseNormalization_layer(rnn_data_normalization_arr)
        self.reverseNormalization_postAE = reverseNormalization_layer(ae_data_normalization_arr)
        self.covmat_lmda = covmat_lmda
        self.time_stddev_ogdata = time_stddev_ogdata
        self.time_mean_ogdata = time_mean_ogdata

        return

    def build(self, input_shape):
        super(AR_AERNN_GRU, self).build(input_shape)
        # if self.rnn_net.stateful == False and self.rnn_net.use_learnable_state == False:
        #     for i in range(len(self.rnn_net.init_state)):
        #         self.rnn_net.init_state[i] = [tf.zeros(shape=(input_shape[0], self.rnn_net.rnn_layers_units[i]), dtype='float32')]

    # @tf.function
    def _warmup(self, inputs, scalar_multiplier_list, training=None, usenoiseflag=None):
        ### Initialize the GRU state.

        states_list = []
        intermediate_outputs_lst = []

        ### Passing input through the GRU layers
        # first layer
        x = inputs
        x = layers.TimeDistributed(self.ae_net.encoder_net)(x, training=training)
        x = self.normalization_preRNN(x)
        x, states = self.rnn_net.rnn_list[0](
            x,
            training=training,
        )
        intermediate_outputs_lst.append(x)
        states_list.append(states)

        # remaining layers
        for i in range(self.rnn_net.num_skip_connections):
            prediction, states = self.rnn_net.rnn_list[i+1](
                x,
                training=training,
            )
            intermediate_outputs_lst.append(prediction)
            states_list.append(states)
            x = intermediate_outputs_lst[0]
            for j in range(i+1):
                x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]
        
        output = x[:, -1:, :]
        # running through the final dense layers
        for j in range(len(self.rnn_net.dense)):
            output = layers.TimeDistributed(self.rnn_net.dense[j])(output, training=training)

        output = self.reverseNormalization_postRNN(output)
        output = layers.TimeDistributed(self.ae_net.decoder_net)(output, training=training)

        return output, states_list, intermediate_outputs_lst

    # @tf.function
    def call(self, inputs, training=None, usenoiseflag=False):

        predictions_list = []

        # computing the scalar multipliers
        if type(self.rnn_net.scalar_weights) != type(None):
            scalar_multiplier_list = self.rnn_net.scalar_weights #* self.dt_rnn
        else:
            scalar_multiplier_list = []
            fac = self.rnn_net.dt_rnn
            for i in range(self.rnn_net.num_skip_connections):
                sum_ = 0.0
                for j in range(i+1):
                    sum_ += tf.math.exp(self.rnn_net.scalar_multiplier_pre_list[int(i*(i+1)/2) + j])
                for j in range(i+1):
                    scalar_multiplier_list.append(
                        tf.math.exp(fac * self.rnn_net.scalar_multiplier_pre_list[int(i*(i+1)/2) + j]) / sum_ # not corrected, weights in a row need not sum to one, they sum to the fractional time step
                    )

        ### warming up the RNN
        x, states_list, intermediate_outputs_lst = self._warmup(
            inputs,
            scalar_multiplier_list,
            training=False,
            usenoiseflag=usenoiseflag
        )
        predictions_list.append(x[:, -1, :])


        ### Passing input through the GRU layers
        for tstep in range(1, self.rnn_net.out_steps):
            x = layers.TimeDistributed(self.ae_net.encoder_net)(x, training=training)
            x = self.normalization_preRNN(x)

            x, states = self.rnn_net.rnn_list[0](
                x,
                initial_state=states_list[0],
                training=training,
            )
            intermediate_outputs_lst[0] = x
            states_list[0] = states

            # remaining layers
            for i in range(self.rnn_net.num_skip_connections):
                prediction, states = self.rnn_net.rnn_list[i+1](
                    x,
                    initial_state=states_list[i+1],
                    training=training,
                )
                intermediate_outputs_lst[i+1] = prediction
                states_list[i+1] = (states)
                x = intermediate_outputs_lst[0]
                for j in range(i+1):
                    x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]
            
            x = x[:, -1:, :]
            # running through the final dense layers
            for j in range(len(self.rnn_net.dense)):
                x = layers.TimeDistributed(self.rnn_net.dense[j])(x, training=training)

            x = self.reverseNormalization_postRNN(x)
            x = layers.TimeDistributed(self.ae_net.decoder_net)(x, training=training)

            predictions_list.append(x[:, -1, :])

        output = tf.stack(predictions_list)
        output = tf.transpose(output, [1, 0, 2])

        return output

    def train_step(self, data):
        # x, y = data
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        sw_cov = 1.0 if sample_weight is None else sample_weight
        
        with tf.GradientTape() as tape:
            ypred = self.call(x, training=True, usenoiseflag=True)
            loss = self.compiled_loss(
                y,
                ypred,
                sample_weight,
                regularization_losses=self.losses
            )
            ypred_renormalized = self.reverseNormalization_postAE(ypred) - self.time_mean_ogdata

            ytrue_renormalized = self.reverseNormalization_postAE(y) - self.time_mean_ogdata

            covmat_pred = tf.linalg.matmul(
                ypred_renormalized,
                ypred_renormalized,
                transpose_a=True,
            )
            covmat_true = tf.linalg.matmul(
                ytrue_renormalized,
                ytrue_renormalized,
                transpose_a=True,
            )
            
            covmat_fro_loss = self.covmat_lmda * sw_cov * tf.norm(
                (covmat_true - covmat_pred) / self.time_stddev_ogdata**2,
                ord='fro',
                axis=[-2, -1]
            )
            loss = loss + covmat_fro_loss
            # print(tf.norm(covmat_true - covmat_pred, ord='fro', axis=[-2, -1]))
        self._validate_target_and_loss(ypred, loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return self.compute_metrics(x, y, ypred, sample_weight, covmat_fro_loss)
    
    def compute_metrics(self, x, y, y_pred, sample_weight, covmat_fro_loss=0.0):
        metric_results = super(AR_AERNN_GRU, self).compute_metrics(x, y, y_pred, sample_weight)
        # return_dict = self.get_metrics_result()
        metric_results['covmat_fro_loss'] = covmat_fro_loss
        return metric_results
    

    def save_everything(self, file_name, H5=True):

        ### saving class attributes
        self.ae_net.save_class_dict(file_name+'_ae_class_dict.txt')
        self.rnn_net.save_class_dict(file_name+'_rnn_class_dict.txt')

        ### saving weights
        self.ae_net.save_model_weights(file_name+'_ae_weights', H5=H5)
        self.rnn_net.save_model_weights(file_name+'_rnn_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name)
        return


################################################################################
