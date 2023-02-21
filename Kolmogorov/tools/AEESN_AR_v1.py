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
            initial_value=self.normalization_arr[0:1],
            trainable=False,
            name='alpha',
            dtype='float32',
            shape=(1,)+tuple(self.normalization_arr.shape[1:])
        )
        self.beta = tf.Variable(
            initial_value=self.normalization_arr[1:],
            trainable=False,
            name='beta',
            dtype='float32',
            shape=(1,) + tuple(self.normalization_arr.shape[1:])
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
            initial_value=self.normalization_arr[0:1],
            trainable=False,
            name='alpha',
            dtype='float32',
            shape=(1,) + tuple(self.normalization_arr.shape[1:])
        )
        self.beta = tf.Variable(
            initial_value=self.normalization_arr[1:],
            trainable=False,
            name='beta',
            dtype='float32',
            shape=(1,) + tuple(self.normalization_arr.shape[1:])
        )

    def call(self, inputs, training=None):
        # batch_size = inputs.shape[0]
        # if batch_size == None:
        #     batch_size = 1
        return (inputs - self.alpha) / self.beta



class AR_AERNN_ESN(Model):
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
            loss_weights=None,
            clipnorm=None,
            global_clipnorm=None,
        ):
        
        super(AR_AERNN_ESN, self).__init__()

        self.ae_net = ae_net
        self.rnn_net = rnn_net
        self.normalization_preRNN = normalization_layer(rnn_data_normalization_arr)
        self.reverseNormalization_postRNN = reverseNormalization_layer(rnn_data_normalization_arr)
        self.reverseNormalization_postAE = reverseNormalization_layer(ae_data_normalization_arr)
        self.covmat_lmda = covmat_lmda
        self.time_stddev_ogdata = time_stddev_ogdata
        self.time_mean_ogdata = time_mean_ogdata
        self.loss_weights = loss_weights
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm # if this is specified then specifying `clipnorm` has no effect

        return

    # def build(self, input_shape):
    #     super(AR_AERNN_ESN, self).build(input_shape)
        # if self.rnn_net.stateful == False and self.rnn_net.use_learnable_state == False:
        #     for i in range(len(self.rnn_net.init_state)):
        #         self.rnn_net.init_state[i] = [tf.zeros(shape=(input_shape[0], self.rnn_net.rnn_layers_units[i]), dtype='float32')]

    # @tf.function
    def _warmup(self, inputs, training=None, usenoiseflag=None):
        ### Initialize the GRU state.

        states_list = []
        # intermediate_outputs_lst = []

        ### Passing input through the GRU layers
        # first layer
        x = inputs
        x = layers.TimeDistributed(self.ae_net.encoder_net)(
            x,
            training=training
        )
        og_shape = x.shape[2:]
        x = tf.reshape(x, (-1, x.shape[1], self.rnn_net.data_dim))
        x = self.normalization_preRNN(x)
        
        rnnwarmup_return_tuple = self.rnn_net._warmup(
            x,
            training=training,
            usenoiseflag=usenoiseflag,
        )
        output = rnnwarmup_return_tuple[0]
        states_list = rnnwarmup_return_tuple[1]
        if len(rnnwarmup_return_tuple) > 2:
            intermediate_outputs_lst = rnnwarmup_return_tuple[2]
            scalar_multiplier_list = rnnwarmup_return_tuple[3]

        # x, states = self.rnn_net.rnn_list[0](
        #     x,
        #     training=training,
        # )
        # intermediate_outputs_lst.append(x)
        # states_list.append(states)

        # # remaining layers
        # for i in range(self.rnn_net.num_skip_connections):
        #     prediction, states = self.rnn_net.rnn_list[i+1](
        #         x,
        #         training=training,
        #     )
        #     intermediate_outputs_lst.append(prediction)
        #     states_list.append(states)
        #     x = intermediate_outputs_lst[0]
        #     for j in range(i+1):
        #         x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]
        
        # output = x[:, -1:, :]
        # # running through the final dense layers
        # for j in range(len(self.rnn_net.dense)):
        #     output = layers.TimeDistributed(self.rnn_net.dense[j])(output, training=training)

        output = self.reverseNormalization_postRNN(output)
        output = tf.reshape(output, (-1, output.shape[1],)+tuple(og_shape))
        output = layers.TimeDistributed(self.ae_net.decoder_net)(
            output,
            training=training
        )

        if len(rnnwarmup_return_tuple) == 2:
            return_tuple = (output, states_list)
        else:
            return_tuple = (output, states_list, intermediate_outputs_lst, scalar_multiplier_list)

        return return_tuple

    # @tf.function
    def call(self, inputs, training=None, usenoiseflag=False):

        predictions_list = []

        ### warming up the RNN
        warmup_tuple = self._warmup(
            inputs,
            training=False,
            usenoiseflag=usenoiseflag,
        )
        
        x = warmup_tuple[0]
        states_list = warmup_tuple[1]
        if len(warmup_tuple) > 2:
            intermediate_outputs_lst = warmup_tuple[2]
            scalar_multiplier_list = warmup_tuple[3]
        else:
            intermediate_outputs_lst = None
            scalar_multiplier_list = None

        predictions_list.append(x[:, -1])

        ### Passing input through the GRU layers
        for tstep in range(1, self.rnn_net.out_steps):
            x = layers.TimeDistributed(self.ae_net.encoder_net)(x, training=training)
            og_shape = x.shape[2:]
            x = tf.reshape(x, (-1, x.shape[1], self.rnn_net.data_dim))
            x = self.normalization_preRNN(x)

            # x, states = self.rnn_net.rnn_list[0](
            #     x,
            #     initial_state=states_list[0],
            #     training=training,
            # )
            # intermediate_outputs_lst[0] = x
            # states_list[0] = states

            # # remaining layers
            # for i in range(self.rnn_net.num_skip_connections):
            #     prediction, states = self.rnn_net.rnn_list[i+1](
            #         x,
            #         initial_state=states_list[i+1],
            #         training=training,
            #     )
            #     intermediate_outputs_lst[i+1] = prediction
            #     states_list[i+1] = (states)
            #     x = intermediate_outputs_lst[0]
            #     for j in range(i+1):
            #         x += scalar_multiplier_list[int(i*(i+1)/2) + j] * intermediate_outputs_lst[j+1]
            
            # x = x[:, -1:, :]
            # # running through the final dense layers
            # for j in range(len(self.rnn_net.dense)):
            #     x = layers.TimeDistributed(self.rnn_net.dense[j])(x, training=training)

            x, states_list = self.rnn_net.onestep(
                x=x,
                training=training,
                states_list=states_list,
                intermediate_outputs_lst=intermediate_outputs_lst,
                scalar_multiplier_list=scalar_multiplier_list,
            )

            x = self.reverseNormalization_postRNN(x)
            x = tf.reshape(x, (-1, x.shape[1],)+tuple(og_shape))
            x = layers.TimeDistributed(self.ae_net.decoder_net)(x, training=training)

            predictions_list.append(x[:, -1])

        output = tf.stack(predictions_list)
        output = tf.transpose(output, [1, 0, 2, 3, 4])

        return output

    def _warmup_upto_tinputminusone(self, inputs, training=False, usenoiseflag=False):
        ### Initialize the GRU state.

        states_list = []
        # intermediate_outputs_lst = []

        ### Passing input through the GRU layers
        # first layer
        x = inputs[:, 0:-1]
        x = layers.TimeDistributed(self.ae_net.encoder_net)(
            x,
            training=training
        )
        # og_shape = x.shape[2:]
        x = tf.reshape(x, (-1, x.shape[1], self.rnn_net.data_dim))
        x = self.normalization_preRNN(x)
        
        rnnwarmup_return_tuple = self.rnn_net._warmup(
            x,
            training=training,
            usenoiseflag=usenoiseflag,
        )
        # output = rnnwarmup_return_tuple[0]
        states_list = rnnwarmup_return_tuple[1]
        if len(rnnwarmup_return_tuple) > 2:
            intermediate_outputs_lst = rnnwarmup_return_tuple[2]
            scalar_multiplier_list = rnnwarmup_return_tuple[3]

        # output = self.reverseNormalization_postRNN(output)
        # output = tf.reshape(output, (-1, output.shape[1],)+tuple(og_shape))
        # output = layers.TimeDistributed(self.ae_net.decoder_net)(
        #     output,
        #     training=training
        # )

        if len(rnnwarmup_return_tuple) == 2:
            return_tuple = (states_list, )
        else:
            return_tuple = (states_list, intermediate_outputs_lst, scalar_multiplier_list)

        return return_tuple

    def _call_for_train(self, inputs, _warmup_upto_tinputminusone_tuple, training=False):
        x = inputs
        states_list = _warmup_upto_tinputminusone_tuple[0]
        if len(_warmup_upto_tinputminusone_tuple) > 1:
            intermediate_outputs_lst = _warmup_upto_tinputminusone_tuple[1]
            scalar_multiplier_list = _warmup_upto_tinputminusone_tuple[2]
        else:
            intermediate_outputs_lst = None
            scalar_multiplier_list = None

        predictions_list = []
        ### Passing input through the GRU layers
        for tstep in range(self.rnn_net.out_steps):
            x = layers.TimeDistributed(self.ae_net.encoder_net)(x, training=training)
            og_shape = x.shape[2:]
            x = tf.reshape(x, (-1, x.shape[1], self.rnn_net.data_dim))
            x = self.normalization_preRNN(x)

            x, states_list = self.rnn_net.onestep(
                x=x,
                training=training,
                states_list=states_list,
                intermediate_outputs_lst=intermediate_outputs_lst,
                scalar_multiplier_list=scalar_multiplier_list,
            )

            x = self.reverseNormalization_postRNN(x)
            x = tf.reshape(x, (-1, x.shape[1],)+tuple(og_shape))
            x = layers.TimeDistributed(self.ae_net.decoder_net)(x, training=training)

            predictions_list.append(x[:, -1])

        output = tf.stack(predictions_list)
        output = tf.transpose(output, [1, 0, 2, 3, 4])
        
        return output

    def train_step(self, data):
        # x, y = data
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        sw_cov = 1.0 if sample_weight is None else sample_weight
        
        # print(x.shape, y.shape, sample_weight, sw_cov)
        
        _warmup_upto_tinputminusone_tuple = self._warmup_upto_tinputminusone(
            x, training=True, usenoiseflag=True
        )
        
        with tf.GradientTape() as tape:
            # ypred = self.call(x, training=True, usenoiseflag=True)
            ypred = self._call_for_train(
                x[:, -1:], _warmup_upto_tinputminusone_tuple, training=True
            )
            loss = self.compiled_loss(
                y,
                ypred,
                sample_weight,
                regularization_losses=self.losses
            )
            ypred_renormalized = self.reverseNormalization_postAE(ypred) - self.time_mean_ogdata

            ytrue_renormalized = self.reverseNormalization_postAE(y) - self.time_mean_ogdata
            
            if isinstance(self.loss_weights, type(None)) == False:
                ypred_renormalized = ypred_renormalized * self.loss_weights**0.5
                ytrue_renormalized = ytrue_renormalized * self.loss_weights**0.5

            TKE_pred = tf.math.reduce_sum(ypred_renormalized**2, axis=-3)
            TKE_true = tf.math.reduce_sum(ytrue_renormalized**2, axis=-3)
            
            TKE_fro_loss = tf.norm(
                TKE_true - TKE_pred,
                ord='fro',
                axis=[-2, -1]
            )
            TKE_fro_loss = self.covmat_lmda * sw_cov * tf.math.reduce_mean(TKE_fro_loss)
            # print(loss.shape)
            loss = loss + TKE_fro_loss
            # print(loss.shape)
            # print(tf.norm(covmat_true - covmat_pred, ord='fro', axis=[-2, -1]))
        self._validate_target_and_loss(ypred, loss)

        # print(x.shape, y.shape, loss.shape, sample_weight, sw_cov)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # print(gradients)
        # print(type(gradients))
        if self.global_clipnorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.global_clipnorm)
        elif self.clipnorm is not None:
            global_clipnorm = 0.0
            for elem in gradients:
                if elem is not None:
                    global_clipnorm += self.clipnorm**2
            global_clipnorm = global_clipnorm**0.5
            gradients, _ = tf.clip_by_global_norm(gradients, global_clipnorm)

        global_gradnorm = 0.0
        for elem in gradients:
            if elem is not None:
                global_gradnorm += tf.norm(elem, ord='euclidean')**2
        global_gradnorm = global_gradnorm**0.5

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return self.compute_metrics(x, y, ypred, sample_weight, TKE_fro_loss, global_gradnorm)
    
    def compute_metrics(self, x, y, y_pred, sample_weight, TKE_fro_loss=0.0, global_gradnorm=0.0):
        metric_results = super(AR_AERNN_ESN, self).compute_metrics(x, y, y_pred, sample_weight)
        # return_dict = self.get_metrics_result()
        metric_results['covmat_fro_loss'] = TKE_fro_loss
        metric_results['global_gradnorm'] = global_gradnorm
        if type(self.rnn_net.train_rho_res) != type(None):
            for i in range(len(self.rnn_net.rnn_list)):
                metric_results['rho_res_{}'.format(i)] = self.rnn_net.rnn_list[i].cell.rho_res
        if type(self.rnn_net.train_alpha) != type(None):
            for i in range(len(self.rnn_net.rnn_list)):
                metric_results['alpha_{}'.format(i)] = self.rnn_net.rnn_list[i].cell.alpha
        if type(self.rnn_net.train_omega_in) != type(None):
            for i in range(len(self.rnn_net.rnn_list)):
                metric_results['omega_in_{}'.format(i)] = self.rnn_net.rnn_list[i].cell.omega_in
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
