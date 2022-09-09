# import os
# import numpy as np
# from scipy import linalg

# import time as time

import tensorflow as tf
from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K
# from tensorflow.keras.regularizers import L2

################################################################################
############################## Autoregressive v1 ###############################

class Autoregressive_RNN:
    """
    Autoregressive extension of the provided RNN model, no warm-up
    """
    def __init__(
            self,
            SingleStepLSTM_model,
            ):

        self.SingleStepLSTM_model = SingleStepLSTM_model
        return

    def predict(self, out_time, inputs, min_warmup_steps=10):

        in_steps = inputs.shape[1]
        out_steps = int(out_time // self.SingleStepLSTM_model.dt_rnn)
        num_rnn_layers = len(self.SingleStepLSTM_model.rnn_layers_list)
        # print('in_steps:{}, out_steps:{}, num_rnn_layers:{}, input_shape:{}'.format(in_steps, out_steps, num_rnn_layers, inputs.shape))

        states_list = []
        predictions_list = []

        # warmup
        # for i in range(in_steps):
            # temp_ = self.SingleStepLSTM_model.predict(inputs[:, i:i+1, :])
        # prediction = inputs
        # for j in range(num_rnn_layers):
        #     prediction, *states = self.SingleStepLSTM_model.rnn_layers_list[j](inputs[:, 0:1, :], training=False)
        #     states_list.append(states)
        # prediction = layers.TimeDistributed(self.SingleStepLSTM_model.dense)(prediction, training=False)
        
        # for i in range(1, in_steps):
        #     for j in range(num_rnn_layers):
        #         prediction, *states = self.SingleStepLSTM_model.rnn_cells_list[j](
        #             prediction[:, i, :],
        #             states=states_list[j],
        #             training=False
        #         )
        #         states_list[j] = states[0]
        #     prediction = layers.TimeDistributed(self.SingleStepLSTM_model.dense)(prediction, training=False)

        prediction = inputs
        for j in range(num_rnn_layers):
            self.SingleStepLSTM_model.rnn_layers_list[j].return_state = True
            prediction, *states = self.SingleStepLSTM_model.rnn_layers_list[j](prediction, training=False)
            states_list.append(states)

        prediction = prediction[:, 0, :]
        prediction = self.SingleStepLSTM_model.dense(prediction, training=False)


        # first prediction
        # prediction = prediction[:, 0, :]
        predictions_list.append(prediction)

        
        for i in range(1, out_steps):
            # x = prediction[:, 0, :]
            for j in range(num_rnn_layers):
                # print('i:{}, j:{}, prediction.shape:{}'.format(i, j, prediction.shape))
                prediction, *states = self.SingleStepLSTM_model.rnn_cells_list[j](
                    prediction,
                    states=states_list[j],
                    training=False
                )
                states_list[j] = states[0]
            prediction = self.SingleStepLSTM_model.dense(prediction, training=False)
            predictions_list.append(prediction)

        # predictions_list.shape => (time, batch, features)
        predictions = tf.stack(predictions_list)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])


        return predictions

################################################################################


