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

    def predict(self, out_time, inputs, min_warmup_steps=10, training=False):

        in_steps = inputs.shape[1]
        out_steps = int((out_time +0.25*self.SingleStepLSTM_model.dt_rnn)//self.SingleStepLSTM_model.dt_rnn)
        num_rnn_layers = len(self.SingleStepLSTM_model.rnn_cells_list)
        # print('in_steps:{}, out_steps:{}, num_rnn_layers:{}, input_shape:{}'.format(in_steps, out_steps, num_rnn_layers, inputs.shape))

        states_list = []
        predictions_list = []

        # first step
        prediction = inputs
        for j in range(num_rnn_layers):
            prediction, *states = self.SingleStepLSTM_model.rnn_cells_list[j](
                prediction,
                states=[
                    self.SingleStepLSTM_model.hidden_states_list[j][0](prediction, training=training),
                    self.SingleStepLSTM_model.hidden_states_list[j][1](prediction, training=training)
                ],
                training=training
            )
            states_list.append(states[0])
        prediction = self.SingleStepLSTM_model.dense(prediction, training=training)
        predictions_list.append(prediction)

        # remaining output steps
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


