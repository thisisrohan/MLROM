import os
import numpy as np
from scipy import linalg

import time as time

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

################################################################################
#################################### LSTM V1 ###################################

# setting up the RNN class
class LSTM_v1(Model):
    """
    Single-shot LSTM network that advances (in time) the latent space representation
    """
    def __init__(self, num_layers=2, num_hidden_units=3, latent_space_dim=2):
        
        super(LSTM_v1, self).__init__()

        self.num_layers = num_layers

        if num_layers > 1:
            self.rnn_layers_list = [layers.LSTM(num_hidden_units, return_sequences=True) for i in range(num_layers-1)]
            self.rnn_layers_list.append(layers.LSTM(num_hidden_units, return_sequences=False))
        else:
            self.rnn_layers_list = [layers.LSTM(num_hidden_units, return_sequences=False)]
        self.rnn_layers_list.append(layers.Dense(num_sample_output*latent_space_dim, kernel_initializer=tf.initializers.zeros()))
        self.rnn_layers_list.append(layers.Reshape([num_sample_output, latent_space_dim]))

        self.net = tf.keras.Sequential(self.rnn_layers_list)

        return

    def call(self, x):

        output = self.net(x)

        return output


################################################################################
#################################### LSTM V2 ###################################

# setting up the RNN class
class LSTM_v2(Model):
    """
    Autoregressive LSTM network that advances (in time) the latent space representation
    """
    def __init__(self, out_steps, num_layers=2, num_hidden_units=3, latent_space_dim=2):
        
        super(LSTM_v2, self).__init__()

        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.latent_sapce_dim = latent_space_dim
        self.out_steps = out_steps

        self.rnn_cells_list = [layers.LSTMCell(num_hidden_units) for i in range(num_layers)]
        
        self.rnn_layers_list = []
        if num_layers > 1:
            self.rnn_layers_list.extend([layers.RNN(self.rnn_cells_list[i], return_sequences=True, return_state=True) for i in range(num_layers-1)])
        self.rnn_layers_list.append(layers.RNN(self.rnn_cells_list[-1], return_sequences=False, return_state=True))
        
        self.dense = layers.Dense(latent_space_dim, kernel_initializer=tf.initializers.zeros())

        # self.rnn = tf.keras.Sequential(self.rnn_layers_list)

        return

    def warmup(self, inputs, training=None, min_warmup_steps=10):
        """
        warming up the internal states of the RNN cells
        """
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]

        # x = inputs

        if num_time_steps >= min_warmup_steps:
            for j in range(num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
        else:
            x = inputs[:, 0, :]
            for j in range(min_warmup_steps-num_time_steps):
                # print('\n---j:{}---'.format(j))
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
                x = self.dense(x)
            
            for j in range(0, num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]

        # Convert the lstm output to a prediction.
        x = self.dense(x)
            
        
        # for i in range(self.num_layers-1):
            # x, *states = self.rnn_layers_list[i](x)
            # states_list.append(states)
        # x, *states = self.rnn_layers_list[-1](x)
        # states_list.append(states)

        # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
        ## predictions.shape => (batch, features)
        # prediction = self.dense(x)

        return x, states_list

    def warmup_v2(self, inputs, min_warmup_steps=10):
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]
        
        if num_time_steps < min_warmup_steps:
            temp_ = tf.zeros(shape=inputs.shape)
            for i in range(0, min_warmup_steps-num_time_steps):
                temp_[:, i, :] = inputs[:, 0, :]
            temp_[:, min_warmup_steps-num_time_steps:, :] = inputs[:, :, :]
            inputs = temp_
            num_time_steps = min_warmup_steps
        
        for j in range(num_time_steps):
            # print('\n---j:{}---'.format(j))
            x = inputs[:, j, :]
            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    # training=training
                )
                states_list[i] = states[0]
        
        # Convert the lstm output to a prediction.
        x = self.dense(x)

        return x, states_list
        
    def call(self, x, training=None):

        predictions_list = []

        # Initialize the LSTM state.
        # prediction, states_list = self.warmup(x, training)
        prediction, states_list = self.warmup_v2(x)

        # Insert the first prediction.
        predictions_list.append(prediction)

        # Run the rest of the prediction steps.
        for j in range(1, self.out_steps):
            # print('\n---j:{}---'.format(j))
            # Use the last prediction as input.
            x = prediction

            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    training=training
                )
                states_list[i] = states[0]

            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions_list.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions_list = tf.stack(predictions_list)
        # predictions.shape => (batch, time, features)
        predictions_list = tf.transpose(predictions_list, [1, 0, 2])

        return predictions_list


################################################################################
#################################### LSTM V3 ###################################

# setting up the RNN class
class LSTM_v3(Model):
    """
    Autoregressive LSTM network that advances (in time) the latent space representation
    """
    def __init__(self, out_steps, num_layers=2, num_hidden_units=3, latent_space_dim=2):
        
        super(LSTM_v3, self).__init__()

        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.latent_space_dim = latent_space_dim
        self.out_steps = out_steps

        self.rnn_cells_list = [layers.LSTMCell(num_hidden_units) for i in range(num_layers)]
        
        self.rnn_layers_list = []
        if num_layers > 1:
            self.rnn_layers_list.extend([layers.RNN(self.rnn_cells_list[i], return_sequences=True, return_state=True) for i in range(num_layers-1)])
        self.rnn_layers_list.append(layers.RNN(self.rnn_cells_list[-1], return_sequences=False, return_state=True))
        
        self.dense = layers.Dense(latent_space_dim, kernel_initializer=tf.initializers.zeros())

        # self.rnn = tf.keras.Sequential(self.rnn_layers_list)

        return

    def warmup_v1(self, inputs):
        
        states_list = []
        x = inputs
        
        for i in range(self.num_layers-1):
            x, *states = self.rnn_layers_list[i](x)
            states_list.append(states)
        x, *states = self.rnn_layers_list[-1](x)
        states_list.append(states)

        # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
        # predictions.shape => (batch, features)
        x = self.dense(x)

        return x, states_list

    
    def warmup_v2(self, inputs, training=None, min_warmup_steps=10):
        """
        warming up the internal states of the RNN cells
        """
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]

        # x = inputs

        if num_time_steps >= min_warmup_steps:
            for j in range(num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
        else:
            x = inputs[:, 0, :]
            for j in range(min_warmup_steps-num_time_steps):
                # print('\n---j:{}---'.format(j))
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
                x = self.dense(x)
            
            for j in range(0, num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]

        # Convert the lstm output to a prediction.
        x = self.dense(x)
            
        
        # for i in range(self.num_layers-1):
            # x, *states = self.rnn_layers_list[i](x)
            # states_list.append(states)
        # x, *states = self.rnn_layers_list[-1](x)
        # states_list.append(states)

        # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
        ## predictions.shape => (batch, features)
        # prediction = self.dense(x)

        return x, states_list

    def warmup_v3(self, inputs, min_warmup_steps=10):
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]
        
        if num_time_steps < min_warmup_steps:
            input_np = inputs.numpy()
            temp_ = np.zeros(shape=inputs.shape)
            for i in range(0, min_warmup_steps-num_time_steps):
                temp_[:, i, :] = inputs_np[:, 0, :]
            temp_[:, min_warmup_steps-num_time_steps:, :] = inputs_np[:, :, :]
            inputs = tf.convert_to_tensor(temp_)
            num_time_steps = min_warmup_steps
        
        for j in range(num_time_steps):
            # print('\n---j:{}---'.format(j))
            x = inputs[:, j, :]
            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    # training=training
                )
                states_list[i] = states[0]
        
        # Convert the lstm output to a prediction.
        x = self.dense(x)

        return x, states_list

    def warmup_v4(self, inputs, min_warmup_steps=10):
        return
        
    def call(self, x, training=None):

        predictions_list = []

        # Initialize the LSTM state.
        # prediction, states_list = self.warmup(x, training)
        prediction, states_list = self.warmup_v1(x)

        # Insert the first prediction.
        predictions_list.append(prediction)

        # Run the rest of the prediction steps.
        for j in range(1, self.out_steps):
            # print('\n---j:{}---'.format(j))
            # Use the last prediction as input.
            x = prediction

            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                # print(x.shape)
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    training=training
                )
                states_list[i] = states[0]

            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions_list.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions_list = tf.stack(predictions_list)
        # predictions.shape => (batch, time, features)
        predictions_list = tf.transpose(predictions_list, [1, 0, 2])

        return predictions_list

################################################################################
#################################### LSTM V4 ###################################


# setting up the RNN class
class LSTM_v4(Model):
    """
    Autoregressive LSTM network that advances (in time) the latent space representation
    """
    def __init__(self, out_steps, dt_rnn, num_layers=2, num_hidden_units=3, latent_space_dim=2):
        
        super(LSTM_v3, self).__init__()

        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.latent_space_dim = latent_space_dim
        self.out_steps = out_steps
        self.dt_rnn = dt_rnn

        self.rnn_cells_list = [layers.LSTMCell(num_hidden_units) for i in range(num_layers)]
        
        self.rnn_layers_list = []
        if num_layers > 1:
            self.rnn_layers_list.extend([layers.RNN(self.rnn_cells_list[i], return_sequences=True, return_state=True) for i in range(num_layers-1)])
        self.rnn_layers_list.append(layers.RNN(self.rnn_cells_list[-1], return_sequences=False, return_state=True))
        
        self.dense = layers.Dense(latent_space_dim, kernel_initializer=tf.initializers.zeros())

        # self.rnn = tf.keras.Sequential(self.rnn_layers_list)

        return

    def warmup_v1(self, inputs):
        
        states_list = []
        x = inputs
        
        for i in range(self.num_layers-1):
            x, *states = self.rnn_layers_list[i](x)
            states_list.append(states)
        x, *states = self.rnn_layers_list[-1](x)
        states_list.append(states)

        # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
        # predictions.shape => (batch, features)
        x = self.dense(x)

        return x, states_list

    
    def warmup_v2(self, inputs, training=None, min_warmup_steps=10):
        """
        warming up the internal states of the RNN cells
        
        if num_time_steps >= min_warmup_steps:
            regular warm up
        else:
            warm up internal states using first time step of
            the `inputs` for `min_warmup_steps-num_time_steps`
            'steps'. Then proceed with regular warm up with
            these internal states.
        """
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]

        # x = inputs

        if num_time_steps >= min_warmup_steps:
            for j in range(num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
        else:
            x = inputs[:, 0, :]
            for j in range(min_warmup_steps-num_time_steps):
                # print('\n---j:{}---'.format(j))
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]
                x = self.dense(x)
            
            for j in range(0, num_time_steps):
                # print('\n---j:{}---'.format(j))
                x = inputs[:, j, :]
                # Execute one lstm step.
                for i in range(self.num_layers):
                    # print('i:{}'.format(i))
                    # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                    x, *states = self.rnn_cells_list[i](
                        x,
                        states=states_list[i],
                        training=training
                    )
                    states_list[i] = states[0]

        # Convert the lstm output to a prediction.
        x = self.dense(x)
            
        
        # for i in range(self.num_layers-1):
            # x, *states = self.rnn_layers_list[i](x)
            # states_list.append(states)
        # x, *states = self.rnn_layers_list[-1](x)
        # states_list.append(states)

        # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
        ## predictions.shape => (batch, features)
        # prediction = self.dense(x)

        return x, states_list

    def warmup_v3(self, inputs, min_warmup_steps=10):
        """
        warming up the internal states of the RNN cells
        
        if num_time_steps >= min_warmup_steps:
            regular warm up
        else:
            attach `min_warmup_steps-num_time_steps` steps
            of the first time step of the `inputs` to the input
            and then proceed with regular warm up.
        """
        
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        states_list = []
        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])
        num_time_steps = inputs.shape[1]
        
        if num_time_steps < min_warmup_steps:
            input_np = inputs.numpy()
            temp_ = np.zeros(shape=inputs.shape)
            for i in range(0, min_warmup_steps-num_time_steps):
                temp_[:, i, :] = inputs_np[:, 0, :]
            temp_[:, min_warmup_steps-num_time_steps:, :] = inputs_np[:, :, :]
            inputs = tf.convert_to_tensor(temp_)
            num_time_steps = min_warmup_steps
        
        for j in range(num_time_steps):
            # print('\n---j:{}---'.format(j))
            x = inputs[:, j, :]
            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    # training=training
                )
                states_list[i] = states[0]
        
        # Convert the lstm output to a prediction.
        x = self.dense(x)

        return x, states_list

    def warmup_v4(self, inputs, training=None, min_warmup_steps=10):
        """
        warming up the internal states of the RNN cells
        
        warm up internal states from all zeroes to something
        using the first time step of the `inputs` and `min_warmup_steps`
        """

        states_list = []
        num_time_steps = inputs.shape[1]
        
        ### warming up internal states from zero state using first time step from `inputs`
#         x = inputs#[:, 0, :]
#         for i in range(self.num_layers-1):
#             x, *states = self.rnn_layers_list[i](x)
#             states_list.append(states)
#         x, *states = self.rnn_layers_list[-1](x)
#         states_list.append(states)

        if inputs.shape[0] == None:
            dim0 = 1
        else:
            dim0 = inputs.shape[0]
        for i in range(self.num_layers):
            states_list.append([tf.zeros(shape=(dim0, self.num_hidden_units)) for j in range(2)])

        for j in range(min_warmup_steps):
            # print('\n---j:{}---'.format(j))
            x = inputs[:, 0, :]
            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    training=training
                )
                states_list[i] = states[0]

        ### warming up internal states and making first prediction using all of `inputs`
        for j in range(num_time_steps):
            # print('\n---j:{}---'.format(j))
            x = inputs[:, j, :]
            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    training=training
                )
                states_list[i] = states[0]
                
        x = self.dense(x)
        return x, states_list
        
    def call(self, x, output_time=None, training=None):

        if output_time == None:
            output_steps = self.out_steps
        else:
            output_steps = output_time // self.dt_rnn
        
        predictions_list = []

        # Initialize the LSTM state.
        # prediction, states_list = self.warmup(x, training)
        prediction, states_list = self.warmup_v4(x, training)

        # Insert the first prediction.
        predictions_list.append(prediction)

        # Run the rest of the prediction steps.
        for j in range(1, output_steps):
            # print('\n---j:{}---'.format(j))
            # Use the last prediction as input.
            x = prediction

            # Execute one lstm step.
            for i in range(self.num_layers):
                # print('i:{}'.format(i))
                # print([[type(elem) for elem in states_list[i]] for i in range(len(states_list))])
                # print(x.shape)
                x, *states = self.rnn_cells_list[i](
                    x,
                    states=states_list[i],
                    training=training
                )
                states_list[i] = states[0]

            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions_list.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions_list = tf.stack(predictions_list)
        # predictions.shape => (batch, time, features)
        predictions_list = tf.transpose(predictions_list, [1, 0, 2])

        return predictions_list

################################################################################