################################################################################
# Regular residual GRU with skip connections, with uniform/normal noise added  #
# to every input. [STATEFUL]                                                   #
#------------------------------------------------------------------------------#
#                        Basic Network Architecture                            #
#------------------------------------------------------------------------------#
#                                                                     z1+d1    #
#                                                  z1+d1                +d2    #
#         __   z1   __   d1     z1+d1  __   d2       +d2  __   d3       +d3    #
# u----->|__|----->|__|----->[+]----->|__|----->[+]----->|__|----->[+]----->   #
#           \________________/ \________________/ \________________/           #
#                                                                              #
# Note here that you can only specify the number of layers and the number of   #
# units in a layer, not the number of units in each layer individually. Also, a#
# single layer network is the same as a regular GRU.                           #
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
        return tf.tile(self.learnable_variable, [batch_size, 1])


class uniform_noise(layers.Layer):
    def __init__(self, mean=0.0, stddev=1e-4, **kwargs):
        super(uniform_noise, self).__init__()
        self.stddev = stddev
        self.mean = mean

    def call(self, x):
        x = x + tf.random.uniform(shape=tf.shape(x), minval=self.mean-1.732051*self.stddev, maxval=self.mean+1.732051*self.stddev)
        return x

class single_weights(layers.Layer):
    def __init__(self, w_regularizer=None, **kwargs):
        super(single_weights, self).__init__()
        self._weights_regularizer = w_regularizer
        
    def build(self, input_shape):
        self.individual_weights = self.add_weight(
            name='individual_weights',
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.33),
            regularizer=self._weights_regularizer,
            trainable=True
        )

    def call(self, x):
        return x * self.individual_weights

class GRUCell_zoneout(layers.GRUCell):
    def __init__(self, **kwargs):
        self.zoneout_rate = kwargs.pop('zoneout_rate', 0.0)
        super(GRUCell_zoneout, self).__init__(**kwargs)

    def call(self, inputs, states, training=None):
        h_tm1 = (
            states[0] if tf.nest.is_nested(states) else states
        )
        output, candidate_states = super(GRUCell_zoneout, self).call(
            inputs,
            states,
            training
        )
        if 0.0 < self.zoneout_rate < 1.0:
            h_tcandidate = (
                candidate_states[0] if tf.nest.is_nested(candidate_states) else candidate_states
            )
            if training == True:
                zoneout_mask = self._random_generator.dropout(tf.ones_like(h_tm1), self.zoneout_rate) # this is the matrix equivalent of 1-self.zoneout_rate
                h_t = h_tm1 + zoneout_mask * (h_tcandidate - h_tm1)
            else:
                h_t = h_tcandidate + self.zoneout_rate * (h_tm1 - h_tcandidate)
            h_t = [h_t] if tf.nest.is_nested(states) else h_t
        else:
            h_t = candidate_states

        return output, h_t

class RNN_GRU(Model):
    """
    Single-step GRU network that advances (in time) the latent space representation,
    and has trainable initial states for the cell and memory states.
    """
    def __init__(
            self, data_dim=None,
            dt_rnn=None,
            lambda_reg=0.0,
            reg_name='L2',
            rnn_layers_units=[3, 3, 3],
            dense_layer_act_func='linear',
            load_file=None,
            stddev=0.0,
            mean=0.0,
            noise_type='uniform',
            dense_dim=None,
            use_learnable_state=True,
            stateful=False,
            zoneout_rate=0.0,
            rnncell_dropout_rate=0.0,
            denselayer_dropout_rate=0.0,
            batch_size=1,
            use_weights_post_dense=False,
            use_trainable_weights_with_reslayers=False,
            **kwargs):
        
        super(RNN_GRU, self).__init__()

        self.load_file = load_file
        self.data_dim = data_dim
        self.dt_rnn = dt_rnn
        self.lambda_reg = lambda_reg
        self.reg_name = reg_name
        self.rnn_layers_units = rnn_layers_units
        self.dense_layer_act_func = dense_layer_act_func
        self.mean = mean
        self.stddev = stddev
        self.noise_type = noise_type
        self.dense_dim = dense_dim
        self.use_learnable_state = use_learnable_state
        self.batch_size = batch_size
        self.stateful = stateful # ideally do not use `stateful`=True and `use_learnable_state`=True at the same time.
        self.zoneout_rate = zoneout_rate
        self.rnncell_dropout_rate = rnncell_dropout_rate
        self.denselayer_dropout_rate = denselayer_dropout_rate
        self.use_weights_post_dense = use_weights_post_dense
        self.use_trainable_weights_with_reslayers = use_trainable_weights_with_reslayers
        if self.load_file is not None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(''.join(lines))
            if 'data_dim' in load_dict.keys():
                self.data_dim = load_dict['data_dim']
            if 'dt_rnn' in load_dict.keys():
                self.dt_rnn = load_dict['dt_rnn']
            if 'lambda_reg' in load_dict.keys():
                self.lambda_reg = load_dict['lambda_reg']
            if 'reg_name' in load_dict.keys():
                self.reg_name = load_dict['reg_name']
            if 'rnn_layers_units' in load_dict.keys():
                self.rnn_layers_units = load_dict['rnn_layers_units']
            if 'dense_layer_act_func' in load_dict.keys():
                self.dense_layer_act_func = load_dict['dense_layer_act_func']
            if 'mean' in load_dict.keys():
                self.mean = load_dict['mean']
            if 'stddev' in load_dict.keys():
                self.stddev = load_dict['stddev']
            if 'noise_type' in load_dict.keys():
                self.noise_type = load_dict['noise_type']
            if 'dense_dim' in load_dict.keys():
                self.dense_dim = load_dict['dense_dim']
            if 'use_learnable_state' in load_dict.keys():
                self.use_learnable_state = load_dict['use_learnable_state']
            if 'zoneout_rate' in load_dict.keys():
                self.zoneout_rate = load_dict['zoneout_rate']
            if 'stateful' in load_dict.keys():
                self.stateful = load_dict['stateful']
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'denselayer_dropout_rate' in load_dict.keys():
                self.denselayer_dropout_rate = load_dict['denselayer_dropout_rate']
            if 'rnncell_dropout_rate' in load_dict.keys():
                self.rnncell_dropout_rate = load_dict['rnncell_dropout_rate']
            if 'use_trainable_weights_with_reslayers' in load_dict.keys():
                self.use_trainable_weights_with_reslayers = load_dict['use_trainable_weights_with_reslayers']
        self.num_rnn_layers = len(self.rnn_layers_units)

        self.zoneout_rate = min(1.0, max(0.0, self.zoneout_rate))
        self.denselayer_dropout_rate = min(1.0, max(0.0, self.denselayer_dropout_rate))
        self.rnncell_dropout_rate = min(1.0, max(0.0, self.rnncell_dropout_rate))

        if isinstance(self.dense_layer_act_func, list) == False:
            self.dense_layer_act_func = [self.dense_layer_act_func]

        if self.dense_dim is None:
            self.dense_dim = [self.data_dim]*len(self.dense_layer_act_func)


        self.noise_gen = eval('tf.random.'+self.noise_type)
        if self.noise_type == 'uniform':
            self.noise_kwargs = {
                'minval':self.mean-1.732051*self.stddev,
                'maxval':self.mean+1.732051*self.stddev
            }
        elif self.noise_type == 'normal':
            self.noise_kwargs = {
                'mean':self.mean,
                'stddev':self.stddev
            }
        self.num_skip_connections = self.num_rnn_layers - 1
        
        # if self.num_skip_connections > 0:
        #    self.final_scalar_multipliers = scalar_multipliers(self.num_skip_connections)

        ### the GRU network
        self.hidden_states_list = []
        reg = lambda x:None
        use_reg = False
        if self.reg_name != None and self.lambda_reg != None and self.lambda_reg != 0:
            reg = eval('tf.keras.regularizers.'+self.reg_name)
            use_reg = True
        self.rnn_list = [
            layers.RNN(
                cell=GRUCell_zoneout(
                    units=units,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg),
                    zoneout_rate=self.zoneout_rate,
                    dropout=self.rnncell_dropout_rate,
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            ) for units in self.rnn_layers_units
        ]
        
        if self.use_learnable_state == True:
            self.hidden_states_list = [
                learnable_state(
                    hidden_shape=units,
                    b_regularizer=reg(self.lambda_reg)
                ) for units in self.rnn_layers_units
            ]

        self.dense = [
            layers.Dense(
                self.dense_dim[i],
                activation=self.dense_layer_act_func[i],
                # kernel_initializer=tf.initializers.zeros(),
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            ) for i in range(len(self.dense_layer_act_func))
        ]
        
        self.dense_dropout = []
        if self.denselayer_dropout_rate > 0.0:
            self.dense_dropout = [
                layers.Dropout(
                    self.denselayer_dropout_rate
                ) for i in range(len(self.dense))
            ]

        if self.use_weights_post_dense == True:
            self.dense.append(
                single_weights(w_regularizer=reg(self.lambda_reg))
            )
        

        self.init_state = [None]*len(self.rnn_layers_units)
        if self.use_learnable_state == True:
            x_in = tf.ones(
                shape=(self.batch_size, 1, self.data_dim),
                dtype='float32'
            )
            for i in range(len(self.rnn_layers_units)):
                self.init_state[i] = [self.hidden_states_list[i](x_in)]
                x_in = tf.ones(
                    shape=(self.batch_size, 1, self.rnn_layers_units[i]),
                    dtype='float32'
                )

        if self.use_trainable_weights_with_reslayers == True:
            self.reslayer_factor = [tf.Variable(
                initial_value=1.0,
                trainable=True,
                name="reslayer_weight_{}".format(i)
            ) for i in range(self.num_skip_connections)]
        else:
            self.reslayer_factor = [1.0]*self.num_skip_connections

        ### initializing weights
        # temp = tf.ones(shape=[self.batch_size, 1, self.data_dim])
        # temp = self.predict(temp)

        return

    # @tf.function
    def call(self, inputs, training=None):

        # inputs shape : (None, time_steps, data_dim)
        out_steps = inputs.shape[1]

        ### Passing input through the GRU layers
        # first layer
        x = inputs
        if training == True:
            x = x + self.noise_gen(shape=tf.shape(x), **self.noise_kwargs)
        # init_state_j = self.init_state[0]
        x = self.rnn_list[0](
            x,
            # initial_state=init_state_j,
            training=training,
        )
        # if self.stateful == True:
        #     self.init_state[0] = None # so that future batches don't use the initial state

        # remaining layers
        for j in range(1, self.num_rnn_layers):
            # init_state_j = self.init_state[j]
            prediction = self.rnn_list[j](
                x,
                # initial_state=init_state_j,
                training=training,
            )
            x = x + self.reslayer_factor[j-1]*prediction
            # if self.stateful == True:
            #    self.init_state[j] = None # so that future batches don't use the initial state
        
        # doing the final weighted sum of the intermediate predictions
        output = x
        # running through the final dense layers
        for j in range(len(self.dense_layer_act_func)):
            if self.denselayer_dropout_rate > 0.0:
                output = self.dense_dropout[j](output, training=training)
            output = layers.TimeDistributed(self.dense[j])(output, training=training)
    
        if self.use_weights_post_dense == True:
            output = layers.TimeDistributed(self.dense[-1])(output, training=training)


        return output


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
            'dt_rnn':self.dt_rnn,
            'lambda_reg':self.lambda_reg,
            'reg_name':self.reg_name,
            'rnn_layers_units':list(self.rnn_layers_units),
            'dense_layer_act_func':list(self.dense_layer_act_func),
            'load_file':self.load_file,
            'mean':self.mean,
            'stddev':self.stddev,
            'noise_type':self.noise_type,
            'module':self.__module__,
            'dense_dim':list(self.dense_dim),
            'use_learnable_state':self.use_learnable_state,
            'stateful':self.stateful,
            'zoneout_rate':self.zoneout_rate,
            'rnncell_dropout_rate':self.rnncell_dropout_rate,
            'denselayer_dropout_rate':self.denselayer_dropout_rate,
            'use_weights_post_dense':self.use_weights_post_dense,
            'use_trainable_weights_with_reslayers':self.use_trainable_weights_with_reslayers,
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

        self.load_weights(file_name)
        return


################################################################################
