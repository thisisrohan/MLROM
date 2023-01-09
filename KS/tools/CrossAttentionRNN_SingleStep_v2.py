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
from keras.engine import data_adapter

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

        
    
class CrossAttentionSimpleRNNCell(layers.AbstractRNNCell):
    def __init__(
            self,
            units,
            activation='tanh',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            attention_kernel_initializer='orthogonal',
            recurrent_attention_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            attention_kernel_regularizer=None,
            recurrent_attention_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            attention_kernel_constraint=None,
            recurrent_attention_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            **kwargs
        ):
        
        super(CrossAttentionSimpleRNNCell, self).__init__(**kwargs)

        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.attention_kernel_initializer = tf.keras.initializers.get(attention_kernel_initializer)
        self.recurrent_attention_initializer = tf.keras.initializers.get(recurrent_attention_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.attention_kernel_regularizer = tf.keras.regularizers.get(attention_kernel_regularizer)
        self.recurrent_attention_regularizer = tf.keras.regularizers.get(recurrent_attention_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.attention_kernel_constraint = tf.keras.constraints.get(attention_kernel_constraint)
        self.recurrent_attention_constraint = tf.keras.constraints.get(recurrent_attention_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # self.dropout = min(1.0, max(0.0, dropout))
        # self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        # self.state_size = self.units
        # self.output_size = self.units

    @property
    def state_size(self):
        return self.units
        
    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weight(
            shape=(self.units, input_shape[-1]),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            # caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            # caching_device=default_caching_device,
        )
        self.attention_kernel = self.add_weight(
            shape=(self.units, input_shape[-1]),
            name="attention_kernel",
            initializer=self.attention_kernel_initializer,
            regularizer=self.attention_kernel_regularizer,
            constraint=self.attention_kernel_constraint,
            # caching_device=default_caching_device,
        )
        self.recurrent_attention_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="recurrent_attention_kernel",
            initializer=self.recurrent_attention_initializer,
            regularizer=self.recurrent_attention_regularizer,
            constraint=self.recurrent_attention_constraint,
            # caching_device=default_caching_device,
        )
        self.gamma_prime_attention = tf.Variable(
            initial_value=-1.0,
            name="gamma_attention",
        )
        self.gamma_prime_recurrent_attention = tf.Variable(
            initial_value=-1.0,
            name="gamma_recurrent_attention",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                # caching_device=default_caching_device,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states
        # dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        # rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        #     prev_output, training
        # )

        # if dp_mask is not None:
        #     h = backend.dot(inputs * dp_mask, self.kernel)
        # else:
        #     h = backend.dot(inputs, self.kernel)
        
        gamma_attention = tf.math.sigmoid(self.gamma_prime_attention)
        gamma_recurrent_attention = tf.math.sigmoid(self.gamma_prime_recurrent_attention)
        
        inputs = tf.expand_dims(inputs, axis=-1)
        prev_output = tf.expand_dims(prev_output, axis=-1)
        
        attention_mat = tf.linalg.matmul(prev_output, inputs, transpose_b=True)
        attention_mat = attention_mat * self.attention_kernel
        attention_mat = tf.math.exp(attention_mat)
        attention_mat = attention_mat / tf.math.reduce_sum(attention_mat, axis=-1, keepdims=True)
        attention_mat = attention_mat * self.kernel

        # h = (1 - gamma_attention) * tf.keras.backend.dot(inputs, self.kernel)
        # h = (1 - gamma_attention) * tf.squeeze(tf.linalg.matmul(self.kernel, inputs), axis=-1)
        h = tf.squeeze(tf.linalg.matmul(self.kernel, inputs), axis=-1)
        h = h + gamma_attention * tf.squeeze(tf.linalg.matmul(attention_mat, inputs), axis=-1)
        if self.bias is not None:
            h = tf.keras.backend.bias_add(h, self.bias)

        # if rec_dp_mask is not None:
        #    prev_output = prev_output * rec_dp_mask

        recurrent_attention_mat = tf.linalg.matmul(prev_output, prev_output, transpose_b=True)
        recurrent_attention_mat = recurrent_attention_mat * self.recurrent_attention_kernel
        recurrent_attention_mat = tf.math.exp(recurrent_attention_mat)
        recurrent_attention_mat = recurrent_attention_mat / tf.math.reduce_sum(recurrent_attention_mat, axis=-1, keepdims=True)
        recurrent_attention_mat = recurrent_attention_mat * self.recurrent_kernel
        
        # output = h + (1 - gamma_recurrent_attention) * tf.keras.backend.dot(prev_output, self.recurrent_kernel)
        # output = h + (1 - gamma_recurrent_attention) * tf.squeeze(tf.linalg.matmul(self.recurrent_kernel, prev_output), axis=-1)
        output = h + tf.squeeze(tf.linalg.matmul(self.recurrent_kernel, prev_output), axis=-1)
        output = output + gamma_recurrent_attention * tf.squeeze(tf.linalg.matmul(recurrent_attention_mat, prev_output), axis=-1)
        if self.activation is not None:
            output = self.activation(output)

        new_state = [output] if tf.nest.is_nested(states) else output
        return output, new_state
        # return output, (output)


class RNN_CrossAttention(Model):
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
            # zoneout_rate=0.0,
            # rnncell_dropout_rate=0.0,
            denselayer_dropout_rate=0.0,
            batch_size=1,
            scalar_weights=None,
            use_weights_post_dense=False,
            **kwargs):
        
        super(RNN_CrossAttention, self).__init__()

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
        # self.zoneout_rate = zoneout_rate
        # self.rnncell_dropout_rate = rnncell_dropout_rate
        self.denselayer_dropout_rate = denselayer_dropout_rate
        self.scalar_weights = scalar_weights
        self.use_weights_post_dense = use_weights_post_dense
        if self.load_file is not None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
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
            # if 'zoneout_rate' in load_dict.keys():
            #     self.zoneout_rate = load_dict['zoneout_rate']
            if 'stateful' in load_dict.keys():
                self.stateful = load_dict['stateful']
            if 'scalar_weights' in load_dict.keys():
                self.scalar_weights = load_dict['scalar_weights']
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'denselayer_dropout_rate' in load_dict.keys():
                self.denselayer_dropout_rate = load_dict['denselayer_dropout_rate']
            # if 'rnncell_dropout_rate' in load_dict.keys():
            #     self.rnncell_dropout_rate = load_dict['rnncell_dropout_rate']
        self.num_rnn_layers = len(self.rnn_layers_units)

        # self.zoneout_rate = min(1.0, max(0.0, self.zoneout_rate))
        self.denselayer_dropout_rate = min(1.0, max(0.0, self.denselayer_dropout_rate))
        # self.rnncell_dropout_rate = min(1.0, max(0.0, self.rnncell_dropout_rate))

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
                cell=CrossAttentionSimpleRNNCell(
                    units=self.rnn_layers_units[0],
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg),
                    recurrent_regularizer=reg(self.lambda_reg),
                    attention_kernel_regularizer=reg(self.lambda_reg),
                    recurrent_attention_regularizer=reg(self.lambda_reg),
                    # zoneout_rate=self.zoneout_rate,
                    # dropout=self.rnncell_dropout_rate,
                ),
                return_sequences=True,
                stateful=self.stateful,
                batch_size=self.batch_size if self.stateful else None,
            )
        ]
        if self.num_skip_connections > 0:
            self.RK_RNNCell = CrossAttentionSimpleRNNCell(
                units=self.rnn_layers_units[1],
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg),
                recurrent_regularizer=reg(self.lambda_reg),
                attention_kernel_regularizer=reg(self.lambda_reg),
                recurrent_attention_regularizer=reg(self.lambda_reg),
                # zoneout_rate=self.zoneout_rate,
                # dropout=self.rnncell_dropout_rate,
            )
            self.rnn_list.extend([
                layers.RNN(
                    self.RK_RNNCell,
                    return_sequences=True,
                    stateful=self.stateful,
                    batch_size=self.batch_size if self.stateful else None,
                ) for i in range(self.num_skip_connections)
            ])
        
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
                ) for i in range(len(self.dense_layer_act_func))
            ]

        if self.use_weights_post_dense == True:
            self.dense.append(
                single_weights(w_regularizer=reg(self.lambda_reg))
            )

        if self.num_skip_connections > 0:
            if type(self.scalar_weights) == type(None):
                self.scalar_multiplier_pre_list = []
                for i in range(self.num_skip_connections):
                    for j in range(i+1):
                        self.scalar_multiplier_pre_list.append(
                            tf.Variable(
                                initial_value=1.0,
                                dtype='float32',
                                trainable=True,
                                name='a_{}{}'.format(j+1, i+1) # this naming convention follows the one in the diagram at the top of this script
                            )
                        )
                # self.scalar_multiplier_pre_list = [
                #     tf.Variable(
                #         initial_value=1.0,
                #         dtype='float32',
                #         trainable=True,
                #     ) for i in range(int(0.5*self.num_skip_connections*(self.num_skip_connections+1)))
                # ]
            else:
                self.scalar_weights = np.array(self.scalar_weights, dtype='float32')

        self.init_state = [None]*len(self.rnn_layers_units)
        # if self.stateful == True:
        #     for j in range(self.num_skip_connections):
        #         self.init_state[j+1] = tf.zeros(shape=(self.batch_size, self.rnn_layers_units[1]), dtype='float32')
        # elif
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
            x = x + prediction
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


    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)


    def compute_metrics(self, x, y, y_pred, sample_weight):
        metric_results = super().compute_metrics(x, y, y_pred, sample_weight)
        metric_results['gamma_prime_attention'] = self.rnn_list[0].cell.gamma_prime_attention
        metric_results['gamma_prime_recurrent_attention'] = self.rnn_list[0].cell.gamma_prime_recurrent_attention   
        return metric_results   
    

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
            # 'zoneout_rate':self.zoneout_rate,
            # 'rnncell_dropout_rate':self.rnncell_dropout_rate,
            'denselayer_dropout_rate':self.denselayer_dropout_rate,
            'use_weights_post_dense':self.use_weights_post_dense,
            # 'eta':self.eta,
            # 'ns':self.ns,
            # 'cell_output_activation':self.cell_output_activation,
            # 'cell_state_activation':self.cell_state_activation,
            # 'cell_nonlinear_expansion_activation':self.cell_nonlinear_expansion_activation,
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
        self.save_model_weights(file_name+'_crossattentioncell_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name)
        return


################################################################################
