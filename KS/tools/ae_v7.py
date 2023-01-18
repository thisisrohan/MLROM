################################################################################
# Deep denoising 1D-CNN autoencoder.                                           #
# Uses a single weights layer at the end.                                      #
################################################################################

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2
from keras.engine import data_adapter

################################################################################

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


class interpolate1D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.size = kwargs.pop('size', 2)
        
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=None):
        output = []
        
        initial_added_elems = self.size//2 - 1 
        for j in range(initial_added_elems):
            output.append(x[:, 0, :])
        
        for i in range(x.shape[-2]-1):
            output.append(x[:, i, :])
            for j in range(1, self.size):
                output.append(
                    ( (self.size-j)*x[:, i, :] + j*x[:, i+1, :] ) / self.size 
                )
                
        for j in range(self.size - initial_added_elems):
            output.append(x[:, -1, :])
            
        output = tf.transpose(output, [1, 0, 2])       

        return output


class Autoencoder(Model):
    """
    Autoencoder network that finds the latent space representation
    """
    def __init__(
            self, data_dim=6,
            enc_layers=[16,12,8,8,4,4,2], # number of filters
            dec_layers=[2,4,4,8,8,12,16], # number of filters
            latent_space_dim=2,
            lambda_reg=0.0,
            reg_name='L2',
            enc_layer_act_func='elu',
            enc_final_layer_act_func='tanh',
            dec_layer_act_func='elu',
            dec_final_layer_act_func='linear',
            load_file=None,
            stddev=0.0,
            contractive_lmda=1.0,
            use_weights_post_dense=False,
            dropout_rate=0.0,
            enc_kernel_sizes=None,
            dec_kernel_sizes=None,
            **kwargs,
        ):
        
        super(Autoencoder, self).__init__()

        self.stddev = stddev
        self.contractive_lmda = contractive_lmda
        self.load_file = load_file
        self.data_dim = data_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.latent_space_dim = latent_space_dim
        self.lambda_reg = lambda_reg
        self.reg_name = reg_name
        self.enc_layer_act_func = enc_layer_act_func
        self.enc_final_layer_act_func = enc_final_layer_act_func
        self.dec_layer_act_func = dec_layer_act_func
        self.dec_final_layer_act_func = dec_final_layer_act_func
        self.use_weights_post_dense = use_weights_post_dense
        self.dropout_rate = dropout_rate
        self.enc_kernel_sizes = enc_kernel_sizes
        self.dec_kernel_sizes = dec_kernel_sizes
        if self.load_file is not None:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            # load_dict = ''
            # for line in lines:
            #     load_dict += line
            # load_dict = eval(load_dict)
            if 'data_dim' in load_dict.keys():
                self.data_dim=load_dict['data_dim']
            if 'enc_layers' in load_dict.keys():
                self.enc_layers=load_dict['enc_layers']
            if 'dec_layers' in load_dict.keys():
                self.dec_layers=load_dict['dec_layers']
            if 'latent_space_dim' in load_dict.keys():
                self.latent_space_dim=load_dict['latent_space_dim']
            if 'lambda_reg' in load_dict.keys():
                self.lambda_reg=load_dict['lambda_reg']
            if 'reg_name' in load_dict.keys():
                self.reg_name=load_dict['reg_name']
            if 'enc_layer_act_func' in load_dict.keys():
                self.enc_layer_act_func=load_dict['enc_layer_act_func']
            if 'enc_final_layer_act_func' in load_dict.keys():
                self.enc_final_layer_act_func=load_dict['enc_final_layer_act_func']
            if 'dec_layer_act_func' in load_dict.keys():
                self.dec_layer_act_func=load_dict['dec_layer_act_func']
            if 'dec_final_layer_act_func' in load_dict.keys():
                self.dec_final_layer_act_func=load_dict['dec_final_layer_act_func']
            if 'stddev' in load_dict.keys():
                self.stddev = load_dict['stddev']
            if 'contractive_lmda' in load_dict.keys():
                self.contractive_lmda = load_dict['contractive_lmda']
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'dropout_rate' in load_dict.keys():
                self.dropout_rate = load_dict['dropout_rate']
            if 'enc_kernel_sizes' in load_dict.keys():
                self.enc_kernel_sizes = load_dict['enc_kernel_sizes']
            if 'dec_kernel_sizes' in load_dict.keys():
                self.dec_kernel_sizes = load_dict['dec_kernel_sizes']

        self.dropout_rate = min(1.0, max(0.0, self.dropout_rate))

        ### input vector
        input_shape = (self.data_dim,)
        input_vec = Input(shape=input_shape)

        reg = lambda x:None
        use_reg = False
        if self.reg_name != None and self.lambda_reg != None and self.lambda_reg != 0:
            reg = eval('tf.keras.regularizers.'+self.reg_name)
            use_reg = True

        ### the encoder network
        self.noise_layer = layers.GaussianNoise(stddev=self.stddev)

        if self.enc_layer_act_func == 'modified_relu':
            a = 1 - np.exp(-1)
            enc_layer_activation = lambda x:tf.keras.activations.relu(x+a)-a
        else:
            enc_layer_activation = self.enc_layer_act_func

        encoder_layers_list = [
            layers.Conv1D(
                filters=self.enc_layers[i],
                kernel_size=self.enc_kernel_sizes[i],
                padding='same',
                use_bias=True,
                activation=enc_layer_activation,
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            ) for i in range(len(self.enc_layers))
        ]
        encoder_layers_list.append(
            layers.Dense(
                self.latent_space_dim,
                activation=self.enc_final_layer_act_func,
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            )
        )


        x = self.noise_layer(input_vec)
        x = tf.expand_dims(x, axis=-1)
        x = encoder_layers_list[0](x)
        x = layers.MaxPool1D(
            pool_size=2,
            strides=2,
            padding='valid',
        )(x)
        for i in range(1, len(encoder_layers_list)-1):
            # if self.dropout_rate > 0.0:
            #     x = layers.Dropout(self.dropout_rate)(x)
            x = encoder_layers_list[i](x)
            x = layers.MaxPool1D(
                pool_size=2,
                strides=2,
                padding='valid',
            )(x)
        encoded_vec_PreFlattenAndDense_shape = list(x.shape[-2:])
        x = layers.Reshape([x.shape[-2]*x.shape[-1]], input_shape=x.shape[-2:])(x)
        x = encoder_layers_list[-1](x)
        encoded_vec = x

        self.encoder_layers_list = encoder_layers_list
        self.encoder_net = Model(inputs=input_vec, outputs=encoded_vec)

        ### the decoder network
        if self.dec_layer_act_func == 'modified_relu':
            dec_layer_activation = lambda x:tf.keras.activations.relu(x+a)-a
        else:
            dec_layer_activation = self.dec_layer_act_func

        decoder_layers_list = [
            layers.Dense(
                encoded_vec_PreFlattenAndDense_shape[-2]*encoded_vec_PreFlattenAndDense_shape[-1],
                activation=dec_layer_activation,
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            )
        ]
        decoder_layers_list.extend([
            layers.Conv1D(
                filters=self.dec_layers[i],
                kernel_size=self.dec_kernel_sizes[i],
                padding='same',
                use_bias=True,
                activation=dec_layer_activation,
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            ) for i in range(len(self.dec_layers))
        ])
        decoder_layers_list.append(
            layers.Dense(
                data_dim,
                activation=self.dec_final_layer_act_func,
                kernel_regularizer=reg(self.lambda_reg),
                bias_regularizer=reg(self.lambda_reg)
            )
        )

        
        encoded_input_vec = Input(shape=encoded_vec.shape[-1:])
        y = decoder_layers_list[0](encoded_input_vec)
        y = layers.Reshape(
            encoded_vec_PreFlattenAndDense_shape[-2:],
            input_shape=y.shape[1:]
        )(y)

        y = interpolate1D(size=2)(y)        
        y = decoder_layers_list[1](y)
        for i in range(2, len(decoder_layers_list)-1):
            # if self.dropout_rate > 0.0:
            #     y = layers.Dropout(self.dropout_rate)(y)
            y = interpolate1D(size=2)(y)
            y = decoder_layers_list[i](y)
        
        y = layers.Reshape([y.shape[-2]*y.shape[-1]], input_shape=y.shape[-2:])(y)
        y = decoder_layers_list[-1](y)
        
        if self.use_weights_post_dense == True:
            decoder_layers_list.append(
                single_weights(w_regularizer=reg(self.lambda_reg))
            )
            y = decoder_layers_list[-1](y)

        decoded_vec = y

        self.decoder_layers_list = decoder_layers_list
        self.decoder_net = Model(inputs=encoded_input_vec, outputs=decoded_vec)

        ### all combined, the autoencoder
        x = self.encoder_net(input_vec)
        x = self.decoder_net(x)
        self.ae_net = Model(inputs=input_vec, outputs=x)

        temp = tf.ones(shape=(1, self.data_dim,))
        temp = self(temp)

        return


    def train_step(self, data):

        # x, y = data
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape1:
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(x)
                tape2.watch(self.trainable_variables)
                encoded = self.encoder_net(x, training=True)
            ls_jacobian = tape2.batch_jacobian(encoded, x)
            # ls_norm = tf.norm(encoded, axis=-1)
            
            ls_jacobian_norm = tf.reduce_mean(tf.norm(
                ls_jacobian,
                ord='fro',
                axis=[-2, -1]
            ))# / ls_norm)
            decoded = self.decoder_net(encoded, training=True)
            loss = self.compiled_loss(
                y,
                decoded,
                sample_weight,
                regularization_losses=self.losses
            ) + self.contractive_lmda*ls_jacobian_norm

        self._validate_target_and_loss(decoded, loss)

        trainable_vars = self.trainable_variables
        gradients = tape1.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return self.compute_metrics(x, y, decoded, sample_weight, ls_jacobian_norm=ls_jacobian_norm)
        
    def compute_metrics(self, x, y, decoded, sample_weight, ls_jacobian_norm=-1.0):
        metric_results = super().compute_metrics(x, y, decoded, sample_weight)
        metric_results['ls_jacobian_norm'] = ls_jacobian_norm
        return metric_results
    

    @tf.function
    def call(self, x):

        x = self.ae_net(x)

        return x

    def save_model_weights(self, file_name, H5=True):

        # file_name = file_dir + '/' + file_name
        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        if H5 == True:
            file_name += '.h5'
        self.save_weights(file_name)
        return

    def save_class_dict(self, file_name):
        ### saving class attributes
        class_dict = {
            'data_dim':self.data_dim,
            'enc_layers':list(self.enc_layers),
            'dec_layers':list(self.dec_layers),
            'latent_space_dim':self.latent_space_dim,
            'lambda_reg':self.lambda_reg,
            'reg_name':self.reg_name,
            'enc_layer_act_func':self.enc_layer_act_func,
            'enc_final_layer_act_func':self.enc_final_layer_act_func,
            'dec_layer_act_func':self.dec_layer_act_func,
            'dec_final_layer_act_func':self.dec_final_layer_act_func,
            'load_file':self.load_file,
            'stddev':self.stddev,
            'contractive_lmda':self.contractive_lmda,
            'module':self.__module__,
            'use_weights_post_dense':self.use_weights_post_dense,
            'dropout_rate':self.dropout_rate,
            'enc_kernel_sizes':self.enc_kernel_sizes,
            'dec_kernel_sizes':self.dec_kernel_sizes,
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
        self.save_model_weights(file_name+'_ae_weights', H5=H5)

        return

    def load_weights_from_file(self, file_name):

        # temp = tf.ones(shape=(1, self.data_dim,))
        # temp = self.call(temp)

        self.load_weights(file_name)
        return

    def set_enc_dec_weights(self, enc_wt_file_name, dec_wt_file_name):

        self.encoder_net.load_weights(enc_wt_file_name)
        self.decoder_net.load_weights(dec_wt_file_name)
        return


################################################################################
