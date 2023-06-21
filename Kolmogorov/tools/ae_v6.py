################################################################################
# Deep denoising multi-scale 2D-CNN autoencoder.                               #
# Uses a single weights layer at the end.                                      #
# Can use the attention-module as well                                         #
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
        super().__init__()
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
    

class periodic_padding(layers.Layer):
    # only operates on square images
    def __init__(self, elp, erp, **kwargs):
        super().__init__()
        self.elp = elp
        self.erp = erp
        
    def build(self, input_shape):
        super().build(input_shape)
        input_xlen = input_shape[-1]
        input_ylen = input_shape[-2] # must be the same as `input_xlen`
        M_mat = np.zeros(shape=(input_ylen+self.elp+self.erp, input_ylen), dtype=np.float32)
        for i in range(1, self.elp + 1):
            M_mat[self.elp - i, (input_ylen - i) % input_ylen] = 1.
        for i in range(self.elp, self.elp + input_ylen):
            M_mat[i, i-self.elp] = 1.
        for i in range(self.elp + input_ylen, M_mat.shape[0]):
            M_mat[i, (i - (self.elp + input_ylen)) % input_ylen] = 1.
        # B_mat = np.empty(shape=(A_mat.shape[1], A_mat.shape[0]), dtype=A_mat.dtype)
        N_mat = np.transpose(M_mat)
        
        self.M_mat = tf.Variable(M_mat, trainable=False, name='M_mat')
        self.N_mat = tf.Variable(N_mat, trainable=False, name='N_mat')

    def call(self, x, training=None):
        '''
        x has shape --> [batch_size, channels, y_len, x_len]
        '''
        return tf.linalg.matmul(tf.linalg.matmul(self.M_mat, x), self.N_mat)

    def compute_output_shape(self, input_shape):
        # input_shape : [batch_size?, time_len?, in_channels, ylen, xlen]
        output_shape = (input_shape[-3], input_shape[-2]+self.elp+self.erp, input_shape[-1]+self.elp+self.erp)
        ndims = len(input_shape)
        if ndims >= 4:
            output_shape = tuple(input_shape[0:ndims-3]) + output_shape
        return output_shape


class attention_module(layers.Layer):
    def __init__(self, att_dim, v_channel_dim=None, reg=None, lambda_reg=None, **kwargs):
        super().__init__()
        self.att_dim = att_dim
        self.v_channel_dim = v_channel_dim
        if reg == None:
            self.reg = lambda x : None
        self.lambda_reg = lambda_reg
        
    def build(self, input_shape):
        super().build(input_shape)
        
        if self.v_channel_dim == None:
            self.v_channel_dim = input_shape[-3]
        
        self.Wg = layers.Conv2D(
            filters=self.att_dim,
            kernel_size=1,
            padding='valid',
            strides=1,
            data_format='channels_first',
            use_bias=False,
            kernel_regularizer=self.reg(self.lambda_reg),
            bias_regularizer=self.reg(self.lambda_reg),
            name='Wg',
        )
        
        self.Wf = layers.Conv2D(
            filters=self.att_dim,
            kernel_size=1,
            padding='valid',
            strides=1,
            data_format='channels_first',
            use_bias=False,
            kernel_regularizer=self.reg(self.lambda_reg),
            bias_regularizer=self.reg(self.lambda_reg),
            name='Wf',
        )
        self.Wh = layers.Conv2D(
            filters=self.att_dim,
            kernel_size=1,
            padding='valid',
            strides=1,
            data_format='channels_first',
            use_bias=False,
            kernel_regularizer=self.reg(self.lambda_reg),
            bias_regularizer=self.reg(self.lambda_reg),
            name='Wh',
        )
        self.Wv = layers.Conv2D(
            filters=self.v_channel_dim,
            kernel_size=1,
            padding='valid',
            strides=1,
            data_format='channels_first',
            use_bias=False,
            kernel_regularizer=self.reg(self.lambda_reg),
            bias_regularizer=self.reg(self.lambda_reg),
            name='Wv',
        )
        
        self.att_layer = layers.Attention()#use_scale=False, score_mode='dot')
        
        self.lambda_att = tf.Variable(0., trainable=True, name='lambda_att')

    def call(self, x, training=None):
    
        ndims = len(x.shape)
        ylen = x.shape[-2]
        xlen = x.shape[-1]
        if ndims == 3:
            x = tf.expand_dims(x, axis=0)

        # query
        g = tf.reshape(self.Wg(x), (-1, self.att_dim, ylen*xlen))
        g = tf.transpose(g, (0, 2, 1))        
        # key
        f = tf.reshape(self.Wf(x), (-1, self.att_dim, ylen*xlen))
        f = tf.transpose(f, (0, 2, 1))
        # value
        h = tf.reshape(self.Wh(x), (-1, self.att_dim, ylen*xlen))
        h = tf.transpose(h, (0, 2, 1))
        
        v = self.att_layer([g, f, h], training=training)
        v = tf.transpose(v, (0, 2, 1))
        v = tf.reshape(v, (-1, v.shape[1], ylen, xlen))
        v = self.Wv(v)
        
        if ndims == 3:
            x = tf.squeeze(x, axis=0)
            v = tf.squeeze(v, axis=0)
        
        return x + self.lambda_att * v

class Autoencoder(Model):
    """
    Autoencoder network that finds the latent space representation
    """
    def __init__(
            self, data_dim=(50, 50, 2),
            kernel_size=[3],
            enc_filters=[8, 16, 32, 32, 2], # number of filters
            enc_strides=[2, 2, 2, 2, 1],
            enc_attn_placement=[1, 2, 4],
            dec_filters=[32, 32, 16, 8, 2], # number of filters
            dec_strides=[1, 2, 2, 2, 2],
            dec_attn_placement=[1, 2],
            lambda_reg=0.0,
            reg_name='L2',
            enc_layer_act_func='elu',
            enc_final_layer_act_func='tanh',
            dec_layer_act_func='elu',
            dec_final_layer_act_func='linear',
            load_file=None,
            stddev=0.0,
            use_weights_post_dense=False,
            use_batch_norm=True,
            use_periodic_padding=False,
            use_attention_module=False,
            **kwargs,
        ):
        
        super(Autoencoder, self).__init__()

        self.stddev = stddev
        self.kernel_size = kernel_size
        self.load_file = load_file
        self.data_dim = data_dim
        self.enc_filters = enc_filters
        self.enc_strides = enc_strides
        self.enc_attn_placement = enc_attn_placement
        self.dec_filters = dec_filters
        self.dec_strides = dec_strides
        self.dec_attn_placement = dec_attn_placement
        self.lambda_reg = lambda_reg
        self.reg_name = reg_name
        self.enc_layer_act_func = enc_layer_act_func
        self.enc_final_layer_act_func = enc_final_layer_act_func
        self.dec_layer_act_func = dec_layer_act_func
        self.dec_final_layer_act_func = dec_final_layer_act_func
        self.use_weights_post_dense = use_weights_post_dense
        self.use_batch_norm = use_batch_norm
        self.use_periodic_padding = use_periodic_padding
        self.use_attention_module = use_attention_module
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
            if 'enc_filters' in load_dict.keys():
                self.enc_filters=load_dict['enc_filters']
            if 'enc_strides' in load_dict.keys():
                self.enc_strides=load_dict['enc_strides']
            if 'enc_attn_placement' in load_dict.keys():
                self.enc_attn_placement=load_dict['enc_attn_placement']
            if 'dec_filters' in load_dict.keys():
                self.dec_filters=load_dict['dec_filters']
            if 'dec_strides' in load_dict.keys():
                self.dec_strides=load_dict['dec_strides']
            if 'dec_attn_placement' in load_dict.keys():
                self.dec_attn_placement=load_dict['dec_attn_placement']
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
            if 'use_weights_post_dense' in load_dict.keys():
                self.use_weights_post_dense = load_dict['use_weights_post_dense']
            if 'use_batch_norm' in load_dict.keys():
                self.use_batch_norm = load_dict['use_batch_norm']
            if 'kernel_size' in load_dict.keys():
                self.kernel_size = load_dict['kernel_size']
            if 'use_periodic_padding' in load_dict.keys():
                self.use_periodic_padding = load_dict['use_periodic_padding']
            if 'use_attention_module' in load_dict.keys():
                self.use_attention_module = load_dict['use_attention_module']

        print(self.kernel_size, type(self.kernel_size))
        try:
            for elem in self.kernel_size:
                pass
        except:
            self.kernel_size = np.array([self.kernel_size])
        self.kernel_size = np.array(self.kernel_size)
        num_kernel_sizes = self.kernel_size.shape[0]
        print(self.kernel_size, type(self.kernel_size))
        

        if self.use_periodic_padding == True:
            padding = 'valid'
        else:
            padding = 'same'
            
        if self.use_attention_module == False:
            self.enc_attn_placement = []
            self.dec_attn_placement = []

        ### input vector
        input_shape = tuple(self.data_dim)
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

        encoder_layers_list = []
        for ks_i in range(num_kernel_sizes):
            encoder_layers_list.append([])
        if self.use_attention_module == True:
            encoder_attention_modules_list = []
            for ks_i in range(num_kernel_sizes):
                encoder_attention_modules_list.append([])
        prev_shape = input_vec.shape[-2:]
        enc_elem_spatial_dims = []
        for ks_i in range(num_kernel_sizes):
            enc_elem_spatial_dims.append([prev_shape])
        # computing left and right paddings
        # note bottom padding = right padding and top padding = left padding
        elp_lst = []
        erp_lst = []
        for ks_i in range(num_kernel_sizes):
            elp = int(0.5*(self.kernel_size[ks_i]-1))
            erp = self.kernel_size[ks_i] - elp - 1
            elp_lst.append(elp)
            erp_lst.append(erp)
            
        len_enc_filters = len(self.enc_filters)
        for ks_i in range(num_kernel_sizes):
            attn_iter = 0
            for i in range(len_enc_filters):            
                # adding the conv layer
                if self.use_periodic_padding == True:
                    # if i == 0:
                    #     temp_elp = int(0.5*(self.kernel_size[ks_i]-3))
                    #     temp_erp = self.kernel_size[ks_i] - temp_elp - 3
                    #     encoder_layers_list[ks_i].append(periodic_padding(temp_elp, temp_erp))
                    # else:
                    encoder_layers_list[ks_i].append(periodic_padding(elp_lst[ks_i], erp_lst[ks_i]))
                enc_filters_i = self.enc_filters[i]
                enc_stride_i = self.enc_strides[i]
                encoder_layers_list[ks_i].append(layers.Conv2D(
                    filters=enc_filters_i,
                    kernel_size=int(self.kernel_size[ks_i]),
                    # padding='valid',
                    # padding='same',
                    padding=padding,
                    strides=enc_stride_i,
                    data_format='channels_first',
                    use_bias=not self.use_batch_norm,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg),
                ))
                # adding batch norm
                if self.use_batch_norm == True:
                    encoder_layers_list[ks_i].append(layers.BatchNormalization(axis=-3))
                # adding the activation
                encoder_layers_list[ks_i].append(layers.Activation(enc_layer_activation))
                
                if self.use_attention_module == True:
                    if attn_iter < len(self.enc_attn_placement):
                        if i == self.enc_attn_placement[attn_iter]:
                            # adding the attention module
                            encoder_layers_list[ks_i].append(attention_module(att_dim=enc_filters_i))
                            encoder_attention_modules_list[ks_i].append(encoder_layers_list[ks_i][-1])
                            attn_iter += 1
                
                in_spatial_dims = enc_elem_spatial_dims[ks_i][-1]
                in_spatial_dims_y = in_spatial_dims[0] + self.kernel_size[ks_i]-1 # because padding is either periodic or `same`
                in_spatial_dims_x = in_spatial_dims[1] + self.kernel_size[ks_i]-1 # because padding is either periodic or `same`
                enc_elem_spatial_dims[ks_i].append(
                    self.compute_post_conv_shape(
                        (in_spatial_dims_y, in_spatial_dims_x),
                        enc_stride_i, ks_i
                    )
                )
                print(enc_elem_spatial_dims[ks_i])

        x = self.noise_layer(input_vec)
        
        if num_kernel_sizes > 1:
            self.encoded_vec_weights = []
            for ks_i in range(num_kernel_sizes):
                self.encoded_vec_weights.append(tf.Variable(initial_value=1.0, trainable=True, name='encoded_vec_wt_'+str(ks_i)))
        else:
            self.encoded_vec_weights = [1.0]
        
        x_ks = []
        for ks_i in range(num_kernel_sizes):
            x_ks_i = encoder_layers_list[ks_i][0](x)
            for i in range(1, len(encoder_layers_list[ks_i])-1):
                x_ks_i = encoder_layers_list[ks_i][i](x_ks_i)
            x_ks.append(x_ks_i)
        
        encoded_vec = encoder_layers_list[0][-1](x_ks[0]) * self.encoded_vec_weights[0]
        for ks_i in range(1, num_kernel_sizes):
            encoded_vec += encoder_layers_list[ks_i][-1](x_ks[ks_i]) * self.encoded_vec_weights[ks_i]


        self.encoder_layers_list = encoder_layers_list
        if self.use_attention_module == True:
            self.encoder_attention_modules_list = encoder_attention_modules_list
        self.encoder_net = Model(inputs=input_vec, outputs=encoded_vec)

        ### the decoder network
        if self.dec_layer_act_func == 'modified_relu':
            dec_layer_activation = lambda x:tf.keras.activations.relu(x+a)-a
        else:
            dec_layer_activation = self.dec_layer_act_func

        if self.use_attention_module == True:
            decoder_attention_modules_list = []
            for ks_i in range(num_kernel_sizes):
                decoder_attention_modules_list.append([])

        decoder_layers_list = []
        for ks_i in range(num_kernel_sizes):
            decoder_layers_list.append([])
        dec_elem_spatial_dims = []
        for ks_i in range(num_kernel_sizes):
            dec_elem_spatial_dims.append([encoded_vec.shape[-2:]])
        len_dec_filters = len(self.dec_filters)

        for ks_i in range(num_kernel_sizes):
            attn_iter = 0
            for i in range(len_dec_filters):
                print('{} -- kernelsize : {} -- {} / {}'.format(ks_i, self.kernel_size[ks_i], i, len_dec_filters-1))
                # periodic padding
                decoder_layers_list[ks_i].append(periodic_padding(elp_lst[ks_i], erp_lst[ks_i]))
                # transposed CNN
                dec_filters_i = self.dec_filters[i]
                dec_stride_i = self.dec_strides[i]
                decoder_layers_list[ks_i].append(
                    layers.Conv2DTranspose(
                        filters=dec_filters_i,
                        kernel_size=int(self.kernel_size[ks_i]),
                        strides=dec_stride_i,
                        padding='valid',
                        data_format='channels_first',
                        use_bias=not self.use_batch_norm,
                        kernel_regularizer=reg(self.lambda_reg),
                        bias_regularizer=reg(self.lambda_reg),
                    )
                )
                # center-crop
                in_spatial_dims = dec_elem_spatial_dims[ks_i][-1]
                print(in_spatial_dims)
                in_spatial_dims_y = in_spatial_dims[0] + self.kernel_size[ks_i]-1 # because padding is either periodic or `same`
                in_spatial_dims_x = in_spatial_dims[1] + self.kernel_size[ks_i]-1 # because padding is either periodic or `same`
                out_spatial_dims_precrop = self.compute_post_tconv_shape(
                    (in_spatial_dims_y, in_spatial_dims_x),
                    dec_stride_i, ks_i
                )
                if i == len(self.dec_filters)-1:
                    # total_crop = 3*self.kernel_size - 4 # VERY SPECIFIC TO MY PROBLEM WITH INPUT BEING 50x50 DIMENSIONAL and enc_filters/dec_filters having four entries (each)
                    print(out_spatial_dims_precrop)
                    total_crop_y = int(out_spatial_dims_precrop[0] - self.data_dim[-2])
                    ct = int(0.5 * total_crop_y)
                    cb = total_crop_y - ct
                    total_crop_x = int(out_spatial_dims_precrop[1] - self.data_dim[-1])
                    cl = int(0.5 * total_crop_x)
                    cr = total_crop_x - cl
                else:
                    # if self.kernel_size[ks_i] == 3:
                    #     total_crop = int((dec_stride_i+1)*(self.kernel_size[ks_i]-1) - (dec_stride_i-1))
                    # else:
                    total_crop = int((dec_stride_i+1)*(self.kernel_size[ks_i]-1))
                    ct = int(0.5 * total_crop)
                    cl = ct
                    cb = total_crop - ct
                    cr = cb
                decoder_layers_list[ks_i].append(
                    layers.Cropping2D(
                        cropping=(
                            (ct, cb),
                            (cl, cr)
                        ),
                        data_format='channels_first'
                    )
                )
                out_spatial_dims_y = out_spatial_dims_precrop[0] - (ct + cb)
                out_spatial_dims_x = out_spatial_dims_precrop[1] - (cl + cr)
                dec_elem_spatial_dims[ks_i].append(
                    (out_spatial_dims_y, out_spatial_dims_x)
                )
                # batch normalization
                if self.use_batch_norm == True:
                    decoder_layers_list[ks_i].append(layers.BatchNormalization(axis=-3))
                # adding the activation
                decoder_layers_list[ks_i].append(layers.Activation(self.dec_layer_act_func))
                if self.use_attention_module == True:
                    if attn_iter < len(self.dec_attn_placement):
                        if i == self.dec_attn_placement[attn_iter]:
                            # adding the attention module
                            decoder_layers_list[ks_i].append(attention_module(att_dim=dec_filters_i))
                            decoder_attention_modules_list[ks_i].append(decoder_layers_list[ks_i][-1])
                            attn_iter += 1

        encoded_input_vec = Input(shape=encoded_vec.shape[1:])

        if num_kernel_sizes > 1:
            self.decoded_vec_weights = []
            for ks_i in range(num_kernel_sizes):
                self.decoded_vec_weights.append(tf.Variable(initial_value=1.0, trainable=True, name='decoded_vec_wt_'+str(ks_i)))
        else:
            self.decoded_vec_weights = [1.0]
        
        y_ks = []
        for ks_i in range(num_kernel_sizes):
            y_ks_i = decoder_layers_list[ks_i][0](encoded_input_vec)
            for i in range(1, len(decoder_layers_list[ks_i])-1):
                y_ks_i = decoder_layers_list[ks_i][i](y_ks_i)
            y_ks.append(y_ks_i)
        
        decoded_vec = decoder_layers_list[0][-1](y_ks[0]) * self.decoded_vec_weights[0]
        for ks_i in range(1, num_kernel_sizes):
            decoded_vec += decoder_layers_list[ks_i][-1](y_ks[ks_i]) * self.decoded_vec_weights[ks_i]

        
        if self.use_weights_post_dense == True:
            decoder_layers_list[0].append(
                single_weights(w_regularizer=reg(self.lambda_reg))
            )
            y_shape = decoded_vec.shape
            decoded_vec = tf.reshape(decoded_vec, (-1, y_shape[-3]*y_shape[-2]*y_shape[-1]))
            decoded_vec = decoder_layers_list[0][-1](decoded_vec)
            decoded_vec = tf.reshape(decoded_vec, y_shape)

        self.decoder_layers_list = decoder_layers_list
        if self.use_attention_module == True:
            self.decoder_attention_modules_list = decoder_attention_modules_list
        self.decoder_net = Model(inputs=encoded_input_vec, outputs=decoded_vec)

        ### all combined, the autoencoder
        input_vec_z = Input(input_vec.shape[-3:])
        z = self.encoder_net(input_vec_z)
        z = self.decoder_net(z)
        self.ae_net = Model(inputs=input_vec_z, outputs=z)

        temp = tf.ones(shape=(1, ) + tuple(self.data_dim))
        temp = self(temp)
        
        self.num_kernel_sizes = num_kernel_sizes

        return

    def compute_post_conv_shape(self, input_shape, stride, ks_i=0):
        out_x = int( (input_shape[1]-self.kernel_size[ks_i]) // stride) + 1
        out_y = int( (input_shape[0]-self.kernel_size[ks_i]) // stride) + 1
        return (out_y, out_x)

    def compute_post_tconv_shape(self, input_shape, stride, ks_i=0):
        out_x = (input_shape[1]-1)*stride + self.kernel_size[ks_i]
        out_y = (input_shape[0]-1)*stride + self.kernel_size[ks_i]
        return (out_y, out_x)

    @tf.function
    def call(self, x):

        x = self.ae_net(x)

        return x

    def train_step(self, data):

        # x, y = data
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            decoded = self.ae_net(x, training=True)
            loss = self.compiled_loss(
                y,
                decoded,
                sample_weight,
                regularization_losses=self.losses
            )

        self._validate_target_and_loss(decoded, loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        kwargs = {}
        if self.use_attention_module == True:
            for ks_i in range(self.num_kernel_sizes):
                for i in range(len(self.encoder_attention_modules_list[ks_i])):
                    l = self.encoder_attention_modules_list[ks_i][i]
                    kwargs['{}__encoder_attention_module_{}_lambda'.format(ks_i, i)] = l.lambda_att
                for i in range(len(self.decoder_attention_modules_list[ks_i])):
                    l = self.decoder_attention_modules_list[ks_i][i]
                    kwargs['{}__decoder_attention_module_{}_lambda'.format(ks_i, i)] = l.lambda_att
        for ks_i in range(self.num_kernel_sizes):
            kwargs['{}__encoded_vec_wt'.format(ks_i)] = self.encoded_vec_weights[ks_i]
        for ks_i in range(self.num_kernel_sizes):
            kwargs['{}__decoded_vec_wt'.format(ks_i)] = self.decoded_vec_weights[ks_i]

        return self.compute_metrics(x, y, decoded, sample_weight, **kwargs)

    def compute_metrics(self, x, y, decoded, sample_weight, **kwargs):
        metric_results = super().compute_metrics(x, y, decoded, sample_weight)
        for key in kwargs.keys():
            metric_results[key] = kwargs[key]
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
        ### saving class attributes
        class_dict = {
            'data_dim':self.data_dim,
            'kernel_size':list(self.kernel_size),
            'enc_filters':list(self.enc_filters),
            'enc_strides':list(self.enc_strides),
            'enc_attn_placement':list(self.enc_attn_placement),
            'dec_filters':list(self.dec_filters),
            'dec_strides':list(self.dec_strides),
            'dec_attn_placement':list(self.dec_attn_placement),
            'lambda_reg':self.lambda_reg,
            'reg_name':self.reg_name,
            'enc_layer_act_func':self.enc_layer_act_func,
            'enc_final_layer_act_func':self.enc_final_layer_act_func,
            'dec_layer_act_func':self.dec_layer_act_func,
            'dec_final_layer_act_func':self.dec_final_layer_act_func,
            'load_file':self.load_file,
            'stddev':self.stddev,
            'use_weights_post_dense':self.use_weights_post_dense,
            'use_batch_norm':self.use_batch_norm,
            'use_attention_module':self.use_attention_module,
            'use_periodic_padding':self.use_periodic_padding,
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
