import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

################################################################################

class Autoencoder(Model):
    """
    Autoencoder network that finds the latent space representation
    """
    def __init__(
            self, data_dim=6,
            enc_layers=[16,12,8,8,4,4,2],
            dec_layers=[2,4,4,8,8,12,16],
            latent_space_dim=2,
            lambda_reg=0.0,
            reg_name='L2',
            enc_layer_act_func='elu',
            enc_final_layer_act_func='tanh',
            dec_layer_act_func='elu',
            dec_final_layer_act_func='linear',
            batch_norm=True,
            load_file=None):
        
        super(Autoencoder, self).__init__()

        self.load_file = load_file
        if load_file == None:
            self.data_dim=data_dim
            self.enc_layers=enc_layers
            self.dec_layers=dec_layers
            self.latent_space_dim=latent_space_dim
            self.lambda_reg=lambda_reg
            self.reg_name=reg_name
            self.enc_layer_act_func=enc_layer_act_func
            self.enc_final_layer_act_func=enc_final_layer_act_func
            self.dec_layer_act_func=dec_layer_act_func
            self.dec_final_layer_act_func=dec_final_layer_act_func
            self.batch_norm=batch_norm
            # self.tf_version = tf.__version__
            # self.compile_options=None
        else:
            with open(load_file, 'r') as f:
                lines = f.readlines()
            load_dict = eval(lines[0])
            # load_dict = ''
            # for line in lines:
            #     load_dict += line
            # load_dict = eval(load_dict)
            self.data_dim=load_dict['data_dim']
            self.enc_layers=load_dict['enc_layers']
            self.dec_layers=load_dict['dec_layers']
            self.latent_space_dim=load_dict['latent_space_dim']
            self.lambda_reg=load_dict['lambda_reg']
            self.reg_name=load_dict['reg_name']
            self.enc_layer_act_func=load_dict['enc_layer_act_func']
            self.enc_final_layer_act_func=load_dict['enc_final_layer_act_func']
            self.dec_layer_act_func=load_dict['dec_layer_act_func']
            self.dec_final_layer_act_func=load_dict['dec_final_layer_act_func']
            self.batch_norm=load_dict['batch_norm']


        ### input vector
        input_shape = (self.data_dim,)
        input_vec = Input(shape=input_shape)



        ### the encoder network
        if self.reg_name is not None and self.lambda_reg is not None and self.lambda_reg != 0:
            reg = eval('tf.keras.regularizers.'+self.reg_name)
            encoder_layers_list = []
            for i in range(len(self.enc_layers)):
                neurons = self.enc_layers[i]
                encoder_layers_list.append(
                    layers.Dense(
                        neurons,
                        activation=self.enc_layer_act_func,
                        kernel_regularizer=reg(self.lambda_reg),
                        bias_regularizer=reg(self.lambda_reg)
                    )
                )
                if self.batch_norm is True:
                    encoder_layers_list.append(layers.BatchNormalization())
            encoder_layers_list.append(
                layers.Dense(
                    latent_space_dim,
                    activation=self.enc_final_layer_act_func,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg)
                )
            )
        else:
            encoder_layers_list = []
            for i in range(len(self.enc_layers)):
                neurons = self.enc_layers[i]
                encoder_layers_list.append(
                    layers.Dense(
                        neurons,
                        activation=self.enc_layer_act_func
                    )
                )
                if self.batch_norm is True:
                    encoder_layers_list.append(layers.BatchNormalization())
            encoder_layers_list.append(
                layers.Dense(
                    latent_space_dim,
                    activation=self.enc_final_layer_act_func
                )
            )

        x = encoder_layers_list[0](input_vec)
        for i in range(1, len(encoder_layers_list)-1):
            x = encoder_layers_list[i](x)
        encoded_vec = encoder_layers_list[-1](x)

        self.encoder_layers_list = encoder_layers_list
        self.encoder_net = Model(inputs=input_vec, outputs=encoded_vec)

        ### the decoder network
        if self.reg_name is not None and self.lambda_reg is not None and self.lambda_reg != 0:
            decoder_layers_list = []
            for i in range(len(self.dec_layers)):
                neurons = self.dec_layers[i]
                decoder_layers_list.append(
                    layers.Dense(
                        neurons,
                        activation=self.dec_layer_act_func,
                        kernel_regularizer=reg(self.lambda_reg),
                        bias_regularizer=reg(self.lambda_reg)
                    )
                )
                if self.batch_norm is True:
                    decoder_layers_list.append(layers.BatchNormalization())
            decoder_layers_list.append(
                layers.Dense(
                    data_dim,
                    activation=self.dec_final_layer_act_func,
                    kernel_regularizer=reg(self.lambda_reg),
                    bias_regularizer=reg(self.lambda_reg)
                )
            )
        else:
            decoder_layers_list = []
            for i in range(len(self.dec_layers)):
                neurons = self.dec_layers[i]
                decoder_layers_list.append(
                    layers.Dense(
                        neurons,
                        activation=self.dec_layer_act_func,
                    )
                )
                if self.batch_norm is True:
                    decoder_layers_list.append(layers.BatchNormalization())
            decoder_layers_list.append(
                layers.Dense(
                    data_dim,
                    activation=self.dec_final_layer_act_func,
                )
            )

        
        encoded_input_vec = Input(shape=encoded_vec.shape[-1],)
        x = decoder_layers_list[0](encoded_input_vec)
        for i in range(1, len(decoder_layers_list)-1):
            x = decoder_layers_list[i](x)
        decoded_vec = decoder_layers_list[-1](x)

        self.decoder_layers_list = decoder_layers_list
        self.decoder_net = Model(inputs=encoded_input_vec, outputs=decoded_vec)

        ### all combined, the autoencoder
        x = self.encoder_net(input_vec)
        x = self.decoder_net(x)
        self.ae_net = Model(inputs=input_vec, outputs=x)

        temp = tf.ones(shape=(1, self.data_dim,))
        temp = self(temp)

        return

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

    def save_everything(self, file_name, H5=True):

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
            'batch_norm':self.batch_norm
        }
        with open(file_name+'_class_dict.txt', 'w') as f:
            f.write(str(class_dict))
            # s = '{\n'
            # for entry in class_dict.keys():
            #     s += '    '+str(entry)+':'+str(class_dict[entry])+',\n'
            # s += '}'
            # f.write(s)

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