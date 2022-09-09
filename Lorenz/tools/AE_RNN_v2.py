from tensorflow.keras import layers
import numpy as np

#################################### AE-RNN ####################################

# setting up the RNN class
class AE_RNN:
    """
    AE-RNN that advances (in time) a given input
    """
    def __init__(self, encoder_net, decoder_net, rnn_net):
        '''
        `encoder_net` : encoder network of the autoencoder
        `decoder_net` : decoder network of the autoencoder
        `rnn_net` : RNN network that advances in time the latent space
                    representation
        '''
        
        super(AE_RNN, self).__init__()

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.rnn_net = rnn_net

        return

    def predict(self, inputs, out_time, params=None, training=False):
        '''
        `inputs` : dimensions = [batch_size, time_steps, input_data_dim]
        `out_time` : total time for which prediction is to be made
        `params` : if None - RNN network uses *only* the latent space variables
                   else - must provide the additonal parameters to pass into
                          the RNN. (1D numpy array)
        '''

        output = layers.TimeDistributed(self.encoder_net)(inputs, training=training)

        # ouput dimensions = [batch_size, time_steps, num_latent_variables]
        if params is not None:
            new_shape = np.array(output.shape)
            new_shape[-1] += len(params)
            temp = np.empty(shape=new_shape)
            output = output.numpy()
            for i in range(0, output.shape[0]):
                for j in range(0, output.shape[1]):
                    temp[i, j, 0:output.shape[-1]] = output[i, j, :]
                    temp[i, j, output.shape[-1]:] = params
            output = temp
        output = self.rnn_net.predict(inputs=output, out_time=out_time)
        
        # output dimensions
        # if params is None:
        #     [batch_size, time_steps, num_latent_variables]
        # else:
        #     [batch_size, time_steps, num_latent_variables+num_params]
        if params is not None:
            output = output.numpy()
            output = output[:, :, 0:-len(params)]
        output = layers.TimeDistributed(self.decoder_net)(output, training=training)

        return output

################################################################################