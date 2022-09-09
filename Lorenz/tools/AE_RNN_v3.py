from tensorflow.keras import layers

#################################### AE-RNN ####################################

# setting up the RNN class
class AE_RNN:
    """
    AE-RNN that advances (in time) the latent space representation
    """
    def __init__(self, encoder_net, decoder_net, rnn_net):
        
        super(AE_RNN, self).__init__()

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.rnn_net = rnn_net

        return

    def predict(self, inputs, out_time, training=False):

        if self.rnn_net.T_output != out_time:
            self.rnn_net.update_Toutput(out_time)

        output = layers.TimeDistributed(self.encoder_net)(inputs, training=training)
        output = self.rnn_net.predict(output)
        output = layers.TimeDistributed(self.decoder_net)(output, training=training)

        return output

################################################################################