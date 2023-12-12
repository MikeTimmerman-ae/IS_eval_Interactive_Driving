from pretext.pretext_models.pretext_rnn_base import LSTM_Pretext

from .decoders import *

'''
network for trait prediction without labels using VAE
the network has two parts:
 1. encoder: base is always GRU
 2. decoder: base can be GRU (ours) or MLP (Morton et al)
'''
class CVAEIntentPredictor(nn.Module):
    def __init__(self, obs_shape, task, decoder_base, config):
        super(CVAEIntentPredictor, self).__init__()
        # pretext_predict: train/test the predictor with supervised learning
        # rl_predict: use the predictor to infer in rl
        assert task in ['pretext_predict', 'rl_predict']

        self.config = config

        # initialize encoder
        self.encoder_base = LSTM_Pretext(obs_shape, config, task)

        # need to wrap the lstm or srnn encoder so that it can output mean and variance for vae
        # since we only have two classes of traits, latent size = 2 is more than enough
        # for latent_size, check driving_config.py
        self.encoder = Encoder(config, self.encoder_base, latent_size=config.env_config.ob_space.latent_size)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs_encoder, rnn_hxs_decoder, each_seq_len):
        # encoder
        z, _, _ = self.encoder(inputs, rnn_hxs_encoder, each_seq_len)
        return None, z, None, rnn_hxs_encoder, None, None

'''
vae encoder model
'''
class Encoder(nn.Module):
    # latent_size: the number of dimensions in latent state
    def __init__(self, config, base, latent_size):
        super(Encoder, self).__init__()
        self.base = base
        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.linear_means = init_(nn.Linear(config.network.rnn_hidden_size, latent_size))
        # self.linear_log_var = init_(nn.Linear(config.network.rnn_hidden_size, latent_size))

        if config.training.cuda:
            self.linear_means = self.linear_means.cuda()
            # self.linear_log_var = self.linear_log_var.cuda()

    # forward function, used for pretext training and testing
    # inputs: state dict
    # x: output features from SRNN [nenv, seq_len, human_num]
    # all humans share the same weights for calculating mean and variance
    # means: [human_num, latent_size], log_vars: [human_num, latent_size]
    def forward(self, inputs, rnn_hxs, each_seq_len, rl=False):
        # inputs: [pretext_nodes]: batch_size, 2
        # [pretext_spatial_edges]: batch_size, 2
        # [pretext_temporal_edges]: batch_size, 1
        # outputs: x: [batch_size, feature_size], rnn_hxs: [batch_size, feature_size]
        x, rnn_hxs = self.base.forward(inputs, rnn_hxs, each_seq_len, rl)
        means = self.linear_means(x) # [nenv, human_num, 2]
        return means, None, rnn_hxs

    # predict function, used in rl training (see vec_pretext_normalize.py)
    def predict(self, inputs, rnn_hxs, each_seq_len, rl=False):
        means, _, _ = self.forward(inputs, rnn_hxs, each_seq_len, rl)
        return means