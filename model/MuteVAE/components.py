#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math


# --------------------------------------------------------------------------------
# ------------       Positinal Encoding BLOCK                ---------------------
# --------------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class InputGrooveLayer(torch.nn.Module):
    """
    Receives a hvo tensor of shape (batch, max_len, 3), and returns a tensor of shape
    (batch, 33, d_model)

    """

    def __init__(self, embedding_size, d_model, max_len,
                 velocity_dropout, offset_dropout,
                 positional_encoding_dropout):
        super(InputGrooveLayer, self).__init__()
        self.velocity_dropout = torch.nn.Dropout(p=velocity_dropout)
        self.offset_dropout = torch.nn.Dropout(p=offset_dropout)
        self.HitsLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.VelocitiesLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.OffsetsLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.HitsReLU = torch.nn.ReLU()
        self.VelocitiesReLU = torch.nn.ReLU()
        self.OffsetsReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, (max_len), positional_encoding_dropout)

    def init_weights(self, initrange=0.1):
        self.HitsLinear.bias.data.zero_()
        self.HitsLinear.weight.data.uniform_(-initrange, initrange)
        self.VelocitiesLinear.bias.data.zero_()
        self.VelocitiesLinear.weight.data.uniform_(-initrange, initrange)
        self.OffsetsLinear.bias.data.zero_()
        self.OffsetsLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, hvo, ):
        '''

        :param hvo: shape (batch, 32, 3)
        :return:
        '''
        hit = hvo[:, :, 0:1]
        vel = hvo[:, :, 1:2]
        offset = hvo[:, :, 2:3]
        # hvo_ = torch.cat((hit, self.velocity_dropout(vel), self.offset_dropout(offset)), dim=2)
        hits_projection = self.HitsReLU(self.HitsLinear(hit))
        velocities_projection = self.VelocitiesReLU(self.VelocitiesLinear(self.velocity_dropout(vel)))
        offsets_projection = self.OffsetsReLU(self.OffsetsLinear(self.offset_dropout(offset)))
        hvo_projection = hits_projection + velocities_projection + offsets_projection
        out = self.PositionalEncoding(hvo_projection)
        return out, hit[:, :, 0], hvo_projection


# --------------------------------------------------------------------------------
# ------------                  ENCODER BLOCK                ---------------------
# --------------------------------------------------------------------------------
class Encoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, ):
        """Transformer Encoder Layers Wrapped into a Single Module"""
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)

        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=norm_encoder)

    def forward(self, src):
        """
        input and output both have shape (batch, seq_len, embed_dim)
        :param src:
        :return:
        """
        out = self.Encoder(src)
        return out


# --------------------------------------------------------------------------------
# ------------         VARIAIONAL REPARAMETERIZE BLOCK       ---------------------
# --------------------------------------------------------------------------------
class LatentLayer(torch.nn.Module):
    """ Latent variable reparameterization layer

   :param input: (Tensor) Input tensor to REPARAMETERIZE [B x max_len_enc x d_model_enc]
   :return: mu, log_var, z (Tensor) [B x max_len_enc x d_model_enc]
   """

    def __init__(self, max_len, d_model, latent_dim):
        super(LatentLayer, self).__init__()

        self.fc_mu = torch.nn.Linear(int(max_len * d_model), latent_dim)
        self.fc_var = torch.nn.Linear(int(max_len * d_model), latent_dim)

    def init_weights(self, initrange=0.1):
        self.fc_mu.bias.data.zero_()
        self.fc_mu.weight.data.uniform_(-initrange, initrange)
        self.fc_var.bias.data.zero_()
        self.fc_var.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self, src: torch.Tensor):
        """ converts the input into a latent space representation

        :param src: (Tensor) Input tensor to REPARAMETERIZE [N x max_encoder_len x d_model]
        :return:  mu , logvar, z (each with dimensions [N, latent_dim_size])
        """
        result = torch.flatten(src, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # Reparameterize
        z = self.reparametrize(mu, log_var)

        return mu, log_var, z

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z


class SingleFeatureOutputLayer(torch.nn.Module):
    """ Maps the dimension of the output of a decoded sequence into to the dimension of the output

        eg. from (batch, 32, 128) to (batch, 32, 9)

        for either hits, velocities or offsets
        """

    def __init__(self, embedding_size, d_model):
        """
        Output layer of the transformer model
        :param embedding_size: size of the embedding (output dim at each time step)
        :param d_model:     size of the model         (input dim at each time step)
        """
        super(SingleFeatureOutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.fill_(0.5)
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        logits = self.Linear(decoder_out)

        return logits


class DecoderInput(torch.nn.Module):
    """ Embeds the controls and adds them to the latent space representation.
    Then the result is reshaped to [batch, max_len, d_model_dec] to be used as input to the decoder
    """

    def __init__(self, max_len, latent_dim, d_model):
        super(DecoderInput, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.kick_is_muted_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=latent_dim)
        self.snare_is_muted_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=latent_dim)
        self.hat_is_muted_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=latent_dim)
        self.tom_is_muted_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=latent_dim)
        self.cymbal_is_muted_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=latent_dim)

        self.fc = torch.nn.Linear(int(latent_dim), int(max_len * d_model))
        self.reLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.kick_is_muted_embedding.weight.data.uniform_(-initrange, initrange)
        self.snare_is_muted_embedding.weight.data.uniform_(-initrange, initrange)
        self.hat_is_muted_embedding.weight.data.uniform_(-initrange, initrange)
        self.tom_is_muted_embedding.weight.data.uniform_(-initrange, initrange)
        self.cymbal_is_muted_embedding.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self,
                latent_z: torch.Tensor,
                kick_is_muted: torch.Tensor,
                snare_is_muted: torch.Tensor,
                hat_is_muted: torch.Tensor,
                tom_is_muted: torch.Tensor,
                cymbal_is_muted: torch.Tensor):
        """
        applies the activation functions and reshapes the input tensor to fix dimensions with decoder

        :param latent_z: shape (batch, latent_dim)
        :param kick_is_muted:  shape (batch) or (batch, 1)
        :param snare_is_muted:  shape (batch) or (batch, 1)
        :param hat_is_muted:  shape (batch) or (batch, 1)
        :param tom_is_muted:  shape (batch) or (batch, 1)
        :param cymbal_is_muted:  shape (batch) or (batch, 1)

        :return:
                projected (latent+controls) into the shape (batch, max_len, d_model_dec)
        """

        if len(kick_is_muted.shape) == 2:
            kick_is_muted = kick_is_muted.squeeze(-1)
        if len(snare_is_muted.shape) == 2:
            snare_is_muted = snare_is_muted.squeeze(-1)
        if len(hat_is_muted.shape) == 2:
            hat_is_muted = hat_is_muted.squeeze(-1)
        if len(tom_is_muted.shape) == 2:
            tom_is_muted = tom_is_muted.squeeze(-1)
        if len(cymbal_is_muted.shape) == 2:
            cymbal_is_muted = cymbal_is_muted.squeeze(-1)

        latent_z = (latent_z +
                    self.kick_is_muted_embedding(kick_is_muted) + self.snare_is_muted_embedding(snare_is_muted) +
                    self.hat_is_muted_embedding(hat_is_muted) + self.tom_is_muted_embedding(tom_is_muted) +
                    self.cymbal_is_muted_embedding(cymbal_is_muted))

        return self.reLU(self.fc.forward(latent_z)).view(-1, self.max_len, self.d_model)

    @torch.jit.export
    def interpolate_2d(self,
                    interpolation_factor: float,
                    latent_z_1: torch.Tensor,
                    mute_controls_1: torch.Tensor,
                    latent_z_2: torch.Tensor,
                    mute_controls_2: torch.Tensor):
        """
        Interpolates between two latent spaces and controls

        :param interpolation_factor:  float
        :param latent_z_1:  shape (batch, latent_dim) or (latent_dim)
        :param mute_controls_1:  shape (batch, 5) or (5)
        :param latent_z_2:  shape (batch, latent_dim) or (latent_dim)
        :param mute_controls_2:  shape (batch, 5) or (5)
        :return:
        """
        if len(latent_z_1.shape) == 1:
            latent_z_1 = latent_z_1.unsqueeze(0)
        if len(latent_z_2.shape) == 1:
            latent_z_2 = latent_z_2.unsqueeze(0)
        if len(mute_controls_1.shape) == 1:
            mute_controls_1 = mute_controls_1.unsqueeze(0)
        if len(mute_controls_2.shape) == 1:
            mute_controls_2 = mute_controls_2.unsqueeze(0)

        latent_z = (latent_z_2 * interpolation_factor + latent_z_1 * (1 - interpolation_factor))

        mute_latent_1 = (self.kick_is_muted_embedding(mute_controls_1[:, 0]) +
                         self.snare_is_muted_embedding(mute_controls_1[:, 1]) +
                         self.hat_is_muted_embedding(mute_controls_1[:, 2]) +
                         self.tom_is_muted_embedding(mute_controls_1[:, 3]) +
                         self.cymbal_is_muted_embedding(mute_controls_1[:, 4]))

        mute_latent_2 = (self.kick_is_muted_embedding(mute_controls_2[:, 0]) +
                         self.snare_is_muted_embedding(mute_controls_2[:, 1]) +
                         self.hat_is_muted_embedding(mute_controls_2[:, 2]) +
                         self.tom_is_muted_embedding(mute_controls_2[:, 3]) +
                         self.cymbal_is_muted_embedding(mute_controls_2[:, 4]))

        mute_latent = (mute_latent_2 * interpolation_factor + mute_latent_1 * (1 - interpolation_factor))

        return self.reLU(self.fc.forward(latent_z + mute_latent)).view(-1, self.max_len, self.d_model)

    @torch.jit.export
    def interpolate_3d(self,
                       ab_interpolation_factor: float,
                       abc_interpolation_factor: float,
                       latent_z_a: torch.Tensor,
                       mute_controls_a: torch.Tensor,
                       latent_z_b: torch.Tensor,
                       mute_controls_b: torch.Tensor,
                       latent_z_c: torch.Tensor,
                       mute_controls_c: torch.Tensor):
        if len(latent_z_a.shape) == 1:
            latent_z_a = latent_z_a.unsqueeze(0)
        if len(latent_z_b.shape) == 1:
            latent_z_b = latent_z_b.unsqueeze(0)
        if len(latent_z_c.shape) == 1:
            latent_z_c = latent_z_c.unsqueeze(0)
        if len(mute_controls_a.shape) == 1:
            mute_controls_a = mute_controls_a.unsqueeze(0)
        if len(mute_controls_b.shape) == 1:
            mute_controls_b = mute_controls_b.unsqueeze(0)
        if len(mute_controls_c.shape) == 1:
            mute_controls_c = mute_controls_c.unsqueeze(0)

        latent_z = (latent_z_b * ab_interpolation_factor + latent_z_a * (1 - ab_interpolation_factor))
        latent_z = (latent_z_c * abc_interpolation_factor + latent_z * (1 - abc_interpolation_factor))

        mute_latent_a = (self.kick_is_muted_embedding(mute_controls_a[:, 0]) +
                         self.snare_is_muted_embedding(mute_controls_a[:, 1]) +
                         self.hat_is_muted_embedding(mute_controls_a[:, 2]) +
                         self.tom_is_muted_embedding(mute_controls_a[:, 3]) +
                         self.cymbal_is_muted_embedding(mute_controls_a[:, 4]))

        mute_latent_b = (self.kick_is_muted_embedding(mute_controls_b[:, 0]) +
                            self.snare_is_muted_embedding(mute_controls_b[:, 1]) +
                            self.hat_is_muted_embedding(mute_controls_b[:, 2]) +
                            self.tom_is_muted_embedding(mute_controls_b[:, 3]) +
                            self.cymbal_is_muted_embedding(mute_controls_b[:, 4]))

        mute_latent_c = (self.kick_is_muted_embedding(mute_controls_c[:, 0]) +
                            self.snare_is_muted_embedding(mute_controls_c[:, 1]) +
                            self.hat_is_muted_embedding(mute_controls_c[:, 2]) +
                            self.tom_is_muted_embedding(mute_controls_c[:, 3]) +
                            self.cymbal_is_muted_embedding(mute_controls_c[:, 4]))

        mute_latent = (mute_latent_b * ab_interpolation_factor + mute_latent_a * (1 - ab_interpolation_factor))
        mute_latent = (mute_latent_c * abc_interpolation_factor + mute_latent * (1 - abc_interpolation_factor))

        return self.reLU(self.fc.forward(latent_z + mute_latent)).view(-1, self.max_len, self.d_model)


