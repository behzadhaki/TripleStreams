#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math


# --------------------------------------------------------------------------------
# ------------       Positinal Encoding BLOCK                ---------------------
# --------------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the token
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


class InputGrooveLayerWithTwoControls(torch.nn.Module):
    """
    Receives a hvo tensor of shape (batch, max_len, 3), and returns a tensor of shape
    (batch, 33, d_model) when prepend_control_tokens=True, or (batch, 32, d_model) when False

    """

    def __init__(self, embedding_size, d_model, max_len,
                 velocity_dropout, offset_dropout,
                 positional_encoding_dropout, n_encoding_control1_tokens, n_encoding_control2_tokens,
                 prepend_control_tokens=False):
        super(InputGrooveLayerWithTwoControls, self).__init__()
        self.prepend_control_tokens = prepend_control_tokens
        self.n_encoding_control1_tokens = n_encoding_control1_tokens
        self.n_encoding_control2_tokens = n_encoding_control2_tokens

        if prepend_control_tokens:
            # When prepending, embeddings should output d_model dimensions directly
            self.encoding_control1_embedding = torch.nn.Embedding(num_embeddings=self.n_encoding_control1_tokens,
                                                                  embedding_dim=d_model)
            self.encoding_control2_embedding = torch.nn.Embedding(num_embeddings=self.n_encoding_control2_tokens,
                                                                  embedding_dim=d_model)
            # Positional encoding needs to handle max_len + 2 (for the two control tokens)
            self.PositionalEncoding = PositionalEncoding(d_model, max_len + 2, positional_encoding_dropout)
        else:
            # Original behavior: embeddings output max_len * d_model for summation
            self.encoding_control1_embedding = torch.nn.Embedding(num_embeddings=self.n_encoding_control1_tokens,
                                                                  embedding_dim=max_len * d_model)
            self.encoding_control2_embedding = torch.nn.Embedding(num_embeddings=self.n_encoding_control2_tokens,
                                                                  embedding_dim=max_len * d_model)
            self.PositionalEncoding = PositionalEncoding(d_model, max_len, positional_encoding_dropout)

        self.velocity_dropout = torch.nn.Dropout(p=velocity_dropout)
        self.offset_dropout = torch.nn.Dropout(p=offset_dropout)
        self.HitsLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.VelocitiesLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.OffsetsLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.HitsReLU = torch.nn.ReLU()
        self.VelocitiesReLU = torch.nn.ReLU()
        self.OffsetsReLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.encoding_control1_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoding_control2_embedding.weight.data.uniform_(-initrange, initrange)
        self.HitsLinear.weight.data.uniform_(-initrange, initrange)
        self.HitsLinear.bias.data.zero_()
        self.VelocitiesLinear.bias.data.zero_()
        self.VelocitiesLinear.weight.data.uniform_(-initrange, initrange)
        self.OffsetsLinear.bias.data.zero_()
        self.OffsetsLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, hvo, encoding_control1_token, encoding_control2_token):
        '''
        :param hvo: shape (batch, 32, 3)
        :return: when prepend_control_tokens=True: (batch, 34, d_model)
                 when prepend_control_tokens=False: (batch, 32, d_model)
        '''
        if len(encoding_control1_token.shape) == 1:
            encoding_control1_token = encoding_control1_token.unsqueeze(-1)

        if len(encoding_control2_token.shape) == 1:
            encoding_control2_token = encoding_control2_token.unsqueeze(-1)

        hit = hvo[:, :, 0:1]
        vel = hvo[:, :, 1:2]
        offset = hvo[:, :, 2:3]

        # Process HVO features
        hits_projection = self.HitsReLU(self.HitsLinear(hit))
        velocities_projection = self.VelocitiesReLU(self.VelocitiesLinear(self.velocity_dropout(vel)))
        offsets_projection = self.OffsetsReLU(self.OffsetsLinear(self.offset_dropout(offset)))

        if self.prepend_control_tokens:
            # Get control embeddings - shape (batch, 1, d_model)
            control1_embedding = self.encoding_control1_embedding(encoding_control1_token)  # (batch, 1, d_model)
            control2_embedding = self.encoding_control2_embedding(encoding_control2_token)  # (batch, 1, d_model)

            # Combine HVO projections
            hvo_projection = hits_projection + velocities_projection + offsets_projection  # (batch, 32, d_model)

            # Prepend control tokens to the sequence
            sequence = torch.cat([control1_embedding, control2_embedding, hvo_projection],
                                 dim=1)  # (batch, 34, d_model)
        else:
            # Original behavior: sum control embeddings with HVO projections
            control1_embedding = self.encoding_control1_embedding(encoding_control1_token)
            control1_embedding = control1_embedding.view(-1, hits_projection.shape[1], hits_projection.shape[-1])
            control2_embedding = self.encoding_control2_embedding(encoding_control2_token)
            control2_embedding = control2_embedding.view(-1, hits_projection.shape[1], hits_projection.shape[-1])
            sequence = hits_projection + velocities_projection + offsets_projection + control1_embedding + control2_embedding

        out = self.PositionalEncoding(sequence)
        return out, hit[:, :, 0], sequence


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

    def __init__(self, max_len, d_model, latent_dim, prepend_control_tokens=False):
        super(LatentLayer, self).__init__()

        # Adjust input size based on whether control tokens are prepended
        if prepend_control_tokens:
            input_size = int((max_len + 2) * d_model)  # +2 for the two encoder control tokens
        else:
            input_size = int(max_len * d_model)

        self.fc_mu = torch.nn.Linear(input_size, latent_dim)
        self.fc_var = torch.nn.Linear(input_size, latent_dim)

    def init_weights(self, initrange=0.1):
        self.fc_mu.bias.data.zero_()
        self.fc_mu.weight.data.uniform_(-initrange, initrange)
        self.fc_var.bias.data.zero_()
        self.fc_var.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self, src: torch.Tensor):
        """ converts the input into a latent space representation

        :param src: (Tensor) Input tensor to REPARAMETERIZE [N x seq_len x d_model]
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

        eg. from (batch, 32, 128) to (batch, 32, 3)

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
    """ Embeds the output controls and adds them to the latent space representation.
    When prepend_control_tokens=False: result is reshaped to [batch, max_len, d_model_dec]
    When prepend_control_tokens=True: result is [batch, max_len + 3, d_model_dec] (3 control tokens prepended)
    """

    def __init__(self, max_len, latent_dim, d_model,
                 n_decoding_control1_tokens, n_decoding_control2_tokens, n_decoding_control3_tokens,
                 prepend_control_tokens=False):

        super(DecoderInput, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.prepend_control_tokens = prepend_control_tokens

        if prepend_control_tokens:
            # Control embeddings output d_model dimensions for prepending
            self.control1_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control1_tokens,
                                                         embedding_dim=d_model)
            self.control2_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control2_tokens,
                                                         embedding_dim=d_model)
            self.control3_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control3_tokens,
                                                         embedding_dim=d_model)
        else:
            # Original behavior: embeddings output latent_dim for summation
            self.control1_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control1_tokens,
                                                         embedding_dim=latent_dim)
            self.control2_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control2_tokens,
                                                         embedding_dim=latent_dim)
            self.control3_embedding = torch.nn.Embedding(num_embeddings=n_decoding_control3_tokens,
                                                         embedding_dim=latent_dim)

        self.fc = torch.nn.Linear(int(latent_dim), int(max_len * d_model))
        self.reLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.control1_embedding.weight.data.uniform_(-initrange, initrange)
        self.control2_embedding.weight.data.uniform_(-initrange, initrange)
        self.control3_embedding.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self,
                latent_z: torch.Tensor,
                decoding_control1_token: torch.Tensor,
                decoding_control2_token: torch.Tensor,
                decoding_control3_token: torch.Tensor):
        """
        applies the activation functions and reshapes the input tensor to fix dimensions with decoder

        :param latent_z: shape (batch, latent_dim)
        :param decoding_control1_token:  shape (batch) or (batch, 1)
        :param decoding_control2_token:  shape (batch) or (batch, 1)
        :param decoding_control3_token:  shape (batch) or (batch, 1)

        :return:
                when prepend_control_tokens=False: (batch, max_len, d_model_dec)
                when prepend_control_tokens=True: (batch, max_len + 3, d_model_dec)
        """

        if len(decoding_control1_token.shape) == 2:
            decoding_control1_token = decoding_control1_token.squeeze(-1)
        if len(decoding_control2_token.shape) == 2:
            decoding_control2_token = decoding_control2_token.squeeze(-1)
        if len(decoding_control3_token.shape) == 2:
            decoding_control3_token = decoding_control3_token.squeeze(-1)

        if self.prepend_control_tokens:
            # Get control embeddings - each has shape (batch, 1, d_model)
            control1_emb = self.control1_embedding(decoding_control1_token).unsqueeze(1)  # (batch, 1, d_model)
            control2_emb = self.control2_embedding(decoding_control2_token).unsqueeze(1)  # (batch, 1, d_model)
            control3_emb = self.control3_embedding(decoding_control3_token).unsqueeze(1)  # (batch, 1, d_model)

            # Project latent_z and reshape to (batch, max_len, d_model)
            latent_projected = self.reLU(self.fc.forward(latent_z)).view(-1, self.max_len, self.d_model)

            # Prepend control tokens to the sequence
            output = torch.cat([control1_emb, control2_emb, control3_emb, latent_projected],
                               dim=1)  # (batch, max_len + 3, d_model)
        else:
            # Original behavior: sum control embeddings with latent_z
            latent_z_with_controls = (latent_z +
                                      self.control1_embedding(decoding_control1_token) +
                                      self.control2_embedding(decoding_control2_token) +
                                      self.control3_embedding(decoding_control3_token))

            output = self.reLU(self.fc.forward(latent_z_with_controls)).view(-1, self.max_len, self.d_model)

        return output