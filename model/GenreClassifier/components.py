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
        n_voices = hvo.shape[2] // 3
        hit = hvo[:, :, 0:n_voices]
        vel = hvo[:, :, n_voices:2*n_voices]
        offset = hvo[:, :, 2*n_voices:]
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
