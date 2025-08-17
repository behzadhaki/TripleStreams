#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math
from typing import List


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


class TensorBasedInputGrooveLayer(torch.nn.Module):
    """
    TorchScript-compatible input layer using tensors instead of lists.

    Control modes: 0 = prepend, 1 = add
    """

    def __init__(self, embedding_size, d_model, max_len,
                 velocity_dropout, offset_dropout,
                 positional_encoding_dropout,
                 n_encoding_control_tokens,
                 encoding_control_modes):
        super(TensorBasedInputGrooveLayer, self).__init__()

        self.n_controls = len(n_encoding_control_tokens)
        self.max_len = max_len
        self.d_model = d_model

        # Convert modes to integers and store as buffer (0=prepend, 1=add)
        mode_ints = [0 if mode == 'prepend' else 1 for mode in encoding_control_modes]
        self.register_buffer('encoding_control_modes', torch.tensor(mode_ints, dtype=torch.long))

        # Count prepended controls
        self.n_prepended_controls = sum(1 for mode in encoding_control_modes if mode == 'prepend')

        # Create control embeddings
        self.control_embeddings = torch.nn.ModuleList()
        for i, (n_tokens, mode) in enumerate(zip(n_encoding_control_tokens, encoding_control_modes)):
            if mode == 'prepend':
                # Prepend mode: embedding outputs d_model dimensions
                embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
            elif mode == 'add':
                # Add mode: embedding outputs max_len * d_model for reshaping and addition
                embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=max_len * d_model)
            self.control_embeddings.append(embedding)

        # Positional encoding handles max_len + number of prepended controls
        self.PositionalEncoding = PositionalEncoding(
            d_model, max_len + self.n_prepended_controls, positional_encoding_dropout
        )

        self.velocity_dropout = torch.nn.Dropout(p=velocity_dropout)
        self.offset_dropout = torch.nn.Dropout(p=offset_dropout)
        self.HitsLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.VelocitiesLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.OffsetsLinear = torch.nn.Linear(embedding_size // 3, d_model, bias=True)
        self.HitsReLU = torch.nn.ReLU()
        self.VelocitiesReLU = torch.nn.ReLU()
        self.OffsetsReLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        for embedding in self.control_embeddings:
            embedding.weight.data.uniform_(-initrange, initrange)
        self.HitsLinear.weight.data.uniform_(-initrange, initrange)
        self.HitsLinear.bias.data.zero_()
        self.VelocitiesLinear.bias.data.zero_()
        self.VelocitiesLinear.weight.data.uniform_(-initrange, initrange)
        self.OffsetsLinear.bias.data.zero_()
        self.OffsetsLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, hvo, encoding_control_tokens):
        '''
        :param hvo: shape (batch, max_len, 3)
        :param encoding_control_tokens: tensor of shape (batch, n_controls)
        :return: (batch, max_len + num_prepended_controls, d_model)
        '''
        hit = hvo[:, :, 0:1]
        vel = hvo[:, :, 1:2]
        offset = hvo[:, :, 2:3]

        # Process HVO features
        hits_projection = self.HitsReLU(self.HitsLinear(hit))
        velocities_projection = self.VelocitiesReLU(self.VelocitiesLinear(self.velocity_dropout(vel)))
        offsets_projection = self.OffsetsReLU(self.OffsetsLinear(self.offset_dropout(offset)))

        # Start with combined HVO projection
        hvo_projection = hits_projection + velocities_projection + offsets_projection

        # Process control tokens
        prepended_embeddings: List[torch.Tensor] = []

        for i, embedding in enumerate(self.control_embeddings):
            token = encoding_control_tokens[:, i]  # (batch,)
            mode = self.encoding_control_modes[i].item()  # 0=prepend, 1=add

            control_embedding = embedding(token)

            if mode == 0:  # prepend
                # control_embedding shape: (batch, d_model)
                prepended_embeddings.append(control_embedding.unsqueeze(1))  # (batch, 1, d_model)
            else:  # add (mode == 1)
                # Reshape and add to HVO projection
                control_reshaped = control_embedding.view(-1, self.max_len, self.d_model)
                hvo_projection = hvo_projection + control_reshaped

        # Combine prepended tokens with HVO projection
        if len(prepended_embeddings) > 0:
            sequence = torch.cat(prepended_embeddings + [hvo_projection], dim=1)
        else:
            sequence = hvo_projection

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
class TensorBasedLatentLayer(torch.nn.Module):
    """ Latent variable reparameterization layer that adapts to variable sequence lengths

   :param input: (Tensor) Input tensor to REPARAMETERIZE [B x seq_len x d_model_enc]
   :return: mu, log_var, z (Tensor) [B x latent_dim]
   """

    def __init__(self, max_len, d_model, latent_dim, n_prepended_encoding_controls=0):
        super(TensorBasedLatentLayer, self).__init__()

        # Calculate input size based on max sequence length including prepended controls
        input_size = int((max_len + n_prepended_encoding_controls) * d_model)

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


class TensorBasedDecoderInput(torch.nn.Module):
    """ TorchScript-compatible decoder input using tensors instead of lists.

    Control modes: 0 = prepend, 1 = add
    """

    def __init__(self, max_len, latent_dim, d_model,
                 n_decoding_control_tokens,
                 decoding_control_modes):

        super(TensorBasedDecoderInput, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_controls = len(n_decoding_control_tokens)

        # Convert modes to integers and store as buffer (0=prepend, 1=add)
        mode_ints = [0 if mode == 'prepend' else 1 for mode in decoding_control_modes]
        self.register_buffer('decoding_control_modes', torch.tensor(mode_ints, dtype=torch.long))

        # Count prepended controls
        self.n_prepended_controls = sum(1 for mode in decoding_control_modes if mode == 'prepend')

        # Create control embeddings
        self.control_embeddings = torch.nn.ModuleList()
        for i, (n_tokens, mode) in enumerate(zip(n_decoding_control_tokens, decoding_control_modes)):
            if mode == 'prepend':
                # Control embeddings output d_model dimensions for prepending
                embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
            elif mode == 'add':
                # Original behavior: embeddings output latent_dim for summation
                embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=latent_dim)
            self.control_embeddings.append(embedding)

        self.fc = torch.nn.Linear(int(latent_dim), int(max_len * d_model))
        self.reLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        for embedding in self.control_embeddings:
            embedding.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self, latent_z: torch.Tensor, decoding_control_tokens: torch.Tensor):
        """
        applies the activation functions and reshapes the input tensor to fix dimensions with decoder

        :param latent_z: shape (batch, latent_dim)
        :param decoding_control_tokens: tensor of shape (batch, n_controls)

        :return:
                when all controls are 'add': (batch, max_len, d_model_dec)
                when some/all controls are 'prepend': (batch, max_len + num_prepended_controls, d_model_dec)
        """
        # Start with latent_z
        latent_modified = latent_z
        prepended_embeddings: List[torch.Tensor] = []

        # Process each control token
        for i, embedding in enumerate(self.control_embeddings):
            token = decoding_control_tokens[:, i]  # (batch,)
            mode = self.decoding_control_modes[i].item()  # 0=prepend, 1=add

            control_embedding = embedding(token)

            if mode == 0:  # prepend
                # control_embedding shape: (batch, d_model)
                prepended_embeddings.append(control_embedding.unsqueeze(1))  # (batch, 1, d_model)
            else:  # add (mode == 1)
                # For add mode, add to latent_z
                latent_modified = latent_modified + control_embedding

        # Project modified latent and reshape to (batch, max_len, d_model)
        latent_projected = self.reLU(self.fc.forward(latent_modified)).view(-1, self.max_len, self.d_model)

        # Combine prepended controls with projected latent
        if len(prepended_embeddings) > 0:
            output = torch.cat(prepended_embeddings + [latent_projected], dim=1)
        else:
            output = latent_projected

        return output