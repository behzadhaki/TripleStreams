#  Copyright (c) 2022.
# Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math
from typing import List, Optional, Union, Tuple


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

    def forward(self, x: torch.Tensor):
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


# --------------------------------------------------------------------------------
# ------------       Control Self-Attention BLOCK           ---------------------
# --------------------------------------------------------------------------------
class ControlSelfAttention(torch.nn.Module):
    """
    Self-attention mechanism for learning inter-dependencies between controls.
    Allows controls to attend to each other and learn conditional relationships.
    """

    def __init__(self, n_controls, d_model, num_heads=4, dropout=0.1):
        super(ControlSelfAttention, self).__init__()
        self.n_controls = n_controls
        self.d_model = d_model
        self.num_heads = num_heads

        # Multi-head self-attention for controls
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization and feedforward
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 4, d_model),
            torch.nn.Dropout(dropout)
        )

    def init_weights(self, initrange=0.1):
        # Initialize feedforward layers
        for module in self.feedforward:
            if isinstance(module, torch.nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                module.bias.data.zero_()

    @torch.jit.export
    def forward(self, control_embeddings: torch.Tensor):
        """
        Apply self-attention to control embeddings to learn inter-dependencies

        :param control_embeddings: (batch, n_controls, d_model)
        :return: (batch, n_controls, d_model) - refined control embeddings
        """
        # Self-attention with residual connection
        attn_out, _ = self.multihead_attn(
            control_embeddings, control_embeddings, control_embeddings
        )
        control_embeddings = self.norm1(control_embeddings + attn_out)

        # Feedforward with residual connection
        ff_out = self.feedforward(control_embeddings)
        control_embeddings = self.norm2(control_embeddings + ff_out)

        return control_embeddings


# --------------------------------------------------------------------------------
# ------------       Compact Control Attention BLOCK        ---------------------
# --------------------------------------------------------------------------------
class CompactControlAttention(torch.nn.Module):
    """
    Compact control attention mechanism optimized for short sequences (32 steps).
    Each control token influences all sequence positions through learned attention.
    """

    def __init__(self, d_model, dropout=0.1):
        super(CompactControlAttention, self).__init__()
        self.d_model = d_model
        self.control_query = torch.nn.Linear(d_model, d_model)
        self.control_key = torch.nn.Linear(d_model, d_model)
        self.control_value = torch.nn.Linear(d_model, d_model)
        self.output_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def init_weights(self, initrange=0.1):
        for module in [self.control_query, self.control_key, self.control_value, self.output_proj]:
            module.weight.data.uniform_(-initrange, initrange)
            module.bias.data.zero_()

    @torch.jit.export
    def forward(self, sequence: torch.Tensor, control_embeddings: List[torch.Tensor]):
        """
        Apply compact control attention to sequence

        :param sequence: (batch, seq_len, d_model)
        :param control_embeddings: list of (batch, d_model) tensors
        :return: (batch, seq_len, d_model)
        """
        queries = self.control_query(sequence)  # (batch, seq_len, d_model)
        output = sequence

        for control_emb in control_embeddings:
            keys = self.control_key(control_emb).unsqueeze(1)  # (batch, 1, d_model)
            values = self.control_value(control_emb).unsqueeze(1)  # (batch, 1, d_model)

            # Attention weights for each position
            attn_weights = torch.softmax(
                torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.d_model),
                dim=-1
            )  # (batch, seq_len, 1)

            # Apply attention
            attended = torch.bmm(attn_weights, values)  # (batch, seq_len, d_model)
            attended = self.dropout(attended)
            output = output + self.output_proj(attended)

        return output


class TensorBasedInputGrooveLayer(torch.nn.Module):
    """
    TorchScript-compatible input layer using tensors instead of lists.
    Now supports both discrete (embedding) and continuous control values.

    Control modes: 0 = prepend, 1 = add, 2 = compact_attention, 3 = self_attention
    Control types: discrete (n_tokens is int) or continuous (n_tokens is None)
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

        # Convert modes to integers and store as buffer (0=prepend, 1=add, 2=compact_attention, 3=self_attention)
        mode_mapping = {'prepend': 0, 'add': 1, 'compact_attention': 2, 'self_attention': 3}
        mode_ints = [mode_mapping[mode] for mode in encoding_control_modes]
        self.register_buffer('encoding_control_modes', torch.tensor(mode_ints, dtype=torch.long))

        # Store control types (True for discrete, False for continuous)
        control_types = [n_tokens is not None for n_tokens in n_encoding_control_tokens]
        self.register_buffer('control_is_discrete', torch.tensor(control_types, dtype=torch.bool))

        # Count prepended controls and self-attention controls
        self.n_prepended_controls = sum(1 for mode in encoding_control_modes if mode == 'prepend')
        self.n_self_attention_controls = sum(1 for mode in encoding_control_modes if mode == 'self_attention')

        # Create control processing layers (embeddings for discrete, linear layers for continuous)
        self.control_embeddings = torch.nn.ModuleList()
        self.control_projections = torch.nn.ModuleList()

        for i, (n_tokens, mode) in enumerate(zip(n_encoding_control_tokens, encoding_control_modes)):
            if n_tokens is not None:  # Discrete control
                if mode in ['prepend', 'compact_attention', 'self_attention']:
                    # These modes need d_model dimensions
                    embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
                elif mode == 'add':
                    # Add mode: embedding outputs max_len * d_model for reshaping and addition
                    embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=max_len * d_model)
                else:
                    raise ValueError(f"Unknown encoding control mode: {mode}")
                self.control_embeddings.append(embedding)
                self.control_projections.append(torch.nn.Identity())  # Placeholder for discrete controls
            else:  # Continuous control
                self.control_embeddings.append(torch.nn.Identity())  # Placeholder for continuous controls
                if mode in ['prepend', 'compact_attention', 'self_attention']:
                    # These modes need d_model dimensions
                    projection = torch.nn.Linear(1, d_model)
                elif mode == 'add':
                    # Project continuous value to max_len * d_model for reshaping and addition
                    projection = torch.nn.Linear(1, max_len * d_model)
                else:
                    raise ValueError(f"Unknown encoding control mode: {mode}")
                self.control_projections.append(projection)

        # Add self-attention module for controls (always create it for TorchScript compatibility)
        if self.n_self_attention_controls > 0:
            self.control_self_attention = ControlSelfAttention(
                n_controls=self.n_self_attention_controls,
                d_model=d_model,
                dropout=positional_encoding_dropout
            )
        else:
            # Create dummy module for TorchScript compatibility
            self.control_self_attention = ControlSelfAttention(
                n_controls=1,  # Minimum size
                d_model=d_model,
                dropout=positional_encoding_dropout
            )

        # Add compact attention module
        self.compact_attention = CompactControlAttention(d_model, positional_encoding_dropout)

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
        for i, (embedding, projection) in enumerate(zip(self.control_embeddings, self.control_projections)):
            if self.control_is_discrete[i]:
                # Initialize embedding
                if hasattr(embedding, 'weight'):
                    embedding.weight.data.uniform_(-initrange, initrange)
            else:
                # Initialize linear projection
                if hasattr(projection, 'weight'):
                    projection.weight.data.uniform_(-initrange, initrange)
                    projection.bias.data.zero_()

        # Always initialize control_self_attention since it always exists now
        self.control_self_attention.init_weights(initrange)
        self.compact_attention.init_weights(initrange)
        self.HitsLinear.weight.data.uniform_(-initrange, initrange)
        self.HitsLinear.bias.data.zero_()
        self.VelocitiesLinear.bias.data.zero_()
        self.VelocitiesLinear.weight.data.uniform_(-initrange, initrange)
        self.OffsetsLinear.bias.data.zero_()
        self.OffsetsLinear.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self, hvo: torch.Tensor, encoding_control_tokens: torch.Tensor):
        '''
        :param hvo: shape (batch, max_len, 3)
        :param encoding_control_tokens: tensor of shape (batch, n_controls)
                                      - For discrete controls: integer token indices
                                      - For continuous controls: float values in [0, 1]
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

        # Stage 1: Process individual controls into embeddings
        control_embeddings_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, (embedding, projection) in enumerate(zip(self.control_embeddings, self.control_projections)):
            control_value = encoding_control_tokens[:, i]  # (batch,)
            is_discrete = self.control_is_discrete[i]
            mode = self.encoding_control_modes[i]  # Keep as tensor, don't call .item()

            if is_discrete:
                # Discrete control: use embedding
                control_token = control_value.long()  # Ensure integer type
                control_embedding = embedding(control_token)
            else:
                # Continuous control: use linear projection
                continuous_value = control_value.float().unsqueeze(1)  # (batch, 1)
                control_embedding = projection(continuous_value)

            control_embeddings_list.append((control_embedding, mode))

        # Stage 2: Apply self-attention to self_attention controls
        if self.n_self_attention_controls > 0:
            # Collect self-attention control embeddings
            self_attention_embeddings: List[torch.Tensor] = []
            self_attention_indices: List[int] = []

            for i, (control_embedding, mode) in enumerate(control_embeddings_list):
                if mode == 3:  # self_attention mode
                    self_attention_embeddings.append(control_embedding)
                    self_attention_indices.append(i)

            if len(self_attention_embeddings) > 0:
                # Stack for self-attention: (batch, n_self_attn_controls, d_model)
                stacked_embeddings = torch.stack(self_attention_embeddings, dim=1)
                refined_embeddings = self.control_self_attention(stacked_embeddings)

                # Replace original embeddings with refined ones
                for idx in range(len(self_attention_indices)):
                    list_idx = self_attention_indices[idx]
                    mode_tensor = control_embeddings_list[list_idx][1]
                    control_embeddings_list[list_idx] = (refined_embeddings[:, idx, :], mode_tensor)

        # Stage 3: Process controls through their respective modes
        prepended_embeddings: List[torch.Tensor] = []
        compact_attention_embeddings: List[torch.Tensor] = []

        for control_embedding, mode in control_embeddings_list:
            mode_val = mode.item()  # Convert to int only when needed for comparison
            if mode_val == 0:  # prepend
                prepended_embeddings.append(control_embedding.unsqueeze(1))  # (batch, 1, d_model)
            elif mode_val == 1:  # add
                # Reshape and add to HVO projection
                control_reshaped = control_embedding.view(-1, self.max_len, self.d_model)
                hvo_projection = hvo_projection + control_reshaped
            elif mode_val == 2:  # compact_attention
                compact_attention_embeddings.append(control_embedding)
            elif mode_val == 3:  # self_attention
                # Self-attention controls are processed but don't directly affect sequence
                # They could be used for compact_attention or other downstream processing
                # For now, we'll treat them as compact_attention
                compact_attention_embeddings.append(control_embedding)

        # Apply compact attention if we have compact attention controls
        if len(compact_attention_embeddings) > 0:
            hvo_projection = self.compact_attention(hvo_projection, compact_attention_embeddings)

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

    def forward(self, src: torch.Tensor):
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

    def forward(self, decoder_out: torch.Tensor):
        logits = self.Linear(decoder_out)
        return logits


class TensorBasedDecoderInput(torch.nn.Module):
    """ TorchScript-compatible decoder input using tensors instead of lists.
    Now supports both discrete (embedding) and continuous control values.

    Control modes: 0 = prepend, 1 = add, 2 = compact_attention, 3 = self_attention
    Control types: discrete (n_tokens is int) or continuous (n_tokens is None)
    """

    def __init__(self, max_len, latent_dim, d_model,
                 n_decoding_control_tokens,
                 decoding_control_modes):

        super(TensorBasedDecoderInput, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_controls = len(n_decoding_control_tokens)

        # Convert modes to integers and store as buffer (0=prepend, 1=add, 2=compact_attention, 3=self_attention)
        mode_mapping = {'prepend': 0, 'add': 1, 'compact_attention': 2, 'self_attention': 3}
        mode_ints = [mode_mapping[mode] for mode in decoding_control_modes]
        self.register_buffer('decoding_control_modes', torch.tensor(mode_ints, dtype=torch.long))

        # Store control types (True for discrete, False for continuous)
        control_types = [n_tokens is not None for n_tokens in n_decoding_control_tokens]
        self.register_buffer('control_is_discrete', torch.tensor(control_types, dtype=torch.bool))

        # Count prepended controls and self-attention controls
        self.n_prepended_controls = sum(1 for mode in decoding_control_modes if mode == 'prepend')
        self.n_self_attention_controls = sum(1 for mode in decoding_control_modes if mode == 'self_attention')

        # Create control processing layers (embeddings for discrete, linear layers for continuous)
        self.control_embeddings = torch.nn.ModuleList()
        self.control_projections = torch.nn.ModuleList()

        for i, (n_tokens, mode) in enumerate(zip(n_decoding_control_tokens, decoding_control_modes)):
            if n_tokens is not None:  # Discrete control
                if mode == 'prepend':
                    # Control embeddings output d_model dimensions for prepending
                    embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
                elif mode == 'add':
                    # Original behavior: embeddings output latent_dim for summation
                    embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=latent_dim)
                elif mode in ['compact_attention', 'self_attention']:
                    # These modes need d_model dimensions
                    embedding = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
                else:
                    raise ValueError(f"Unknown decoding control mode: {mode}")
                self.control_embeddings.append(embedding)
                self.control_projections.append(torch.nn.Identity())  # Placeholder for discrete controls
            else:  # Continuous control
                self.control_embeddings.append(torch.nn.Identity())  # Placeholder for continuous controls
                if mode == 'prepend':
                    # Project continuous value to d_model dimensions
                    projection = torch.nn.Linear(1, d_model)
                elif mode == 'add':
                    # Project continuous value to latent_dim for addition to latent_z
                    projection = torch.nn.Linear(1, latent_dim)
                elif mode in ['compact_attention', 'self_attention']:
                    # These modes need d_model dimensions
                    projection = torch.nn.Linear(1, d_model)
                else:
                    raise ValueError(f"Unknown decoding control mode: {mode}")
                self.control_projections.append(projection)

        # Add self-attention module for controls (always create it for TorchScript compatibility)
        if self.n_self_attention_controls > 0:
            self.control_self_attention = ControlSelfAttention(
                n_controls=self.n_self_attention_controls,
                d_model=d_model
            )
        else:
            # Create dummy module for TorchScript compatibility
            self.control_self_attention = ControlSelfAttention(
                n_controls=1,  # Minimum size
                d_model=d_model
            )

        # Add compact attention module
        self.compact_attention = CompactControlAttention(d_model)

        self.fc = torch.nn.Linear(int(latent_dim), int(max_len * d_model))
        self.reLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.compact_attention.init_weights(initrange)

        # Always initialize control_self_attention since it always exists now
        self.control_self_attention.init_weights(initrange)

        for i, (embedding, projection) in enumerate(zip(self.control_embeddings, self.control_projections)):
            if self.control_is_discrete[i]:
                # Initialize embedding
                if hasattr(embedding, 'weight'):
                    embedding.weight.data.uniform_(-initrange, initrange)
            else:
                # Initialize linear projection
                if hasattr(projection, 'weight'):
                    projection.weight.data.uniform_(-initrange, initrange)
                    projection.bias.data.zero_()

    @torch.jit.export
    def forward(self, latent_z: torch.Tensor, decoding_control_tokens: torch.Tensor):
        """
        applies the activation functions and reshapes the input tensor to fix dimensions with decoder

        :param latent_z: shape (batch, latent_dim)
        :param decoding_control_tokens: tensor of shape (batch, n_controls)
                                      - For discrete controls: integer token indices
                                      - For continuous controls: float values in [0, 1]

        :return:
                when all controls are 'add': (batch, max_len, d_model_dec)
                when some/all controls are 'prepend': (batch, max_len + num_prepended_controls, d_model_dec)
                when compact_attention: affects the sequence through attention
        """
        # Start with latent_z
        latent_modified = latent_z

        # Stage 1: Process individual controls into embeddings
        control_embeddings_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, (embedding, projection) in enumerate(zip(self.control_embeddings, self.control_projections)):
            control_value = decoding_control_tokens[:, i]  # (batch,)
            is_discrete = self.control_is_discrete[i]
            mode = self.decoding_control_modes[i]  # Keep as tensor, don't call .item()

            if is_discrete:
                # Discrete control: use embedding
                control_token = control_value.long()  # Ensure integer type
                control_embedding = embedding(control_token)
            else:
                # Continuous control: use linear projection
                continuous_value = control_value.float().unsqueeze(1)  # (batch, 1)
                control_embedding = projection(continuous_value)

            control_embeddings_list.append((control_embedding, mode))

        # Stage 2: Apply self-attention to self_attention controls
        if self.n_self_attention_controls > 0:
            # Collect self-attention control embeddings
            self_attention_embeddings: List[torch.Tensor] = []
            self_attention_indices: List[int] = []

            for i, (control_embedding, mode) in enumerate(control_embeddings_list):
                if mode == 3:  # self_attention mode
                    self_attention_embeddings.append(control_embedding)
                    self_attention_indices.append(i)

            if len(self_attention_embeddings) > 0:
                # Stack for self-attention: (batch, n_self_attn_controls, d_model)
                stacked_embeddings = torch.stack(self_attention_embeddings, dim=1)
                refined_embeddings = self.control_self_attention(stacked_embeddings)

                # Replace original embeddings with refined ones
                for idx in range(len(self_attention_indices)):
                    list_idx = self_attention_indices[idx]
                    mode_tensor = control_embeddings_list[list_idx][1]
                    control_embeddings_list[list_idx] = (refined_embeddings[:, idx, :], mode_tensor)

        # Stage 3: Process controls through their respective modes
        prepended_embeddings: List[torch.Tensor] = []
        compact_attention_embeddings: List[torch.Tensor] = []

        for control_embedding, mode in control_embeddings_list:
            mode_val = mode.item()  # Convert to int only when needed for comparison
            if mode_val == 0:  # prepend
                prepended_embeddings.append(control_embedding.unsqueeze(1))  # (batch, 1, d_model)
            elif mode_val == 1:  # add
                # For add mode, add to latent_z
                latent_modified = latent_modified + control_embedding
            elif mode_val == 2:  # compact_attention
                compact_attention_embeddings.append(control_embedding)
            elif mode_val == 3:  # self_attention
                # Self-attention controls can be used for compact_attention or other downstream processing
                # For now, we'll treat them as compact_attention after self-attention refinement
                compact_attention_embeddings.append(control_embedding)

        # Project modified latent and reshape to (batch, max_len, d_model)
        latent_projected = self.reLU(self.fc.forward(latent_modified)).view(-1, self.max_len, self.d_model)

        # Apply compact attention if we have compact attention controls
        if len(compact_attention_embeddings) > 0:
            latent_projected = self.compact_attention(latent_projected, compact_attention_embeddings)

        # Combine prepended controls with projected latent
        if len(prepended_embeddings) > 0:
            output = torch.cat(prepended_embeddings + [latent_projected], dim=1)
        else:
            output = latent_projected

        return output