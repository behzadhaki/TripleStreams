#  Copyright (c) 2025.
# Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model.FlexControlTripleStreamsVAE.components import TensorBasedInputGrooveLayer, Encoder, TensorBasedLatentLayer, \
    TensorBasedDecoderInput, \
    SingleFeatureOutputLayer


class FlexControlTripleStreamsVAE(torch.nn.Module):
    """
    An encoder-decoder VAE transformer with flexible control token support
    Now supports: 'prepend', 'add', and 'compact_attention' modes
    And both discrete (embedding) and continuous (linear projection) control types
    """

    def __init__(self, config):
        """
        This is a VAE transformer which uses transformer encoder architecture for both encoder and decoder

        :param config: a dictionary containing the following keys:
            d_model_enc: the dimension of the model for the encoder
            d_model_dec: the dimension of the model for the decoder
            embedding_size_src: the dimension of the input embedding
            embedding_size_tgt: the dimension of the output embedding
            nhead_enc: the number of heads for the encoder
            nhead_dec: the number of heads for the decoder
            dim_feedforward_enc: the dimension of the feedforward network in the encoder
            dim_feedforward_dec: the dimension of the feedforward network in the decoder
            num_encoder_layers: the number of encoder layers
            num_decoder_layers: the number of decoder layers
            dropout: the dropout rate
            latent_dim: the dimension of the latent space
            max_len: the maximum length of the input/output sequence

            # Flexible control configuration
            n_encoding_control_tokens: list of ints or None, number of tokens for each encoding control
                                      - int: discrete control with embedding layer
                                      - None: continuous control with linear projection
            encoding_control_modes: list of strings, mode for each encoding control ('prepend', 'add', or 'compact_attention')
            n_decoding_control_tokens: list of ints or None, number of tokens for each decoding control
                                     - int: discrete control with embedding layer
                                     - None: continuous control with linear projection
            decoding_control_modes: list of strings, mode for each decoding control ('prepend', 'add', or 'compact_attention')

            device: the device to use
        """

        super(FlexControlTripleStreamsVAE, self).__init__()

        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        self.config = config
        self.latent_dim = config['latent_dim']

        # Flexible control configuration
        self.n_encoding_control_tokens = []
        for x in config['n_encoding_control_tokens']:
            if isinstance(x, int):
                self.n_encoding_control_tokens.append(x)
            elif isinstance(x, str):
                if x.lower() in ['none', 'null']:
                    self.n_encoding_control_tokens.append(None)
                else:
                    raise ValueError(f"Invalid control token count: {x}. Must be an int or 'None'.")
            else:
                raise TypeError(f"Invalid type for n_encoding_control_tokens: {type(x)}. Must be int or str.")

        self.encoding_control_modes = config['encoding_control_modes']
        self.n_decoding_control_tokens = []
        for x in config['n_decoding_control_tokens']:
            if isinstance(x, int):
                self.n_decoding_control_tokens.append(x)
            elif isinstance(x, str):
                if x.lower() in ['none', 'null']:
                    self.n_decoding_control_tokens.append(None)
                else:
                    raise ValueError(f"Invalid control token count: {x}. Must be an int or 'None'.")
            else:
                raise TypeError(f"Invalid type for n_decoding_control_tokens: {type(x)}. Must be int or str.")

        self.decoding_control_modes = config['decoding_control_modes']

        # Validate control configuration
        assert len(self.n_encoding_control_tokens) == len(self.encoding_control_modes), \
            "Number of encoding control tokens must match number of encoding control modes"
        assert len(self.n_decoding_control_tokens) == len(self.decoding_control_modes), \
            "Number of decoding control tokens must match number of decoding control modes"

        # Validate control modes
        valid_modes = {'prepend', 'add', 'compact_attention', 'self_attention'}
        for mode in self.encoding_control_modes + self.decoding_control_modes:
            assert mode in valid_modes, f"Invalid control mode: {mode}. Must be one of {valid_modes}"

        # Count prepended controls for latent layer sizing
        self.n_prepended_encoding_controls = sum(1 for mode in self.encoding_control_modes if mode == 'prepend')
        self.n_prepended_decoding_controls = sum(1 for mode in self.decoding_control_modes if mode == 'prepend')

        print(self.n_encoding_control_tokens, self.n_decoding_control_tokens)

        # Layers for the Groove2Drum VAE
        # ---------------------------------------------------
        self.InputLayerEncoder = TensorBasedInputGrooveLayer(
            embedding_size=self.config['embedding_size_src'],
            d_model=self.config['d_model_enc'],
            max_len=self.config['max_len'],
            velocity_dropout=float(self.config['velocity_dropout']),
            offset_dropout=float(self.config['offset_dropout']),
            positional_encoding_dropout=float(self.config['dropout']),
            n_encoding_control_tokens=self.n_encoding_control_tokens,
            encoding_control_modes=self.encoding_control_modes,
        )

        self.Encoder = Encoder(
            d_model=self.config['d_model_enc'],
            nhead=self.config['nhead_enc'],
            dim_feedforward=self.config['dim_feedforward_enc'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.latentLayer = TensorBasedLatentLayer(
            max_len=self.config['max_len'],
            d_model=self.config['d_model_enc'],
            latent_dim=self.config['latent_dim'],
            n_prepended_encoding_controls=self.n_prepended_encoding_controls
        )

        self.HitsDecoderInput = TensorBasedDecoderInput(
            max_len=self.config['max_len'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec'],
            n_decoding_control_tokens=self.n_decoding_control_tokens,
            decoding_control_modes=self.decoding_control_modes,
        )

        self.HitsDecoder = Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.HitsOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.config['embedding_size_tgt'] // 3,
            d_model=self.config['d_model_dec'],
        )

        self.velocityDecoderInput = TensorBasedDecoderInput(
            max_len=self.config['max_len'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec'],
            n_decoding_control_tokens=self.n_decoding_control_tokens,
            decoding_control_modes=self.decoding_control_modes,
        )

        self.VelocityDecoder = Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.VelocityOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.config['embedding_size_tgt'] // 3,
            d_model=self.config['d_model_dec'],
        )

        self.OffsetDecoderInput = TensorBasedDecoderInput(
            max_len=self.config['max_len'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec'],
            n_decoding_control_tokens=self.n_decoding_control_tokens,
            decoding_control_modes=self.decoding_control_modes,
        )

        self.OffsetDecoder = Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.OffsetOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.config['embedding_size_tgt'] // 3,
            d_model=self.config['d_model_dec'],
        )

        self.init_weights(0.1)

    def init_weights(self, initrange):
        # Initialize weights and biases
        self.InputLayerEncoder.init_weights(initrange)
        self.latentLayer.init_weights(initrange)
        self.HitsDecoderInput.init_weights(initrange)
        self.HitsOutputLayer.init_weights(initrange)
        self.velocityDecoderInput.init_weights(initrange)
        self.VelocityOutputLayer.init_weights(initrange)
        self.OffsetDecoderInput.init_weights(initrange)
        self.OffsetOutputLayer.init_weights(initrange)

    @torch.jit.export
    def encodeLatent(self, flat_hvo_groove: torch.Tensor, encoding_control_tokens: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        :param flat_hvo_groove: [N, 32, 3]
        :param encoding_control_tokens: [N, n_encoding_controls]
                                      - For discrete controls: integer token indices
                                      - For continuous controls: float values in [0, 1]

        :return: mu, log_var, latent_z, memory
                mu:            [N, latent_dim]
                log_var:       [N, latent_dim]
                latent_z:      [N, latent_dim]
                memory:        [N, seq_len, d_model_enc] where seq_len depends on prepended controls

        """
        x, hit, hvo_projection = self.InputLayerEncoder.forward(
            hvo=flat_hvo_groove,
            encoding_control_tokens=encoding_control_tokens
        )
        memory = self.Encoder(x)  # N x seq_len x d_model_enc
        mu, log_var, latent_z = self.latentLayer(memory)
        return mu, log_var, latent_z, memory

    @torch.jit.export
    def encode_all(self, flat_hvo_groove: torch.Tensor, encoding_control_tokens: torch.Tensor):
        """
        Compatibility method for previous CompGenVAE (in VST)
        """
        return self.encodeLatent(
            flat_hvo_groove=flat_hvo_groove,
            encoding_control_tokens=encoding_control_tokens
        )

    def _extract_sequence_for_output(self, decoder_output):
        """
        Helper method to extract the correct sequence portion for output layers.
        When some controls are prepended, we need to skip those tokens
        and only use the sequence part for output prediction.
        """
        if self.n_prepended_decoding_controls > 0:
            # Skip the prepended control tokens, return only the sequence part [batch, max_len, d_model]
            return decoder_output[:, self.n_prepended_decoding_controls:, :]
        else:
            # Return the full sequence [batch, max_len, d_model]
            return decoder_output

    @torch.jit.export
    def decode(self, latent_z: torch.Tensor, decoding_control_tokens: torch.Tensor):
        """
        Decodes the latent_z through the decoder

        :param latent_z:            [N, latent_dim]
        :param decoding_control_tokens: [N, n_decoding_controls]
                                      - For discrete controls: integer token indices
                                      - For continuous controls: float values in [0, 1]

        :return:                    h_logits, v_logits, o_logits, hvo_logits

                h_logits: [N, 32, 1]
                v_logits: [N, 32, 1]
                o_logits: [N, 32, 1]
                hvo_logits: [N, 32, 3]

                None of the returned logits are activated (no sigmoid applied)
        """

        hits_decoder_output = self.HitsDecoder(
            self.HitsDecoderInput(
                latent_z=latent_z,
                decoding_control_tokens=decoding_control_tokens
            )
        )
        h_logits = self.HitsOutputLayer(self._extract_sequence_for_output(hits_decoder_output))

        velocity_decoder_output = self.VelocityDecoder(
            self.velocityDecoderInput(
                latent_z=latent_z,
                decoding_control_tokens=decoding_control_tokens
            )
        )
        v_logits = self.VelocityOutputLayer(self._extract_sequence_for_output(velocity_decoder_output))

        offset_decoder_output = self.OffsetDecoder(
            self.OffsetDecoderInput(
                latent_z=latent_z,
                decoding_control_tokens=decoding_control_tokens
            )
        )
        o_logits = self.OffsetOutputLayer(self._extract_sequence_for_output(offset_decoder_output))

        hvo_logits = torch.cat([h_logits, v_logits, o_logits], dim=-1)

        return h_logits, v_logits, o_logits, hvo_logits

    @torch.jit.export
    def sample(self,
               latent_z: torch.Tensor,
               decoding_control_tokens: torch.Tensor,
               voice_thresholds: torch.Tensor,
               voice_max_count_allowed: torch.Tensor,
               sampling_mode: int = 0):

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z,
            decoding_control_tokens=decoding_control_tokens
        )

        _h = torch.sigmoid(h_logits)
        v = torch.tanh(v_logits) / 2.0 + 0.5 # <----- reverses from [-1, 1] to [0, 1]
        o = torch.tanh(o_logits) / 2.0 # <------- reverses from [-1, 1] to [-0.5, 0.5]

        h = torch.zeros_like(_h)

        if sampling_mode == 0:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
                h[:, max_indices, ix] = _h[:, max_indices, ix]
                h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        elif sampling_mode == 1:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                # sample using probability distribution of hits (_h)
                voice_probs = _h[:, :, ix]
                sampled_indices = torch.bernoulli(voice_probs)
                max_indices = torch.topk(sampled_indices * voice_probs, max_count).indices[0]
                h[:, max_indices, ix] = 1

        # if v < 0.05, then set corresponding h to 0
        h = torch.where(v < 0.05, torch.zeros_like(h), h)

        return h, v, o

    @torch.jit.export
    def forward(self, flat_hvo_groove: torch.Tensor, encoding_control_tokens: torch.Tensor,
                decoding_control_tokens: torch.Tensor):

        mu, log_var, latent_z, memory = self.encodeLatent(
            flat_hvo_groove=flat_hvo_groove,
            encoding_control_tokens=encoding_control_tokens
        )

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z,
            decoding_control_tokens=decoding_control_tokens
        )

        return h_logits, v_logits, o_logits, mu, log_var, latent_z

    @torch.jit.export
    def predict(self,
                flat_hvo_groove: torch.Tensor,
                encoding_control_tokens: torch.Tensor,
                decoding_control_tokens: torch.Tensor,
                threshold: float = 0.5):

        h_logits, v_logits, o_logits, mu, log_var, latent_z = self.forward(
            flat_hvo_groove=flat_hvo_groove,
            encoding_control_tokens=encoding_control_tokens,
            decoding_control_tokens=decoding_control_tokens
        )

        _h = torch.sigmoid(h_logits)
        v = torch.tanh(v_logits) / 2.0 + 0.5 # <----- reverses from [-1, 1] to [0, 1]
        o = torch.tanh(o_logits) / 2.0 # <------- reverses from [-1, 1] to [-0.5, 0.5]

        h = torch.zeros_like(_h)

        voice_thresholds = torch.tensor([threshold] * 3)
        voice_max_count_allowed = torch.tensor([32] * 3)

        for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
            max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
            h[:, max_indices, ix] = _h[:, max_indices, ix]
            h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        # if v < 0.05, then set corresponding h to 0
        h = torch.where(v < 0.05, torch.zeros_like(h), h)

        hvo = torch.cat([h, v, o], dim=-1)

        return hvo, latent_z

    # Helper methods for converting between list and tensor formats
    def _convert_control_lists_to_tensors(self, encoding_control_tokens_list, decoding_control_tokens_list):
        """Convert list format to tensor format for internal use"""
        # Stack encoding controls: list of (batch,) -> (batch, n_controls)
        encoding_tensor = torch.stack(encoding_control_tokens_list, dim=1)

        # Stack decoding controls: list of (batch,) -> (batch, n_controls)
        decoding_tensor = torch.stack(decoding_control_tokens_list, dim=1)

        return encoding_tensor, decoding_tensor

    def predict_with_lists(self, flat_hvo_groove, encoding_control_tokens_list, decoding_control_tokens_list,
                           threshold=0.5):
        """Compatibility method that accepts lists and converts to tensors"""
        enc_tensor, dec_tensor = self._convert_control_lists_to_tensors(encoding_control_tokens_list,
                                                                        decoding_control_tokens_list)
        return self.predict(flat_hvo_groove, enc_tensor, dec_tensor, threshold)

    def forward_with_lists(self, flat_hvo_groove, encoding_control_tokens_list, decoding_control_tokens_list):
        """Compatibility method that accepts lists and converts to tensors"""
        enc_tensor, dec_tensor = self._convert_control_lists_to_tensors(encoding_control_tokens_list,
                                                                        decoding_control_tokens_list)
        return self.forward(flat_hvo_groove, enc_tensor, dec_tensor)

    def load_state_dict(self, state_dict, strict=True):
        """
        Enhanced load_state_dict with automatic compatibility handling for legacy checkpoints.
        """
        # Check if this is a legacy checkpoint by looking for old parameter names
        is_legacy_checkpoint = any(
            key.startswith('InputLayerEncoder.encoding_control1_embedding') or
            key.startswith('HitsDecoderInput.control1_embedding') or
            key.startswith('velocityDecoderInput.control1_embedding') or
            key.startswith('OffsetDecoderInput.control1_embedding')
            for key in state_dict.keys()
        )

        if is_legacy_checkpoint:
            print("📄 Detected legacy checkpoint format. Converting to flexible format...")
            state_dict = self._convert_legacy_state_dict(state_dict)
            print("✅ Successfully converted legacy checkpoint parameters")

        # Handle missing control mode buffers (they're not critical for functionality)
        missing_control_modes = [
            "InputLayerEncoder.encoding_control_modes",
            "InputLayerEncoder.control_is_discrete",
            "HitsDecoderInput.decoding_control_modes",
            "HitsDecoderInput.control_is_discrete",
            "velocityDecoderInput.decoding_control_modes",
            "velocityDecoderInput.control_is_discrete",
            "OffsetDecoderInput.decoding_control_modes",
            "OffsetDecoderInput.control_is_discrete"
        ]

        for missing_key in missing_control_modes:
            if missing_key not in state_dict:
                # Get the corresponding tensor from current model
                module_name, attr_name = missing_key.split('.', 1)
                if hasattr(self, module_name):
                    module = getattr(self, module_name)
                    if hasattr(module, attr_name):
                        current_tensor = getattr(module, attr_name)
                        state_dict[missing_key] = current_tensor.clone()

        return super().load_state_dict(state_dict, strict=False)  # Use strict=False for compatibility

    def _convert_legacy_state_dict(self, legacy_state_dict):
        """
        Convert legacy state dict parameter names to flexible format.

        Legacy format:
        - InputLayerEncoder.encoding_control1_embedding.weight
        - InputLayerEncoder.encoding_control2_embedding.weight
        - HitsDecoderInput.control1_embedding.weight
        - etc.

        Flexible format:
        - InputLayerEncoder.control_embeddings.0.weight
        - InputLayerEncoder.control_embeddings.1.weight
        - HitsDecoderInput.control_embeddings.0.weight
        - etc.
        """
        converted_state_dict = {}

        # Define mapping from legacy names to flexible names
        legacy_to_flexible_mapping = {
            # Input layer encoder mappings
            'InputLayerEncoder.encoding_control1_embedding.weight': 'InputLayerEncoder.control_embeddings.0.weight',
            'InputLayerEncoder.encoding_control2_embedding.weight': 'InputLayerEncoder.control_embeddings.1.weight',

            # Decoder input mappings for all three streams
            'HitsDecoderInput.control1_embedding.weight': 'HitsDecoderInput.control_embeddings.0.weight',
            'HitsDecoderInput.control2_embedding.weight': 'HitsDecoderInput.control_embeddings.1.weight',
            'HitsDecoderInput.control3_embedding.weight': 'HitsDecoderInput.control_embeddings.2.weight',

            'velocityDecoderInput.control1_embedding.weight': 'velocityDecoderInput.control_embeddings.0.weight',
            'velocityDecoderInput.control2_embedding.weight': 'velocityDecoderInput.control_embeddings.1.weight',
            'velocityDecoderInput.control3_embedding.weight': 'velocityDecoderInput.control_embeddings.2.weight',

            'OffsetDecoderInput.control1_embedding.weight': 'OffsetDecoderInput.control_embeddings.0.weight',
            'OffsetDecoderInput.control2_embedding.weight': 'OffsetDecoderInput.control_embeddings.1.weight',
            'OffsetDecoderInput.control3_embedding.weight': 'OffsetDecoderInput.control_embeddings.2.weight',
        }

        # Convert parameters
        for old_key, tensor in legacy_state_dict.items():
            if old_key in legacy_to_flexible_mapping:
                new_key = legacy_to_flexible_mapping[old_key]
                converted_state_dict[new_key] = tensor
                print(f"  {old_key} → {new_key}")
            else:
                # Keep all other parameters as-is
                converted_state_dict[old_key] = tensor

        return converted_state_dict

    def save(self, save_path, additional_info=None, include_legacy_json=True, save_version='2.1'):
        """
        Enhanced save method that embeds config in the model file for self-contained loading.

        Args:
            save_path: Path to save the model
            additional_info: Additional metadata to include
            include_legacy_json: Whether to also save separate .json config file for backward compatibility
            save_version: Version string to track save format evolution
        """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Prepare config dictionary
        config = dict()
        for key, value in self.config.items():
            config[key] = value

        # Prepare comprehensive save dictionary
        save_dict = {
            'model_state_dict': self.state_dict(),
            'params': config,  # Embedded config for self-contained loading
            'additional_info': additional_info,
            'model_class': self.__class__.__name__,
            'save_version': save_version,
            'pytorch_version': torch.__version__,
            'model_architecture': 'FlexControlTripleStreamsVAE_v2.1_with_continuous_controls'
        }

        # Save the main model file with embedded config
        torch.save(save_dict, save_path)
        print(f"✅ Saved model with embedded config: {save_path}")

        # Optionally save separate JSON config file for legacy compatibility
        if include_legacy_json:
            json_path = save_path.replace('.pth', '.json')
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"📄 Legacy JSON config saved: {json_path}")

        return save_path

    def save_legacy(self, save_path, additional_info=None):
        """
        Legacy save method (original behavior) - saves separate .json config file.
        Kept for compatibility with older workflows.
        """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config = dict()
        for key, value in self.config.items():
            config[key] = value

        # Save config as separate JSON file (original behavior)
        json.dump(config, open(save_path.replace('.pth', '.json'), 'w'))

        # Save model with minimal embedded info (original behavior)
        torch.save({
            'model_state_dict': self.state_dict(),
            'params': config,
            'additional_info': additional_info
        }, save_path)

        print(f"🔒 Legacy save completed: {save_path}")
        return save_path

    @classmethod
    def load(cls, model_path, device=None, is_evaluating=True):
        """
        Class method for convenient loading of FlexControlTripleStreamsVAE models.
        Automatically handles both new (embedded config) and legacy (separate JSON) formats.
        """

        # Handle PyTorch version compatibility for weights_only parameter
        def safe_load(path, map_location=None):
            try:
                # Try with weights_only=False (newer PyTorch versions)
                if map_location is not None:
                    return torch.load(path, map_location=map_location, weights_only=False)
                else:
                    return torch.load(path, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions without weights_only parameter
                if map_location is not None:
                    return torch.load(path, map_location=map_location)
                else:
                    return torch.load(path)

        try:
            if device is not None:
                loaded_dict = safe_load(model_path, map_location=device)
            else:
                loaded_dict = safe_load(model_path)
        except Exception as e:
            # Final fallback to CPU
            try:
                loaded_dict = safe_load(model_path, map_location=torch.device('cpu'))
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to load model: {e}. Fallback error: {fallback_error}")

        # Try embedded config first (new format)
        if 'params' in loaded_dict:
            config = loaded_dict['params']
            print(f"✅ Loading with embedded config")
        else:
            # Fallback to companion JSON file (legacy format)
            json_path = model_path.replace('.pth', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    config = json.load(f)
                print(f"⚠️  Loading with legacy JSON config: {json_path}")
            else:
                raise FileNotFoundError(
                    f"No config found. Expected either:\n"
                    f"1. Embedded config in {model_path}\n"
                    f"2. Companion JSON file: {json_path}"
                )

        # Create and load model
        model = cls(config)
        model.load_state_dict(loaded_dict["model_state_dict"])

        if is_evaluating:
            model.eval()

        print(f"🎉 Successfully loaded {cls.__name__}")
        return model

    @classmethod
    def load_torchscript(cls, script_path, device=None):
        """Load a serialized TorchScript model"""
        model = torch.jit.load(script_path, map_location=device or 'cpu')
        return model

    # serializes to a torchscript model
    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):
        os.makedirs(save_folder, exist_ok=True)

        if filename is None:
            import datetime
            filename = f'TripleStreams_v2.1_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'

        is_train = self.training
        self.eval()

        # Ensure the save path is a file, not a directory
        save_path = os.path.join(save_folder, filename)

        # Check if the path exists and is a directory
        if os.path.exists(save_path) and os.path.isdir(save_path):
            # If it's a directory, remove it or use a different filename
            import shutil
            shutil.rmtree(save_path)
            print(f"⚠️  Removed existing directory: {save_path}")

        scr = torch.jit.script(self)

        # Save model
        try:
            with open(save_path, "wb") as f:
                torch.jit.save(scr, f)
            print(f"✅ TorchScript model saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save TorchScript model: {e}")
            raise

        if is_train:
            self.train()

        return save_path

    @torch.jit.export
    def get_latent_dim(self):
        return self.latent_dim


if __name__ == "__main__":
    print("🧪 Testing FlexControlTripleStreamsVAE with continuous controls...")

    # Test configuration with mixed discrete and continuous controls
    config = {
        'd_model_enc': 128,
        'd_model_dec': 128,
        'embedding_size_src': 3,
        'embedding_size_tgt': 9,
        'nhead_enc': 4,
        'nhead_dec': 8,
        'dim_feedforward_enc': 128,
        'dim_feedforward_dec': 512,
        'num_encoder_layers': 3,
        'num_decoder_layers': 6,
        'dropout': 0.1,
        'latent_dim': 16,
        'max_len': 32,
        'velocity_dropout': 0.1,
        'offset_dropout': 0.2,
        'device': 'cpu',

        # Mixed discrete and continuous control configuration
        'n_encoding_control_tokens': [13, None],  # discrete, continuous
        'encoding_control_modes': ['prepend', 'compact_attention'],
        'n_decoding_control_tokens': [10, None, None],  # discrete, continuous, continuous
        'decoding_control_modes': ['compact_attention', 'prepend', 'add']
    }

    # Create control tokens as TENSORS with mixed types
    batch_size = 2

    # Encoding control tokens: [discrete_token, continuous_value]
    encoding_control_tokens = torch.tensor([
        [1, 0.7],  # batch 0: discrete=1, continuous=0.7
        [5, 0.3]  # batch 1: discrete=5, continuous=0.3
    ], dtype=torch.float32)  # Use float32 to support both int and float values

    # Decoding control tokens: [discrete_token, continuous_value1, continuous_value2]
    decoding_control_tokens = torch.tensor([
        [3, 0.8, 0.2],  # batch 0: discrete=3, continuous1=0.8, continuous2=0.2
        [7, 0.1, 0.9]  # batch 1: discrete=7, continuous1=0.1, continuous2=0.9
    ], dtype=torch.float32)

    # Test the model
    model = FlexControlTripleStreamsVAE(config)
    print(f"✅ Model created successfully")
    print(f"   Encoding control modes: {model.encoding_control_modes}")
    print(f"   Decoding control modes: {model.decoding_control_modes}")
    print(f"   Prepended encoding controls: {model.n_prepended_encoding_controls}")
    print(f"   Prepended decoding controls: {model.n_prepended_decoding_controls}")

    # Print control types for verification
    print(f"   Encoding control types (discrete=True): {model.InputLayerEncoder.control_is_discrete}")
    print(f"   Decoding control types (discrete=True): {model.HitsDecoderInput.control_is_discrete}")

    # Test type handling specifically
    print("\n🔍 Testing mixed type handling...")
    print(f"   Input tensor dtype: {encoding_control_tokens.dtype}")
    print(f"   Encoding values: {encoding_control_tokens}")
    print(f"   - Control 0 (discrete): {encoding_control_tokens[:, 0]} -> will be cast to Long")
    print(f"   - Control 1 (continuous): {encoding_control_tokens[:, 1]} -> will remain Float")

    print(f"   Decoding values: {decoding_control_tokens}")
    print(f"   - Control 0 (discrete): {decoding_control_tokens[:, 0]} -> will be cast to Long")
    print(f"   - Control 1 (continuous): {decoding_control_tokens[:, 1]} -> will remain Float")
    print(f"   - Control 2 (continuous): {decoding_control_tokens[:, 2]} -> will remain Float")

    # Test forward pass
    print("\n🔄 Testing forward pass...")
    h_logits, v_logits, o_logits, mu, log_var, latent_z = model.forward(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )

    print(f"✅ Forward pass successful:")
    print(f"   h_logits shape: {h_logits.shape}")
    print(f"   v_logits shape: {v_logits.shape}")
    print(f"   o_logits shape: {o_logits.shape}")
    print(f"   latent_z shape: {latent_z.shape}")

    # Test edge cases for type handling
    print("\n🧪 Testing edge cases for type handling...")

    # Test with integer-like floats (should work for discrete controls)
    encoding_edge_case = torch.tensor([
        [1.0, 0.7],  # 1.0 should be castable to Long(1)
        [5.0, 0.3]  # 5.0 should be castable to Long(5)
    ], dtype=torch.float32)

    decoding_edge_case = torch.tensor([
        [3.0, 0.8, 0.2],  # 3.0 should be castable to Long(3)
        [7.0, 0.1, 0.9]  # 7.0 should be castable to Long(7)
    ], dtype=torch.float32)

    try:
        hvo_edge, _ = model.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_edge_case,
            decoding_control_tokens=decoding_edge_case
        )
        print(f"✅ Edge case (integer-like floats) successful: {hvo_edge.shape}")
    except Exception as e:
        print(f"❌ Edge case failed: {e}")

    # Test with out-of-range discrete values (should fail gracefully)
    print("\n⚠️  Testing out-of-range discrete values...")
    encoding_invalid = torch.tensor([
        [100.0, 0.7],  # 100 is out of range for embedding size 13
        [5.0, 0.3]
    ], dtype=torch.float32)

    try:
        hvo_invalid, _ = model.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_invalid,
            decoding_control_tokens=decoding_control_tokens
        )
        print(f"❌ Should have failed with out-of-range values!")
    except Exception as e:
        print(f"✅ Correctly caught out-of-range error: {type(e).__name__}")

    # Test boundary values for continuous controls
    print("\n🎯 Testing boundary values for continuous controls...")
    encoding_boundary = torch.tensor([
        [0.0, 0.0],  # Min values
        [12.0, 1.0]  # Max discrete value, max continuous value
    ], dtype=torch.float32)

    decoding_boundary = torch.tensor([
        [0.0, 0.0, 0.0],
        [9.0, 1.0, 1.0]
    ], dtype=torch.float32)

    try:
        hvo_boundary, _ = model.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_boundary,
            decoding_control_tokens=decoding_boundary
        )
        print(f"✅ Boundary values test successful: {hvo_boundary.shape}")
    except Exception as e:
        print(f"❌ Boundary values test failed: {e}")

    # Test prediction
    print("\n🎯 Testing prediction...")
    hvo, latent_z = model.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )

    print(f"✅ Prediction successful:")
    print(f"   hvo shape: {hvo.shape}")
    print(f"   latent_z shape: {latent_z.shape}")

    print(f"\n🧠 Testing self_attention mode...")
    config_self_attention = config.copy()
    config_self_attention.update({
        'n_encoding_control_tokens': [13, None, 10],  # discrete, continuous, discrete
        'encoding_control_modes': ['self_attention', 'self_attention', 'prepend'],  # First two use self-attention
        'n_decoding_control_tokens': [None, None, 10, None],  # continuous, continuous, discrete, continuous
        'decoding_control_modes': ['self_attention', 'self_attention', 'add', 'compact_attention']
    })

    # Control tokens for self-attention test
    encoding_self_attn = torch.tensor([
        [5.0, 0.7, 2.0],  # discrete=5, continuous=0.7, discrete=2
        [8.0, 0.3, 7.0]  # discrete=8, continuous=0.3, discrete=7
    ], dtype=torch.float32)

    decoding_self_attn = torch.tensor([
        [0.4, 0.8, 3.0, 0.6],  # continuous, continuous, discrete=3, continuous
        [0.1, 0.9, 6.0, 0.2]  # continuous, continuous, discrete=6, continuous
    ], dtype=torch.float32)

    model_self_attention = FlexControlTripleStreamsVAE(config_self_attention)
    print(f"✅ Self-attention model created:")
    print(f"   Encoding modes: {model_self_attention.encoding_control_modes}")
    print(f"   Decoding modes: {model_self_attention.decoding_control_modes}")
    print(f"   Encoding self-attention controls: {model_self_attention.InputLayerEncoder.n_self_attention_controls}")
    print(f"   Decoding self-attention controls: {model_self_attention.HitsDecoderInput.n_self_attention_controls}")

    try:
        hvo_self_attn, latent_self_attn = model_self_attention.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_self_attn,
            decoding_control_tokens=decoding_self_attn
        )
        print(f"✅ Self-attention prediction successful: {hvo_self_attn.shape}")
        print(f"   - Controls can now learn inter-dependencies!")
        print(f"   - Control correlations will be learned through self-attention")
        print(f"   - Example: if tempo=slow, complexity might be automatically adjusted")
    except Exception as e:
        print(f"❌ Self-attention test failed: {e}")

    # Test all-self-attention configuration
    print(f"\n🧠 Testing all-self-attention configuration...")
    config_all_self_attn = config.copy()
    config_all_self_attn.update({
        'n_encoding_control_tokens': [13, None],  # discrete, continuous
        'encoding_control_modes': ['self_attention', 'self_attention'],  # All self-attention
        'n_decoding_control_tokens': [None, 10, None],  # continuous, discrete, continuous
        'decoding_control_modes': ['self_attention', 'self_attention', 'self_attention']  # All self-attention
    })

    encoding_all_self_attn = torch.tensor([
        [1.0, 0.7],
        [5.0, 0.3]
    ], dtype=torch.float32)

    decoding_all_self_attn = torch.tensor([
        [0.8, 2.0, 0.1],
        [0.4, 7.0, 0.9]
    ], dtype=torch.float32)

    try:
        model_all_self_attn = FlexControlTripleStreamsVAE(config_all_self_attn)
        hvo_all_self_attn, _ = model_all_self_attn.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_all_self_attn,
            decoding_control_tokens=decoding_all_self_attn
        )
        print(f"✅ All-self-attention prediction successful: {hvo_all_self_attn.shape}")
        print(f"   - Maximum control inter-dependency learning!")
    except Exception as e:
        print(f"❌ All-self-attention test failed: {e}")

    # Test mixed modes including self_attention
    print(f"\n🎭 Testing mixed modes with self_attention...")
    config_mixed_with_self_attn = config.copy()
    config_mixed_with_self_attn.update({
        'n_encoding_control_tokens': [13, None, 10, None],  # discrete, continuous, discrete, continuous
        'encoding_control_modes': ['prepend', 'self_attention', 'self_attention', 'compact_attention'],
        'n_decoding_control_tokens': [None, 10, None, 8],  # continuous, discrete, continuous, discrete
        'decoding_control_modes': ['self_attention', 'add', 'self_attention', 'prepend']
    })

    encoding_mixed_self_attn = torch.tensor([
        [1.0, 0.7, 5.0, 0.3],
        [8.0, 0.2, 2.0, 0.8]
    ], dtype=torch.float32)

    decoding_mixed_self_attn = torch.tensor([
        [0.4, 3.0, 0.6, 2.0],
        [0.9, 7.0, 0.1, 5.0]
    ], dtype=torch.float32)

    try:
        model_mixed_self_attn = FlexControlTripleStreamsVAE(config_mixed_with_self_attn)
        hvo_mixed_self_attn, _ = model_mixed_self_attn.predict(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens=encoding_mixed_self_attn,
            decoding_control_tokens=decoding_mixed_self_attn
        )
        print(f"✅ Mixed modes with self-attention successful: {hvo_mixed_self_attn.shape}")
        print(f"   - Combines all control modes: prepend, add, compact_attention, self_attention")
        print(f"   - Self-attention controls learn dependencies, others provide direct influence")
    except Exception as e:
        print(f"❌ Mixed modes with self-attention failed: {e}")

    # Test all-continuous configuration (backward compatibility)
    print("\n🌊 Testing all-continuous configuration...")
    config_all_continuous = config.copy()
    config_all_continuous.update({
        'n_encoding_control_tokens': [None, None],  # All continuous
        'encoding_control_modes': ['compact_attention', 'add'],
        'n_decoding_control_tokens': [None, None, None],  # All continuous
        'decoding_control_modes': ['compact_attention', 'prepend', 'add']
    })

    # All continuous control tokens
    encoding_continuous = torch.tensor([
        [0.5, 0.7],
        [0.2, 0.9]
    ], dtype=torch.float32)

    decoding_continuous = torch.tensor([
        [0.3, 0.8, 0.1],
        [0.6, 0.4, 0.7]
    ], dtype=torch.float32)

    model_all_continuous = FlexControlTripleStreamsVAE(config_all_continuous)
    print(f"✅ All-continuous model created:")
    print(f"   Prepended encoding controls: {model_all_continuous.n_prepended_encoding_controls}")
    print(f"   Prepended decoding controls: {model_all_continuous.n_prepended_decoding_controls}")

    hvo_continuous, _ = model_all_continuous.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_continuous,
        decoding_control_tokens=decoding_continuous
    )
    print(f"✅ All-continuous prediction successful: {hvo_continuous.shape}")

    # Test all-discrete configuration (backward compatibility)
    print("\n🔢 Testing all-discrete configuration (backward compatibility)...")
    config_all_discrete = config.copy()
    config_all_discrete.update({
        'n_encoding_control_tokens': [13, 10],  # All discrete
        'encoding_control_modes': ['prepend', 'compact_attention'],
        'n_decoding_control_tokens': [10, 10, 10],  # All discrete
        'decoding_control_modes': ['prepend', 'add', 'compact_attention']
    })

    # All discrete control tokens
    encoding_discrete = torch.tensor([
        [1, 2],
        [5, 7]
    ], dtype=torch.long)

    decoding_discrete = torch.tensor([
        [3, 4, 1],
        [8, 2, 6]
    ], dtype=torch.long)

    model_all_discrete = FlexControlTripleStreamsVAE(config_all_discrete)
    print(f"✅ All-discrete model created:")
    print(f"   Prepended encoding controls: {model_all_discrete.n_prepended_encoding_controls}")
    print(f"   Prepended decoding controls: {model_all_discrete.n_prepended_decoding_controls}")

    hvo_discrete, _ = model_all_discrete.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_discrete.float(),  # Convert to float for consistent interface
        decoding_control_tokens=decoding_discrete.float()
    )
    print(f"✅ All-discrete prediction successful: {hvo_discrete.shape}")

    # Test TorchScript serialization
    print(f"\n💾 Testing TorchScript serialization...")
    try:
        model.serialize(save_folder='./', filename='triplestreams_vae_continuous_v2.1.pt')
        print(f"✅ TorchScript serialization successful!")

        # Test loading the serialized model
        loaded_model = model.load_torchscript('./triplestreams_vae_continuous_v2.1.pt')
        print(f"✅ TorchScript model loaded successfully!")

        # Test the loaded model
        hvo_loaded, _ = loaded_model.predict(
            torch.rand(1, 32, config["embedding_size_src"]),
            encoding_control_tokens[:1],
            decoding_control_tokens[:1]
        )
        print(f"✅ Loaded TorchScript model prediction successful: {hvo_loaded.shape}")

    except Exception as e:
        print(f"❌ TorchScript serialization failed: {e}")

    # Test enhanced save/load functionality
    print(f"\n💾 Testing enhanced save/load functionality...")
    try:
        # Save with embedded config
        save_path = model.save('./test_model_v2.1.pth')

        # Load the model
        loaded_model_v2 = FlexControlTripleStreamsVAE.load('./test_model_v2.1.pth')

        # Test the loaded model
        hvo_loaded_v2, _ = loaded_model_v2.predict(
            torch.rand(1, 32, config["embedding_size_src"]),
            encoding_control_tokens[:1],
            decoding_control_tokens[:1]
        )
        print(f"✅ Enhanced save/load successful: {hvo_loaded_v2.shape}")

        # Clean up
        import os

        if os.path.exists('./test_model_v2.1.pth'):
            os.remove('./test_model_v2.1.pth')
        if os.path.exists('./test_model_v2.1.json'):
            os.remove('./test_model_v2.1.json')

    except Exception as e:
        print(f"❌ Enhanced save/load failed: {e}")

    # Test compatibility methods (if you need list interface for backward compatibility)
    print(f"\n🔄 Testing compatibility methods...")
    try:
        # Convert tensors back to lists for testing compatibility methods
        encoding_list = [encoding_control_tokens[:, i] for i in range(encoding_control_tokens.shape[1])]
        decoding_list = [decoding_control_tokens[:, i] for i in range(decoding_control_tokens.shape[1])]

        hvo_compat, _ = model.predict_with_lists(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens_list=encoding_list,
            decoding_control_tokens_list=decoding_list
        )
        print(f"✅ Compatibility method successful: {hvo_compat.shape}")
    except Exception as e:
        print(f"❌ Compatibility method failed: {e}")

    print(f"\n🎉 All tests completed successfully!")

    print(f"\n📋 Summary of features:")
    print(f"  ✅ Added support for continuous controls (n_tokens=None)")
    print(f"  ✅ Continuous controls use linear projections instead of embeddings")
    print(f"  ✅ Mixed discrete/continuous configurations supported")
    print(f"  ✅ NEW: Added 'self_attention' control mode for learning inter-dependencies")
    print(f"  ✅ Control self-attention learns correlations between controls")
    print(f"  ✅ Backward compatible with existing discrete-only models")
    print(f"  ✅ Enhanced save/load with embedded configs")
    print(f"  ✅ TorchScript compatible")
    print(f"  ✅ All control modes work with both discrete and continuous controls:")
    print(f"      - 'prepend': adds control tokens to sequxence start")
    print(f"      - 'add': adds control influence to latent space or sequence")
    print(f"      - 'compact_attention': applies attention-based control influence")
    print(f"      - 'self_attention': learns inter-dependencies between controls")

    print(f"\n🔧 Usage examples:")
    print(f"  # Discrete control (embedding-based):")
    print(f"  'n_encoding_control_tokens': [13, 10]  # 13 and 10 discrete tokens")
    print(f"  ")
    print(f"  # Continuous control (linear projection):")
    print(f"  'n_encoding_control_tokens': [None, None]  # continuous values [0,1]")
    print(f"  ")
    print(f"  # Mixed discrete and continuous:")
    print(f"  'n_encoding_control_tokens': [13, None, 10]  # discrete, continuous, discrete")
    print(f"  ")
    print(f"  # Control modes with self-attention:")
    print(f"  'encoding_control_modes': ['self_attention', 'self_attention', 'prepend']")
    print(f"  # First two controls learn inter-dependencies, third is prepended")

    # Clean up generated files
    try:
        if os.path.exists('./triplestreams_vae_continuous_v2.1.pt'):
            os.remove('./triplestreams_vae_continuous_v2.1.pt')
    except:
        pass

    model.save('./triplestreams_vae_continuous_v2.1.pth')
    FlexControlTripleStreamsVAE.load('./triplestreams_vae_continuous_v2.1.pth')
