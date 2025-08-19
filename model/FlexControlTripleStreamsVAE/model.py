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
            n_encoding_control_tokens: list of ints, number of tokens for each encoding control
            encoding_control_modes: list of strings, mode for each encoding control ('prepend', 'add', or 'compact_attention')
            n_decoding_control_tokens: list of ints, number of tokens for each decoding control
            decoding_control_modes: list of strings, mode for each decoding control ('prepend', 'add', or 'compact_attention')

            device: the device to use
        """

        super(FlexControlTripleStreamsVAE, self).__init__()

        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        self.config = config
        self.latent_dim = config['latent_dim']

        # Flexible control configuration
        self.n_encoding_control_tokens = config['n_encoding_control_tokens']
        self.encoding_control_modes = config['encoding_control_modes']
        self.n_decoding_control_tokens = config['n_decoding_control_tokens']
        self.decoding_control_modes = config['decoding_control_modes']

        # Validate control configuration
        assert len(self.n_encoding_control_tokens) == len(self.encoding_control_modes), \
            "Number of encoding control tokens must match number of encoding control modes"
        assert len(self.n_decoding_control_tokens) == len(self.decoding_control_modes), \
            "Number of decoding control tokens must match number of decoding control modes"

        # Validate control modes
        valid_modes = {'prepend', 'add', 'compact_attention'}
        for mode in self.encoding_control_modes + self.decoding_control_modes:
            assert mode in valid_modes, f"Invalid control mode: {mode}. Must be one of {valid_modes}"

        # Count prepended controls for latent layer sizing
        self.n_prepended_encoding_controls = sum(1 for mode in self.encoding_control_modes if mode == 'prepend')
        self.n_prepended_decoding_controls = sum(1 for mode in self.decoding_control_modes if mode == 'prepend')

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
        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

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
        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

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
            "HitsDecoderInput.decoding_control_modes",
            "velocityDecoderInput.decoding_control_modes",
            "OffsetDecoderInput.decoding_control_modes"
        ]

        for missing_key in missing_control_modes:
            if missing_key not in state_dict:
                # Get the corresponding tensor from current model
                if hasattr(self, missing_key.split('.')[0]):
                    module = getattr(self, missing_key.split('.')[0])
                    if hasattr(module, missing_key.split('.')[1]):
                        current_tensor = getattr(module, missing_key.split('.')[1])
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

    def save(self, save_path, additional_info=None, include_legacy_json=True, save_version='2.0'):
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
            'model_architecture': 'FlexControlTripleStreamsVAE'
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

        print(f"📁 Legacy save completed: {save_path}")
        return save_path

    @classmethod
    def load(cls, model_path, device=None, is_evaluating=True):
        """
        Class method for convenient loading of FlexControlTripleStreamsVAE models.
        Automatically handles both new (embedded config) and legacy (separate JSON) formats.
        """
        try:
            if device is not None:
                loaded_dict = torch.load(model_path, map_location=device, weights_only=False)
            else:
                loaded_dict = torch.load(model_path, weights_only=False)
        except:
            loaded_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

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
            filename = f'TripleStreams_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        is_train = self.training
        self.eval()
        save_path = os.path.join(save_folder, filename)

        scr = torch.jit.script(self)
        # save model
        with open(save_path, "wb") as f:
            torch.jit.save(scr, f)

        if is_train:
            self.train()

    @torch.jit.export
    def get_latent_dim(self):
        return self.latent_dim


if __name__ == "__main__":
    # Test configuration with flexible controls including compact_attention
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

        # Flexible control configuration with compact_attention
        'n_encoding_control_tokens': [13, 10],
        'encoding_control_modes': ['prepend', 'compact_attention'],
        'n_decoding_control_tokens': [10, 10, 10],
        'decoding_control_modes': ['compact_attention', 'prepend', 'add']
    }

    # Create control tokens as TENSORS (not lists)
    batch_size = 1
    n_encoding_controls = len(config['n_encoding_control_tokens'])
    n_decoding_controls = len(config['n_decoding_control_tokens'])

    # Create encoding control tokens tensor: (batch, n_encoding_controls)
    encoding_control_tokens = torch.tensor([
        [1, 2]  # [first_control, second_control] for batch
    ], dtype=torch.long)  # Shape: (1, 2)

    # Create decoding control tokens tensor: (batch, n_decoding_controls)
    decoding_control_tokens = torch.tensor([
        [1, 2, 3]  # [first_control, second_control, third_control] for batch
    ], dtype=torch.long)  # Shape: (1, 3)

    # Test the model
    model = FlexControlTripleStreamsVAE(config)
    print(f"Encoding control modes: {model.encoding_control_modes}")
    print(f"Decoding control modes: {model.decoding_control_modes}")
    print(f"Prepended encoding controls: {model.n_prepended_encoding_controls}")
    print(f"Prepended decoding controls: {model.n_prepended_decoding_controls}")

    # Test forward pass
    h_logits, v_logits, o_logits, mu, log_var, latent_z = model.forward(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )

    print(f"Forward pass successful:")
    print(f"  h_logits shape: {h_logits.shape}")
    print(f"  v_logits shape: {v_logits.shape}")
    print(f"  o_logits shape: {o_logits.shape}")
    print(f"  latent_z shape: {latent_z.shape}")

    # Test prediction
    hvo, latent_z = model.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )

    print(f"Prediction successful:")
    print(f"  hvo shape: {hvo.shape}")
    print(f"  latent_z shape: {latent_z.shape}")

    # Test with different control configuration (all compact_attention mode)
    config_all_compact = config.copy()
    config_all_compact.update({
        'n_encoding_control_tokens': [13, 10],
        'encoding_control_modes': ['compact_attention', 'compact_attention'],
        'n_decoding_control_tokens': [10, 10, 10],
        'decoding_control_modes': ['compact_attention', 'compact_attention', 'compact_attention']
    })

    model_all_compact = FlexControlTripleStreamsVAE(config_all_compact)
    print(f"\nAll-compact-attention model:")
    print(f"  Prepended encoding controls: {model_all_compact.n_prepended_encoding_controls}")
    print(f"  Prepended decoding controls: {model_all_compact.n_prepended_decoding_controls}")

    hvo_compact, _ = model_all_compact.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )
    print(f"  All-compact-attention prediction successful: {hvo_compact.shape}")

    # Test mixed modes
    config_mixed = config.copy()
    config_mixed.update({
        'n_encoding_control_tokens': [13, 10, 5],
        'encoding_control_modes': ['prepend', 'add', 'compact_attention'],
        'n_decoding_control_tokens': [10, 10, 10, 5],
        'decoding_control_modes': ['prepend', 'add', 'compact_attention', 'compact_attention']
    })

    # Update control tokens for mixed test
    encoding_control_tokens_mixed = torch.tensor([[1, 2, 1]], dtype=torch.long)  # (1, 3)
    decoding_control_tokens_mixed = torch.tensor([[1, 2, 3, 1]], dtype=torch.long)  # (1, 4)

    model_mixed = FlexControlTripleStreamsVAE(config_mixed)
    print(f"\nMixed-modes model:")
    print(f"  Encoding modes: {model_mixed.encoding_control_modes}")
    print(f"  Decoding modes: {model_mixed.decoding_control_modes}")
    print(f"  Prepended encoding controls: {model_mixed.n_prepended_encoding_controls}")
    print(f"  Prepended decoding controls: {model_mixed.n_prepended_decoding_controls}")

    hvo_mixed, _ = model_mixed.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens_mixed,
        decoding_control_tokens=decoding_control_tokens_mixed
    )
    print(f"  Mixed-modes prediction successful: {hvo_mixed.shape}")

    # Test TorchScript serialization
    print(f"\nTesting TorchScript serialization...")
    try:
        model.serialize(save_folder='./')
        print(f"✅ TorchScript serialization successful!")
    except Exception as e:
        print(f"❌ TorchScript serialization failed: {e}")

    # Test compatibility methods (if you need list interface for backward compatibility)
    print(f"\nTesting compatibility methods...")

    # Convert tensors back to lists for testing compatibility methods
    encoding_list = [encoding_control_tokens[:, i] for i in range(encoding_control_tokens.shape[1])]
    decoding_list = [decoding_control_tokens[:, i] for i in range(decoding_control_tokens.shape[1])]

    try:
        hvo_compat, _ = model.predict_with_lists(
            flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
            encoding_control_tokens_list=encoding_list,
            decoding_control_tokens_list=decoding_list
        )
        print(f"✅ Compatibility method successful: {hvo_compat.shape}")
    except Exception as e:
        print(f"❌ Compatibility method failed: {e}")

    print(f"\n🎉 All tests completed!")
    print(f"\n📝 Summary:")
    print(f"  - Added 'compact_attention' mode alongside 'prepend' and 'add'")
    print(f"  - CompactControlAttention applies learned attention between controls and all sequence positions")
    print(f"  - Controls using compact_attention influence the entire 32-step sequence through attention weights")
    print(f"  - Backward compatible with existing 'prepend' and 'add' modes")
    print(f"  - TorchScript compatible")

    model.serialize(save_folder='./', filename='triplestreams_vae_compact_attention.pt')
    model.load_torchscript('./triplestreams_vae_compact_attention.pt')
    model.config