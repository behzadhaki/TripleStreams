#  Copyright (c) 2025. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model.FlexControlTripleStreamsVAE.components import TensorBasedInputGrooveLayer, Encoder, TensorBasedLatentLayer, \
    TensorBasedDecoderInput, \
    SingleFeatureOutputLayer


class FlexControlTripleStreamsVAE(torch.nn.Module):
    """
    An encoder-decoder VAE transformer with flexible control token support
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
            encoding_control_modes: list of strings, mode for each encoding control ('prepend' or 'add')
            n_decoding_control_tokens: list of ints, number of tokens for each decoding control
            decoding_control_modes: list of strings, mode for each decoding control ('prepend' or 'add')

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
    def get_latent_dim(self):
        return self.latent_dim

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
            print("🔄 Detected legacy checkpoint format. Converting to flexible format...")
            state_dict = self._convert_legacy_state_dict(state_dict)
            print("✅ Successfully converted legacy checkpoint parameters")

        return super().load_state_dict(state_dict, strict=strict)

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

    def save(self, save_path, additional_info=None):
        """ Saves the model to the given path. The Saved pickle has all the parameters ('params' field) as well as
        the state_dict ('state_dict' field) """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_ = dict()
        for key, value in self.config.items():
            config_[key] = value
        json.dump(config_, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': config_,
                    'additional_info': additional_info}, save_path)

    # serializes to a torchscript model
    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):

        os.makedirs(save_folder, exist_ok=True)

        if filename is None:
            import datetime
            filename = f'GenDensTempVAE_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        is_train = self.training
        self.eval()
        save_path = os.path.join(save_folder, filename)

        scr = torch.jit.script(self)
        # save model
        with open(save_path, "wb") as f:
            torch.jit.save(scr, f)

        if is_train:
            self.train()


if __name__ == "__main__":
    # Test configuration with flexible controls
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

        # Flexible control configuration
        'n_encoding_control_tokens': [13, 10],
        'encoding_control_modes': ['prepend', 'add'],
        'n_decoding_control_tokens': [10, 10, 10],
        'decoding_control_modes': ['prepend', 'prepend', 'prepend']
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

    # Test with different control configuration (all add mode)
    config_all_add = config.copy()
    config_all_add.update({
        'n_encoding_control_tokens': [13, 10],
        'encoding_control_modes': ['add', 'add'],
        'n_decoding_control_tokens': [10, 10, 10],
        'decoding_control_modes': ['add', 'add', 'add']
    })

    model_all_add = FlexControlTripleStreamsVAE(config_all_add)
    print(f"\nAll-add model:")
    print(f"  Prepended encoding controls: {model_all_add.n_prepended_encoding_controls}")
    print(f"  Prepended decoding controls: {model_all_add.n_prepended_decoding_controls}")

    hvo_add, _ = model_all_add.predict(
        flat_hvo_groove=torch.rand(batch_size, 32, config["embedding_size_src"]),
        encoding_control_tokens=encoding_control_tokens,
        decoding_control_tokens=decoding_control_tokens
    )
    print(f"  All-add prediction successful: {hvo_add.shape}")

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

    model.serialize(save_folder='./', filename='triplestreams_vae_test.pt')