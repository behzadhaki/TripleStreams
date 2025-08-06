#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

from model.MuteVAE import components as components


class MuteVAE(torch.nn.Module):
    """
    An encoder-encoder VAE transformer
    """
    def __init__(self, config):
        """
        This is a VAE transformer which for encoder and decoder uses the same transformer architecture
        (that is, uses the Vanilla Transformer Encoder)
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
            max_len_enc: the maximum length of the input sequence
            max_len_dec: the maximum length of the output sequence
            device: the device to use
        """

        super(MuteVAE, self).__init__()

        assert config['embedding_size_tgt'] % 3 == 0, 'embedding_size_tgt must be divisible by 3'

        self.config = config
        self.latent_dim = config['latent_dim']

        # Layers for the Groove2Drum VAE
        # ---------------------------------------------------
        self.InputLayerEncoder = components.InputGrooveLayer(
            embedding_size=self.config['embedding_size_src'],
            d_model=self.config['d_model_enc'],
            max_len=self.config['max_len_enc'],
            velocity_dropout=float(self.config['velocity_dropout']),
            offset_dropout=float(self.config['offset_dropout']),
            positional_encoding_dropout=float(self.config['dropout'])
        )

        self.Encoder = components.Encoder(
            d_model=self.config['d_model_enc'],
            nhead=self.config['nhead_enc'],
            dim_feedforward=self.config['dim_feedforward_enc'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.latentLayer = components.LatentLayer(
            max_len=self.config['max_len_enc'],
            d_model=self.config['d_model_enc'],
            latent_dim=self.config['latent_dim']
        )

        self.HitsDecoderInput = components.DecoderInput(
            max_len=self.config['max_len_dec'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec']
        )

        self.HitsDecoder = components.Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.HitsOutputLayer = components.SingleFeatureOutputLayer(
            embedding_size=self.config['embedding_size_tgt'] // 3,
            d_model=self.config['d_model_dec'],
        )

        self.velocityDecoderInput = components.DecoderInput(
            max_len=self.config['max_len_dec'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec']
        )

        self.VelocityDecoder = components.Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.VelocityOutputLayer = components.SingleFeatureOutputLayer(
            embedding_size=self.config['embedding_size_tgt'] // 3,
            d_model=self.config['d_model_dec'],
        )

        self.OffsetDecoderInput = components.DecoderInput(
            max_len=self.config['max_len_dec'],
            latent_dim=self.config['latent_dim'],
            d_model=self.config['d_model_dec']
        )

        self.OffsetDecoder = components.Encoder(
            d_model=self.config['d_model_dec'],
            nhead=self.config['nhead_dec'],
            dim_feedforward=self.config['dim_feedforward_dec'],
            num_encoder_layers=self.config['num_decoder_layers'],
            dropout=float(self.config['dropout'])
        )

        self.OffsetOutputLayer = components.SingleFeatureOutputLayer(
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
    def encodeLatent(self, flat_hvo_groove: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        :param flat_hvo_groove: [N, 32, 3]
        :return: mu, log_var, latent_z, memory
                mu:            [N, latent_dim]
                log_var:       [N, latent_dim]
                latent_z:      [N, latent_dim]
                memory:        [N, 32, d_model_enc]

        """
        x, hit, hvo_projection = self.InputLayerEncoder(hvo=flat_hvo_groove)
        memory = self.Encoder(x)  # N x (32+1) x d_model_enc
        mu, log_var, latent_z = self.latentLayer(memory)
        return mu, log_var, latent_z, memory

    @torch.jit.export
    def encode_all(self, flat_hvo_groove: torch.Tensor):
        """
        just here to make the model compatible with the previous CompGenVAE (in VST)
        """
        return self.encodeLatent(flat_hvo_groove)

    @torch.jit.export
    def decode(self, latent_z: torch.Tensor,
               kick_is_muted: torch.Tensor, snare_is_muted: torch.Tensor,
               hat_is_muted: torch.Tensor, tom_is_muted: torch.Tensor,
               cymbal_is_muted: torch.Tensor):
        """
        Decodes the latent_z (N x latent_dim)  (through the decoder


        :param latent_z:            [N, latent_dim]
        :param kick_is_muted:       [N, 1]
        :param snare_is_muted:      [N, 1]
        :param hat_is_muted:        [N, 1]
        :param tom_is_muted:        [N, 1]
        :param cymbal_is_muted:     [N, 1]

        :return:                    h_logits, v_logits, o_logits, hvo_logits

                h_logits: [N, 32, 1]
                v_logits: [N, 32, 1]
                o_logits: [N, 32, 1]
                hvo_logits: [N, 32, 27]

                None of the returned logits are activated (no sigmoid applied)
        """
        
        h_logits = self.HitsOutputLayer(
            self.HitsDecoder(
                self.HitsDecoderInput(
                    latent_z=latent_z,
                    kick_is_muted=kick_is_muted,
                    snare_is_muted=snare_is_muted,
                    hat_is_muted=hat_is_muted,
                    tom_is_muted=tom_is_muted,
                    cymbal_is_muted=cymbal_is_muted
                )
            )
        )
        
        v_logits = self.VelocityOutputLayer(
            self.VelocityDecoder(
                self.velocityDecoderInput(
                    latent_z=latent_z,
                    kick_is_muted=kick_is_muted,
                    snare_is_muted=snare_is_muted,
                    hat_is_muted=hat_is_muted,
                    tom_is_muted=tom_is_muted,
                    cymbal_is_muted=cymbal_is_muted
                )
            )
        )
        
        o_logits = self.OffsetOutputLayer(
            self.OffsetDecoder(
                self.OffsetDecoderInput(
                    latent_z=latent_z,
                    kick_is_muted=kick_is_muted,
                    snare_is_muted=snare_is_muted,
                    hat_is_muted=hat_is_muted,
                    tom_is_muted=tom_is_muted,
                    cymbal_is_muted=cymbal_is_muted
                )
            )
        )

        hvo_logits = torch.cat([h_logits, v_logits, o_logits], dim=-1)
        
        return h_logits, v_logits, o_logits, hvo_logits

    @torch.jit.export
    def interpolate_2d(self,
                       interpolation_factor: float,
                       latent_z_1: torch.Tensor,
                       mute_controls_1: torch.Tensor,
                       latent_z_2: torch.Tensor,
                       mute_controls_2: torch.Tensor,
                       voice_thresholds: torch.Tensor,
                       voice_max_count_allowed: torch.Tensor,
                       sampling_mode: int = 0):

        h_logits = self.HitsOutputLayer(
            self.HitsDecoder(
                self.HitsDecoderInput.interpolate_2d(
                    interpolation_factor=interpolation_factor,
                    latent_z_1=latent_z_1,
                    mute_controls_1=mute_controls_1,
                    latent_z_2=latent_z_2,
                    mute_controls_2=mute_controls_2
                )
            )
        )

        v_logits = self.VelocityOutputLayer(
            self.VelocityDecoder(
                self.velocityDecoderInput.interpolate_2d(
                    interpolation_factor=interpolation_factor,
                    latent_z_1=latent_z_1,
                    mute_controls_1=mute_controls_1,
                    latent_z_2=latent_z_2,
                    mute_controls_2=mute_controls_2
                )
            )
        )

        o_logits = self.OffsetOutputLayer(
            self.OffsetDecoder(
                self.OffsetDecoderInput.interpolate_2d(
                    interpolation_factor=interpolation_factor,
                    latent_z_1=latent_z_1,
                    mute_controls_1=mute_controls_1,
                    latent_z_2=latent_z_2,
                    mute_controls_2=mute_controls_2
                )
            )
        )

        h = torch.zeros_like(h_logits)

        if sampling_mode == 0:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                max_indices = torch.topk(h_logits[:, :, ix], max_count).indices[0]
                h[:, max_indices, ix] = h_logits[:, max_indices, ix]
                h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        elif sampling_mode == 1:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                # sample using probability distribution of hits (_h)
                voice_probs = h_logits[:, :, ix]
                sampled_indices = torch.bernoulli(voice_probs)
                max_indices = torch.topk(sampled_indices * voice_probs, max_count).indices[0]
                h[:, max_indices, ix] = 1

        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

        # if v < 0.05, then set corresponding h to 0
        h = torch.where(v < 0.05, torch.zeros_like(h), h)

        return h, v, o

    @torch.jit.export
    def interpolate_3d(self,
                       ab_interpolation_factor: float,
                       abc_interpolation_factor: float,
                       latent_z_a: torch.Tensor,
                       mute_controls_a: torch.Tensor,
                       latent_z_b: torch.Tensor,
                       mute_controls_b: torch.Tensor,
                       latent_z_c: torch.Tensor,
                       mute_controls_c: torch.Tensor,
                       voice_thresholds: torch.Tensor,
                       voice_max_count_allowed: torch.Tensor,
                       sampling_mode: int = 0):

        h_logits = self.HitsOutputLayer(
            self.HitsDecoder(
                self.HitsDecoderInput.interpolate_3d(
                    ab_interpolation_factor=ab_interpolation_factor,
                    abc_interpolation_factor=abc_interpolation_factor,
                    latent_z_a=latent_z_a,
                    mute_controls_a=mute_controls_a,
                    latent_z_b=latent_z_b,
                    mute_controls_b=mute_controls_b,
                    latent_z_c=latent_z_c,
                    mute_controls_c=mute_controls_c
                )
            )
        )

        v_logits = self.VelocityOutputLayer(
            self.VelocityDecoder(
                self.velocityDecoderInput.interpolate_3d(
                    ab_interpolation_factor=ab_interpolation_factor,
                    abc_interpolation_factor=abc_interpolation_factor,
                    latent_z_a=latent_z_a,
                    mute_controls_a=mute_controls_a,
                    latent_z_b=latent_z_b,
                    mute_controls_b=mute_controls_b,
                    latent_z_c=latent_z_c,
                    mute_controls_c=mute_controls_c
                )
            )
        )

        o_logits = self.OffsetOutputLayer(
            self.OffsetDecoder(
                self.OffsetDecoderInput.interpolate_3d(
                    ab_interpolation_factor=ab_interpolation_factor,
                    abc_interpolation_factor=abc_interpolation_factor,
                    latent_z_a=latent_z_a,
                    mute_controls_a=mute_controls_a,
                    latent_z_b=latent_z_b,
                    mute_controls_b=mute_controls_b,
                    latent_z_c=latent_z_c,
                    mute_controls_c=mute_controls_c
                )
            )
        )

        h = torch.zeros_like(h_logits)

        if sampling_mode == 0:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                max_indices = torch.topk(h_logits[:, :, ix], max_count).indices[0]
                h[:, max_indices, ix] = h_logits[:, max_indices, ix]
                h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        elif sampling_mode == 1:
            for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
                # sample using probability distribution of hits (_h)
                voice_probs = h_logits[:, :, ix]
                sampled_indices = torch.bernoulli(voice_probs)
                max_indices = torch.topk(sampled_indices * voice_probs, max_count).indices[0]
                h[:, max_indices, ix] = 1

        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

        # if v < 0.05, then set corresponding h to 0
        h = torch.where(v < 0.05, torch.zeros_like(h), h)

        return h, v, o

    @torch.jit.export
    def sample(self,
               latent_z: torch.Tensor,
               kick_is_muted: torch.Tensor,
               snare_is_muted: torch.Tensor,
               hat_is_muted: torch.Tensor,
               tom_is_muted: torch.Tensor,
               cymbal_is_muted: torch.Tensor,
               voice_thresholds: torch.Tensor,
               voice_max_count_allowed: torch.Tensor,
               sampling_mode: int = 0):

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z,
            kick_is_muted=kick_is_muted,
            snare_is_muted=snare_is_muted,
            hat_is_muted=hat_is_muted,
            tom_is_muted=tom_is_muted,
            cymbal_is_muted=cymbal_is_muted
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
    def predict(self,
                flat_hvo_groove: torch.Tensor,
                kick_is_muted: torch.Tensor,
                snare_is_muted: torch.Tensor,
                hat_is_muted: torch.Tensor,
                tom_is_muted: torch.Tensor,
                cymbal_is_muted: torch.Tensor,
                threshold: float=0.5):

        h_logits, v_logits, o_logits, mu, log_var, latent_z = self.forward(
            flat_hvo_groove=flat_hvo_groove,
            kick_is_muted=kick_is_muted,
            snare_is_muted=snare_is_muted,
            hat_is_muted=hat_is_muted,
            tom_is_muted=tom_is_muted,
            cymbal_is_muted=cymbal_is_muted
        )

        _h = torch.sigmoid(h_logits)
        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

        h = torch.zeros_like(_h)

        voice_thresholds = torch.tensor([threshold] * 9)
        voice_max_count_allowed = torch.tensor([32] * 9)

        for ix, (thres, max_count) in enumerate(zip(voice_thresholds, voice_max_count_allowed)):
            max_indices = torch.topk(_h[:, :, ix], max_count).indices[0]
            h[:, max_indices, ix] = _h[:, max_indices, ix]
            h[:, :, ix] = torch.where(h[:, :, ix] > thres, 1, 0)

        # if v < 0.05, then set corresponding h to 0
        h = torch.where(v < 0.05, torch.zeros_like(h), h)


        hvo = torch.cat([h, v, o], dim=-1)

        return hvo, latent_z

    @torch.jit.export
    def forward(self, flat_hvo_groove: torch.Tensor,
                kick_is_muted: torch.Tensor,
                snare_is_muted: torch.Tensor,
                hat_is_muted: torch.Tensor,
                tom_is_muted: torch.Tensor,
                cymbal_is_muted: torch.Tensor):


        mu, log_var, latent_z, memory = self.encodeLatent(flat_hvo_groove)

        h_logits, v_logits, o_logits, hvo_logits = self.decode(
            latent_z=latent_z,
            kick_is_muted=kick_is_muted,
            snare_is_muted=snare_is_muted,
            hat_is_muted=hat_is_muted,
            tom_is_muted=tom_is_muted,
            cymbal_is_muted=cymbal_is_muted
        )

        return h_logits, v_logits, o_logits, mu, log_var, latent_z


    @torch.jit.ignore
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

    config = {
        'd_model_enc': 128,
        'd_model_dec': 128,
        'embedding_size_src': 3,
        'embedding_size_tgt': 27,
        'nhead_enc': 4,
        'nhead_dec': 8,
        'dim_feedforward_enc': 128,
        'dim_feedforward_dec': 512,
        'num_encoder_layers': 3,
        'num_decoder_layers': 6,
        'dropout': 0.1,
        'latent_dim': 16,
        'max_len_enc': 32,
        'max_len_dec': 32,
        'velocity_dropout': 0.1,
        'offset_dropout': 0.2,
        'n_density_bins': 3,
        'n_tempo_bins': 7,
        'device': 'cpu'
    }

    model = MuteVAE(config)
    kick_is_muted = torch.tensor([1])
    snare_is_muted = torch.tensor([1])
    hat_is_muted = torch.tensor([1])
    tom_is_muted = torch.tensor([1])
    cymbal_is_muted = torch.tensor([1])

    model.forward(
        flat_hvo_groove=torch.rand(1, 32, 3),
        kick_is_muted=kick_is_muted,
        snare_is_muted=snare_is_muted,
        hat_is_muted=hat_is_muted,
        cymbal_is_muted=cymbal_is_muted,
        tom_is_muted=tom_is_muted,
    )

    model.decode(
        latent_z=torch.rand(1, 16),
        kick_is_muted=kick_is_muted,
        snare_is_muted=snare_is_muted,
        hat_is_muted=hat_is_muted,
        cymbal_is_muted=cymbal_is_muted,
        tom_is_muted=tom_is_muted,
    )

    model.serialize(save_folder='./')

    latent_a = torch.rand(1, 16)
    latent_b = torch.rand(1, 16)
    latent_c = torch.rand(1, 16)
    mute_a = torch.tensor([0, 1, 0, 0, 0])
    mute_b = torch.tensor([1, 0, 0, 0, 0])
    mute_c = torch.tensor([0, 0, 1, 1, 1])

    model.interpolate_3d(
        ab_interpolation_factor=0.5,
        abc_interpolation_factor=0.5,
        latent_z_a=latent_a,
        mute_controls_a=mute_a,
        latent_z_b=latent_b,
        mute_controls_b=mute_b,
        latent_z_c=latent_c,
        mute_controls_c=mute_c,
        voice_thresholds=torch.tensor([0.5]*9),
        voice_max_count_allowed=torch.tensor([32]*9),
        sampling_mode=0
    )

    model.interpolate_2d(
        interpolation_factor=0.5,
        latent_z_1=latent_a,
        mute_controls_1=mute_a,
        latent_z_2=latent_b,
        mute_controls_2=mute_b,
        voice_thresholds=torch.tensor([0.5]*9),
        voice_max_count_allowed=torch.tensor([32]*9),
        sampling_mode=0
    )