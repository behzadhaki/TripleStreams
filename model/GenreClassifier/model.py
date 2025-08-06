#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import json
import os

import model.GenreClassifier.components as components


class GenreClassifier(torch.nn.Module):
    """
    An encoder-encoder VAE transformer
    """
    def __init__(self, config):
        """
        This is a VAE transformer which for encoder and decoder uses the same transformer architecture
        (that is, uses the Vanilla Transformer Encoder)
        :param config: a dictionary containing the following keys:
            d_model: the dimension of the model for the encoder
            nhead: the number of heads for the encoder
            dim_feedforward: the dimension of the feedforward network in the encoder
            num_layers: the number of encoder layers
            dropout: the dropout rate
            device: the device to use
        """

        super(GenreClassifier, self).__init__()

        self.config = config
        self.n_genres = config['n_genres']

        # Layers for the Groove2Drum VAE
        # ---------------------------------------------------
        self.InputLayerEncoder = components.InputGrooveLayer(
            embedding_size=27,
            d_model=self.config['d_model'],
            max_len=32,
            velocity_dropout=float(self.config['velocity_dropout']),
            offset_dropout=float(self.config['offset_dropout']),
            positional_encoding_dropout=float(self.config['dropout'])
        )

        self.Encoder = components.Encoder(
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            dim_feedforward=self.config['dim_feedforward'],
            num_encoder_layers=self.config['num_layers'],
            dropout=float(self.config['dropout'])
        )

        self.ClassifierLinear = torch.nn.Linear(self.config['d_model'], self.n_genres)
        
        self.init_weights(0.1)

    def init_weights(self, initrange):
        # Initialize weights and biases
        self.InputLayerEncoder.init_weights(initrange)
        self.ClassifierLinear.weight.data.uniform_(-initrange, initrange)
        self.ClassifierLinear.bias.data.zero_()


    @torch.jit.export
    def forward(self, drum_pattern_hvo: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        :param drum_pattern_hvo: [N, 32, 27]
        :return: last time step of the memory
        [N,

        """
        x, hit, hvo_projection = self.InputLayerEncoder(hvo=drum_pattern_hvo)
        memory = self.Encoder(x)  # N x (32+1) x d_model

        return self.ClassifierLinear(memory[:, -1, :])

    @torch.jit.export
    def predict(self, drum_pattern_hvo: torch.Tensor):

        logits = self.forward(drum_pattern_hvo)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1), probs

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
        'd_model': 128,
        'nhead': 4,
        'dim_feedforward': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'velocity_dropout': 0.1,
        'offset_dropout': 0.2,
        'n_genres': 12,
        'device': 'cpu'
    }

    model = GenreClassifier(config)

    model.forward(
        drum_pattern_hvo=torch.rand(1, 32, 27)
    )

    model.predict(
        drum_pattern_hvo=torch.rand(1, 32, 27)
    )

    model.serialize(save_folder='./')

