# MuteLatentGenreInputVAE Imports
from model.MuteLatentGenreInputVAE.model import MuteLatentGenreInputVAE

# GenMuteVAEMultiTask Imports
from model.MuteGenreLatentVAE.model import MuteGenreLatentVAE

# MuteVAE Imports
from model.MuteVAE.model import MuteVAE

# BaseVAE Imports
from model.BaseVAE.model import BaseVAE

# GenreClassifier Imports
from model.GenreClassifier.model import GenreClassifier

import torch
def load_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = model_class(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model