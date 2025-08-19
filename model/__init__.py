# Enhanced __init__.py with improved loading mechanism

# BaseVAE Imports
from model.BaseVAE.model import BaseVAE

# TripleStreamVAE Imports
from model.TripleStreamsVAE.model import TripleStreamsVAE
from model.FlexControlTripleStreamsVAE.model import FlexControlTripleStreamsVAE

import torch
import json
import os
from pathlib import Path


def load_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    """
    Enhanced model loading with automatic config detection.

    Args:
        model_path: Path to the .pth model file
        model_class: Model class to instantiate
        params_dict: Optional config dict or path. If None, tries to load from model file
        is_evaluating: Whether to set model to eval mode
        device: Device to load model on

    Returns:
        Loaded model instance
    """
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device, weights_only=False)
        else:
            loaded_dict = torch.load(model_path, weights_only=False)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # Try to get config from various sources in order of preference
    config = None

    if params_dict is not None:
        # User provided explicit config
        if isinstance(params_dict, str):
            with open(params_dict, 'r') as f:
                config = json.load(f)
        else:
            config = params_dict
        print(f"✅ Using user-provided config")

    elif 'params' in loaded_dict:
        # Config embedded in model file (this has been the standard format)
        config = loaded_dict['params']
        print(f"✅ Using config from model file")

    else:
        # Fallback: try to find companion .json file (legacy format)
        model_path_obj = Path(model_path)
        json_path = model_path_obj.with_suffix('.json')

        if json_path.exists():
            with open(json_path, 'r') as f:
                config = json.load(f)
            print(f"⚠️  Using legacy config file: {json_path}")
        else:
            raise Exception(
                f"Could not find model configuration. Tried:\n"
                f"1. User-provided params_dict\n"
                f"2. Embedded config in model file\n"
                f"3. Companion JSON file: {json_path}\n"
                f"Please provide a params_dict or ensure config is embedded in model file."
            )

    # Instantiate and load model
    model = model_class(config)
    model.load_state_dict(loaded_dict["model_state_dict"])

    if is_evaluating:
        model.eval()

    print(f"🎉 Successfully loaded {model_class.__name__}")
    return model


def load_model_auto(model_path, is_evaluating=True, device=None):
    """
    Automatically detect model type and load with embedded config.
    Works with the standard format where config is stored as 'params'.

    Args:
        model_path: Path to the .pth model file
        is_evaluating: Whether to set model to eval mode
        device: Device to load model on

    Returns:
        Loaded model instance
    """
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device, weights_only=False)
        else:
            loaded_dict = torch.load(model_path, weights_only=False)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    if 'params' not in loaded_dict:
        raise Exception(
            f"No embedded config found in model file. "
            f"Use load_model() with explicit model_class and params_dict for models without embedded config."
        )

    config = loaded_dict['params']

    # Auto-detect model class based on config keys
    if 'n_encoding_control_tokens' in config and 'encoding_control_modes' in config:
        model_class = FlexControlTripleStreamsVAE
        print(f"🔍 Auto-detected: FlexControlTripleStreamsVAE")
    elif 'embedding_size_tgt' in config and config.get('embedding_size_tgt', 0) % 3 == 0:
        model_class = TripleStreamsVAE
        print(f"🔍 Auto-detected: TripleStreamsVAE")
    else:
        model_class = BaseVAE
        print(f"🔍 Auto-detected: BaseVAE")

    # Load using standard mechanism
    return load_model(model_path, model_class, config, is_evaluating, device)


# Enhanced save method for FlexControlTripleStreamsVAE
def enhanced_save(model, save_path, additional_info=None, include_json=True):
    """
    Enhanced save method that embeds config in the model file.

    Args:
        model: Model instance to save
        save_path: Path to save model
        additional_info: Additional metadata to include
        include_json: Whether to also save separate .json config file for legacy compatibility
    """
    if not save_path.endswith('.pth'):
        save_path += '.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare config
    config = dict()
    for key, value in model.config.items():
        config[key] = value

    # Save model with embedded config
    save_dict = {
        'model_state_dict': model.state_dict(),
        'params': config,
        'additional_info': additional_info,
        'model_class': model.__class__.__name__,
        'save_version': '2.0'  # Version for tracking save format
    }

    torch.save(save_dict, save_path)
    print(f"✅ Saved model with embedded config: {save_path}")

    # Optionally save separate JSON for legacy compatibility
    if include_json:
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Also saved legacy JSON config: {json_path}")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Load with auto-detection (new models with embedded config)
    try:
        model = load_model_auto('path/to/new_model.pth')
        print("Auto-loading successful!")
    except Exception as e:
        print(f"Auto-loading failed: {e}")

    # Example 2: Load legacy model with explicit config
    try:
        model = load_model(
            'path/to/legacy_model.pth',
            FlexControlTripleStreamsVAE,
            'path/to/config.json'
        )
        print("Legacy loading successful!")
    except Exception as e:
        print(f"Legacy loading failed: {e}")

    # Example 3: Load with embedded config but explicit model class
    try:
        model = load_model(
            'path/to/new_model.pth',
            FlexControlTripleStreamsVAE
        )
        print("Explicit class loading successful!")
    except Exception as e:
        print(f"Explicit class loading failed: {e}")