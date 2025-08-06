import torch
from sklearn.metrics import confusion_matrix
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# model loader
def load_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    """ Load a GenreGlobalDensityWithVoiceMutesVAE model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        is_evaluating (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GenreDensityTempoVAE): the loaded model
    """

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

import wandb, os
from model import GenreClassifier, BaseVAE, MuteVAE, MuteGenreLatentVAE, MuteLatentGenreInputVAE
import shutil
def download_model_from_wandb(epoch, version, run_name, model_type, new_path=None):

    if model_type is GenreClassifier:
        wandb_p = "GenreClassifier"
    elif model_type is BaseVAE:
        wandb_p = "BaseVAE"
    elif model_type is MuteVAE:
        wandb_p = "MuteVAE"
    elif model_type is MuteGenreLatentVAE:
        wandb_p = "MuteGenreLatentVAE"
    elif model_type is MuteLatentGenreInputVAE:
        wandb_p = "MuteLatentGenreInputVAE"
    else:
        raise ValueError("Model type not supported")

    artifact_path = f"behzadhaki/{wandb_p}/model_epoch_{epoch}:v{version}"
    epoch = artifact_path.split("model_epoch_")[-1].split(":")[0]

    print("Downloading artifact")
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    # rename {epoch}.pth to {run_name}.pth
    if new_path is not None:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(os.path.join(artifact_dir, f"{epoch}.pth"), new_path)

    if new_path is not None:
        print("Artifact available at: ", new_path)
    else:
        print("Artifact available at: ", os.path.join(artifact_dir, f"{run_name}.pth"))

def predict_using_model(model, dataset, indices=None):
    if indices is None:
        data = dataset[:]
    else:
        data = dataset[indices]

    # extract i/o
    flat_grooves = data[0]
    genre_tags = data[4]
    drum_grooves = data[5]
    kick_is_muted = data[8]
    snare_is_muted = data[9]
    hat_is_muted = data[10]
    tom_is_muted = data[11]
    cymbal_is_muted = data[12]

    model.eval()

    # predict
    with torch.no_grad():
        if isinstance(model, GenreClassifier):
            predicted_genres, predicted_probs_per_genre = model.predict(drum_grooves)
            return predicted_genres, predicted_probs_per_genre
        elif isinstance(model, BaseVAE):
            hvo, latent_z = model.predict(flat_hvo_groove=flat_grooves)
            return hvo, latent_z
        elif isinstance(model, MuteVAE):
            hvo, latent_z = model.predict(
                flat_hvo_groove=flat_grooves,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
            return hvo, latent_z
        elif isinstance(model, MuteGenreLatentVAE) or isinstance(model, MuteLatentGenreInputVAE):
            hvo, latent_z = model.predict(
                flat_hvo_groove=flat_grooves,
                genre_tags=genre_tags,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
            return hvo, latent_z
        else:
            raise ValueError("Model type not supported")

def sample_using_latents(model, latents, genre_tag):

    # extract i/o
    kick_is_muted = torch.tensor([0])
    snare_is_muted = torch.tensor([0])
    hat_is_muted = torch.tensor([0])
    tom_is_muted = torch.tensor([0])
    cymbal_is_muted = torch.tensor([0])
    voice_thresholds = torch.tensor([0.5] * 9)
    voice_max_count_allowed = torch.tensor([32] * 9)
    model.eval()

    # predict
    with torch.no_grad():
        if isinstance(model, BaseVAE):
            h, v, o = model.sample(
                latent_z=latents,
                voice_thresholds=voice_thresholds,
                voice_max_count_allowed=voice_max_count_allowed
            )
            return torch.cat([h, v, o], dim=2)

        elif isinstance(model, MuteVAE) or isinstance(model, MuteLatentGenreInputVAE):
            h, v, o = model.sample(
                latent_z=latents,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted,
                voice_thresholds=voice_thresholds,
                voice_max_count_allowed=voice_max_count_allowed)
            return torch.cat([h, v, o], dim=2)

        elif isinstance(model, MuteGenreLatentVAE):
            h, v, o = model.sample(
                latent_z=latents,
                genre=genre_tag,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted,
                voice_thresholds=voice_thresholds,
                voice_max_count_allowed=voice_max_count_allowed)
            return torch.cat([h, v, o], dim=2)
        else:
            raise ValueError("Model type not supported")

def sample_using_latent_per_genre_dict(model, latents_per_genre_dict, style_tags):
    patterns_per_genre_dict = {}
    for genre, latents in latents_per_genre_dict.items():
        genre_ = torch.tensor([style_tags.index(genre)]).to(latents.device)
        hvo = sample_using_latents(model, latents, genre_)
        patterns_per_genre_dict[genre] = hvo
    return patterns_per_genre_dict


def predict_genres_using_hvo_and_model(hvos, model):
    model.eval()
    with torch.no_grad():
        predicted_genres, predicted_probs_per_genre = model.predict(hvos)

    # sample 100 times from probs
    sampled_genres = []
    genres = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(1):
        for probs in predicted_probs_per_genre:
            sampled_genres.append(np.random.choice(genres, p=probs.cpu().numpy()))

    return np.array(sampled_genres), predicted_probs_per_genre

from bokeh.layouts import row
from bokeh.models import Panel, Tabs

def synthesize_visualize_using_models(models, dataset, idx):
    """
    Synthesize and visualize the samples using the given model
    Args:
        dataset: torch.utils.data.Dataset
        model: torch.nn.Module
        indices: list of int, optional
            indices of the dataset to be used for umap generation

    Returns:
        dictionary ready to be logged by wandb {f"{subset_name}_{umap}": wandb.Html}
    """
    if not isinstance(models, list):
        models = [models]

    sample = dataset[idx]
    print("density =", sample[6], "genre =", dataset.genre_tags[sample[4]])

    generated_tabs = []
    audios = []

    for model in models:
        hvo_pred, _ = predict_using_model(model, dataset, indices=[idx])
        target_hvo_seq = dataset.hvo_sequences[idx]
        predicted_hvo_seq = dataset.hvo_sequences[idx].copy_empty()
        predicted_hvo_seq.hvo = hvo_pred[0, :, :].squeeze().detach().cpu().numpy()

        # get_plots
        pr_true = target_hvo_seq.piano_roll(width=600, height=300)
        pr_pred = predicted_hvo_seq.piano_roll(width=600, height=300)

        # HStack the plots
        fig = row(pr_true, pr_pred)

        generated_tabs.append(Panel(child=fig, title=f"{model.__class__.__name__}"))

        audio = predicted_hvo_seq.synthesize(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")
        audios.append(audio)

    gt_audio = target_hvo_seq.synthesize(sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2")

    return Tabs(tabs=generated_tabs), audios, gt_audio


def plot_confusion_matrix(cm, labels):
    if not HAS_SEABORN:
        raise ImportError("Seaborn is not installed. Please install it using 'pip install seaborn'")
    if plt is None:
        raise ImportError("Matplotlib is not installed. Please install it using 'pip install matplotlib'")

    plt.figure(figsize=(6, 6))
    sns.set(font_scale=0.8)
    return sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False,
                       annot_kws={"size": 9})


def generated_data_confusion_matrix(dataset, model_drum, model_classifier, indices=None):

    # predict drum grooves
    hvo, _ = predict_using_model(model_drum, dataset, indices=indices)

    # predict genres
    predicted_genres, predicted_probs_per_genre = predict_genres_using_hvo_and_model(hvo, model_classifier)

    # get ground truth
    target_genres = dataset.genre_targets[indices] if indices is not None else dataset.genre_targets

    labels = dataset.genre_tags
    labels = [label.split("/")[0] for label in labels]

    cm = confusion_matrix(target_genres, predicted_genres)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    scores = {
        "accuracy": accuracy_score(target_genres, predicted_genres),
        "precision": precision_score(target_genres, predicted_genres, average='weighted'),
        "recall": recall_score(target_genres, predicted_genres, average='weighted'),
        "f1": f1_score(target_genres, predicted_genres, average='weighted'),
    }

    return cm, scores, labels

def get_genre_accuracy_of_patterns(hvos, model_classifier, targets, indices=None):

        # predict genres
        predicted_genres, predicted_probs_per_genre = predict_genres_using_hvo_and_model(hvos, model_classifier)

        # get ground truth
        target_genres = targets[indices] if indices is not None else targets

        # cm = confusion_matrix(target_genres, predicted_genres)
        # cm = cm / cm.sum(axis=1)[:, np.newaxis]

        scores = {
            "accuracy": accuracy_score(target_genres, predicted_genres),
            "precision": precision_score(target_genres, predicted_genres, average='macro'),
            "recall": recall_score(target_genres, predicted_genres, average='macro'),
            "f1": f1_score(target_genres, predicted_genres, average='macro'),
        }

        return scores

def classifier_confusion_matrix(dataset, model_classifier, indices=None):
# predict genres
    predicted_genres, predicted_probs_per_genre = predict_using_model(model_classifier, dataset, indices=indices)

    # get ground truth
    target_genres = dataset.genre_targets[indices] if indices is not None else dataset.genre_targets

    cm = confusion_matrix(target_genres, predicted_genres)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    labels = dataset.genre_tags
    labels = [label.split("/")[0] for label in labels]


    scores = {
        "accuracy": accuracy_score(target_genres, predicted_genres),
        "precision": precision_score(target_genres, predicted_genres, average='weighted'),
        "recall": recall_score(target_genres, predicted_genres, average='weighted'),
        "f1": f1_score(target_genres, predicted_genres, average='weighted'),
    }

    return cm, scores, labels


def jaccard_similarity(hvo1, hvo2):
    assert len(hvo1.shape) == 3 and len(hvo2.shape) == 3, "make sure you have batch dimension"
    n_voices = hvo1.shape[-1] // 3
    hvo_flat1 = torch.flatten(hvo1[:,:,:n_voices], start_dim=1)
    hvo_flat2 = torch.flatten(hvo2[:,:,:n_voices], start_dim=1)
    overlap_hits = hvo_flat1 * hvo_flat2
    overlap_sum = torch.sum(overlap_hits, dim=1)
    union_hits = hvo_flat1 + hvo_flat2 - overlap_hits
    union_sum = torch.sum(union_hits, dim=1)
    jaccard = overlap_sum / union_sum
    return jaccard


def velocity_distribution(hvo, only_at_hits=False, average_per_sample_first=False):
    assert len(hvo.shape) == 3, "make sure you have batch dimension"
    n_voices = hvo.shape[-1] // 3
    if not only_at_hits:
        vel = hvo[:, :, n_voices:n_voices*2]
    else:
        hits = hvo[:, :, :n_voices]
        vel = hvo[:, :, n_voices:n_voices*2]
        vel = vel[hits > 0.5]

    if average_per_sample_first:
        vel = vel.mean(dim=-1).mean(dim=-1)
        print(vel.shape)

    vel_mean = torch.mean(vel).item()
    vel_std = torch.std(vel).item()

    return np.round(vel_mean, 2), np.round(vel_std, 2)

def velocity_error(hvo1, hvo2, only_at_hits=False):
    assert len(hvo1.shape) == 3 and len(hvo2.shape) == 3, "make sure you have batch dimension"

    n_voices = hvo2.shape[-1] // 3

    if not only_at_hits:
        vel1 = hvo1[:, :, n_voices:n_voices*2]
        vel2 = hvo2[:, :, n_voices:n_voices*2]
    else:
        hits1 = hvo1[:, :, :n_voices]
        hits2 = hvo2[:, :, :n_voices]
        vel1 = hvo1[:, :, n_voices:n_voices*2]
        vel2 = hvo2[:, :, n_voices:n_voices*2]
        vel1 = vel1[(hits1 + hits2) > 0.5]
        vel2 = vel2[(hits1 + hits2) > 0.5]

    vel_diff = torch.abs(vel1 - vel2)
    print(vel_diff.shape)
    vel_diff_mean = torch.mean(vel_diff).item()
    vel_diff_std = torch.std(vel_diff).item()

    return np.round(vel_diff_mean, 2), np.round(vel_diff_std, 2)

def offset_distribution(hvo, only_at_hits=False, average_per_sample_first=False):
    assert len(hvo.shape) == 3, "make sure you have batch dimension"
    n_voices = hvo.shape[-1] // 3
    if not only_at_hits:
        offset = hvo[:, :, (n_voices*2):]
    else:
        hits = hvo[:, :, :n_voices]
        offset = hvo[:, :, (n_voices*2):]
        offset = offset[hits > 0.5]

    if average_per_sample_first:
        offset = offset.mean(dim=-1).mean(dim=-1)

    offset_mean = torch.mean(offset).item()
    offset_std = torch.std(offset).item()

    return np.round(offset_mean, 2), np.round(offset_std, 2)


def separate_latents_by_style(latents, corresponding_styles, style_tags):
    style_latents = {}
    for i in range(len(style_tags)):
        style_indices = torch.argwhere(corresponding_styles == i).squeeze()
        style_latents[style_tags[i]] = latents[style_indices]
    return style_latents

def get_mean_std_per_latent_dim_per_genre(latents, corresponding_styles, style_tags):
    style_latents = separate_latents_by_style(latents, corresponding_styles, style_tags)
    style_latents_mean_std = {}
    for style, latent in style_latents.items():
        style_latents_mean_std[style] = {
            'mean': latent.mean(dim=0),
            'std': latent.std(dim=0)
        }
    return style_latents_mean_std


def generate_n_random_samples_per_style(n, latents=None, corresponding_styles=None, style_tags=None):
    if latents is None or corresponding_styles is None or style_tags is None:
        return torch.normal(mean=0, std=1, size=(n, 128))
    else:
        mean_stds_per_style = get_mean_std_per_latent_dim_per_genre(latents, corresponding_styles, style_tags)
        random_samples = {}
        for style, mean_std in mean_stds_per_style.items():
            random_samples[style] = torch.zeros((n, 128))
            for i in range(n):
                random_samples[style][i] = torch.normal(mean=mean_std['mean'], std=mean_std['std'])
        return random_samples


def extract_rhythms(drum_hvos, hvo_seq_template, flatten_voices=False):
    extracted_groove = np.zeros((drum_hvos.shape[0], 32, 27))
    for i, hvo_s in enumerate(drum_hvos.numpy()):
        hvo_seq_ = hvo_seq_template.copy_empty()
        hvo_seq_.hvo = hvo_s
        if flatten_voices:
            extracted_groove[i] = hvo_seq_.flatten_voices()
        else:
            extracted_groove[i] = hvo_seq_.flatten_voices(reduce_dim=True)
    return extracted_groove