#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from eval.GrooveEvaluator import load_evaluator_template
from eval.UMAP import UMapper
import tqdm
import time
from model import BaseVAE, MuteVAE, MuteGenreLatentVAE, MuteLatentGenreInputVAE, TripleStreamsVAE

from logging import getLogger
logger = getLogger("helpers.eval_utils")
logger.setLevel("DEBUG")
from data import Groove2Drum2BarDataset


def batch_data_extractor(data_, device):
    # Extract the data from the batch
    inputs_ = data_[0].to(device) if data_[0].device.type != device else data_[0]
    genre_tags = data_[4].to(device) if data_[4].device.type != device else data_[4]
    outputs_ = data_[5].to(device) if data_[5].device.type != device else data_[5]
    kick_is_muted = data_[8].to(device) if data_[8].device.type != device else data_[8]
    snare_is_muted = data_[9].to(device) if data_[9].device.type != device else data_[9]
    hat_is_muted = data_[10].to(device) if data_[10].device.type != device else data_[10]
    tom_is_muted = data_[11].to(device) if data_[11].device.type != device else data_[11]
    cymbal_is_muted = data_[12].to(device) if data_[12].device.type != device else data_[12]
    return inputs_, genre_tags, outputs_, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted


def predict_using_batch_data(batch_data, model_, device):
    flat_hvo_groove, genre_tags, outputs_, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted = batch_data_extractor(
        batch_data, device)
    if isinstance(model_, BaseVAE):
        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=flat_hvo_groove)
    elif isinstance(model_, MuteVAE):
        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=flat_hvo_groove,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
    elif isinstance(model_, MuteGenreLatentVAE) or isinstance(model_, MuteLatentGenreInputVAE):
        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=flat_hvo_groove,
                genre_tags=genre_tags,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
    elif isinstance(model_, TripleStreamsVAE):
        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=flat_hvo_groove,
                genre_tags=genre_tags,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
    else:
        raise ValueError("Model is not recognized")

    return hvo, latent_z

def generate_umap_for_wandb(
        predict_using_batch_data,
        dataset_setting_json_path, subset_name,
        previous_loaded_dataset=None,
        down_sampled_ratio = 0.3):
    """
    Generate the umap for the given model and dataset setting.
    Args:
        :param model_: The model to be evaluated
        :param dataset_setting_json_path: The path to the dataset setting json file
        :param subset_name: The name of the subset to be evaluated
        :param previous_loaded_dataset: The previous loaded dataset to be used for evaluation (this optimizes the loading of the dataset - in the second epoch, pass the returned dataset from the first epoch)
        :param down_sampled_ratio: The ratio of the subset to be evaluated

    Returns:
        dictionary ready to be logged by wandb {f"{subset_name}_{umap}": wandb.Html}
    """

    start = time.time()

    if previous_loaded_dataset is None:
        test_dataset = Groove2Drum2BarDataset(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_name,
                max_len=32,
                tapped_voice_idx=2,
                collapse_tapped_sequence=True,
                down_sampled_ratio=down_sampled_ratio,
                move_all_to_gpu=False,
                use_cached=True
            )
    else:
        test_dataset = previous_loaded_dataset

    tags = [hvo_seq.metadata["style_primary"] for hvo_seq in test_dataset.hvo_sequences]

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    latents_z = None
    for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating UMAP"):
        _, z = predict_using_batch_data(batch_data=batch_data)

        if latents_z is None:
            latents_z = z.detach().cpu().numpy()
        else:
            latents_z = np.concatenate((latents_z, z.detach().cpu().numpy()), axis=0)

    try:
        umapper = UMapper(subset_name)
        umapper.fit(latents_z, tags_=tags)
        p = umapper.plot(show_plot=False, prepare_for_wandb=True)
        end = time.time()
        logger.info(f"UMAP Generation for {subset_name} took {end - start} seconds")
        return {f"{subset_name}_umap": p}, test_dataset

    except Exception as e:
        logger.info("UMAP failed for subset: {}".format(subset_name))
        return None, test_dataset

def generate_umap(
        model,
        device,
        test_dataset,
        subset_name):
    """
    Generate the umap for the given model and dataset setting.
    Args:
        :param model_: The model to be evaluated
        :param dataset_setting_json_path: The path to the dataset setting json file
        :param subset_name: The name of the subset to be evaluated
        :param previous_loaded_dataset: The previous loaded dataset to be used for evaluation (this optimizes the loading of the dataset - in the second epoch, pass the returned dataset from the first epoch)
        :param down_sampled_ratio: The ratio of the subset to be evaluated

    Returns:
        dictionary ready to be logged by wandb {f"{subset_name}_{umap}": wandb.Html}
    """

    start = time.time()

    tags = [hvo_seq.metadata["style_primary"] for hvo_seq in test_dataset.hvo_sequences]

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    latents_z = None
    for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating UMAP"):
        _, z = predict_using_batch_data(batch_data=batch_data, model_=model, device=device)

        if latents_z is None:
            latents_z = z.detach().cpu().numpy()
        else:
            latents_z = np.concatenate((latents_z, z.detach().cpu().numpy()), axis=0)

    try:
        umapper = UMapper(subset_name)
        umapper.fit(latents_z, tags_=tags)
        p = umapper.plot(show_plot=False, prepare_for_wandb=False, save_plot=False)
        end = time.time()
        logger.info(f"UMAP Generation for {subset_name} took {end - start} seconds")
        return p

    except Exception as e:
        logger.info("UMAP failed for subset: {}".format(subset_name))
        return None


def get_pianoroll_for_wandb(
        predict_using_batch_data,
        dataset_setting_json_path, subset_name,
        down_sampled_ratio,
        cached_folder="cached/Evaluators/templates/",
        divide_by_genre=True,
        previous_evaluator=None,
        **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :param predict_using_batch_data: The function to be used for prediction
    :param dataset_setting_json_path: The path to the dataset setting json file
    :param subset_name: The name of the subset to be evaluated
    :param down_sampled_ratio: The ratio of the subset to be evaluated
    :param cached_folder: The folder to be used for caching the evaluator template
    :param divide_by_genre: Whether to divide the subset by genre or not
    :param previous_evaluator: The previous evaluator to be used for logging (this optimizes the loading/creating of the evaluator). In the second epoch, pass the returned evaluator from the first epoch.
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    start = time.time()

    if previous_evaluator is not None:
        evaluator = previous_evaluator
    else:
        # load the evaluator template (or create a new one if it does not exist)
        evaluator = load_evaluator_template(
            dataset_setting_json_path=dataset_setting_json_path,
            subset_name=subset_name,
            down_sampled_ratio=down_sampled_ratio,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre
        )
    batch_data = evaluator.dataset[:]
    full_midi_filenames = [hvo_seq.metadata["full_midi_filename"] for hvo_seq in evaluator.dataset.hvo_sequences]
    hvos_array, _ = predict_using_batch_data(batch_data=batch_data)

    evaluator.add_unsorted_predictions(hvos_array.detach().cpu().numpy(), full_midi_filenames)

    # Get the media from the evaluator
    # -------------------------------
    media = evaluator.get_logging_media(
        prepare_for_wandb=True,
        need_hit_scores=kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else False,
        need_velocity_distributions=kwargs["need_velocity_distributions"] if "need_velocity_distributions" in kwargs.keys() else False,
        need_offset_distributions=kwargs["need_offset_distributions"] if "need_offset_distributions" in kwargs.keys() else False,
        need_rhythmic_distances=kwargs["need_rhythmic_distances"] if "need_rhythmic_distances" in kwargs.keys() else False,
        need_heatmap=kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else False,
        need_global_features=kwargs["need_global_features"] if "need_global_features" in kwargs.keys() else False,
        need_piano_roll=kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else False,
        need_audio=kwargs["need_audio"] if "need_audio" in kwargs.keys() else False,
        need_kl_oa=kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else False)

    end = time.time()
    logger.info(f"PianoRoll Generation for {subset_name} took {end - start} seconds")

    return media, evaluator


def get_hit_scores(
        predict_using_batch_data, dataset_setting_json_path, subset_name,
        down_sampled_ratio,
        cached_folder="cached/Evaluators/templates/",
        previous_evaluator=None,
        divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_name))
    # and model is correct type

    start = time.time()

    if previous_evaluator is not None:
        evaluator = previous_evaluator
    else:
        # load the evaluator template (or create a new one if it does not exist)
        evaluator = load_evaluator_template(
            dataset_setting_json_path=dataset_setting_json_path,
            subset_name=subset_name,
            down_sampled_ratio=down_sampled_ratio,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre
        )

    print(f"evaluator = load_evaluator_template("
          f"dataset_setting_json_path={dataset_setting_json_path},"
          f"subset_name={subset_name},"
          f"down_sampled_ratio={down_sampled_ratio},"
          f"cached_folder={cached_folder},"
          f"divide_by_genre={divide_by_genre}")

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        evaluator.dataset,
        batch_size=128,
        shuffle=False,
    )

    predictions = []


    for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating Hit Scores - {subset_name}"):
        hvos_array, latent_z = predict_using_batch_data(batch_data=batch_data)
        predictions.append(hvos_array.detach().cpu().numpy())

    full_midi_filenames = [hvo_seq.metadata["full_midi_filename"] for hvo_seq in evaluator.dataset.hvo_sequences]
    evaluator.add_unsorted_predictions(np.concatenate(predictions), full_midi_filenames)

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"Hit_Scores/{key}_mean_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['mean']) for key, value
                  in sorted(hit_dict.items())}

    score_dict.update({f"Hit_Scores/{key}_std_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['std']) for key, value
                  in sorted(hit_dict.items())})

    end = time.time()
    logger.info(f"Hit Scores Generation for {subset_name} took {end - start} seconds")
    return score_dict, evaluator
