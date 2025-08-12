#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from eval.GrooveEvaluator import load_evaluator_template, load_triple_streams_evaluator_template, stack_groove_with_outputs
from eval.UMAP import UMapper
import tqdm
import time

from helpers.eval_utils import predict_using_batch_data
from model import BaseVAE, MuteVAE, MuteGenreLatentVAE, MuteLatentGenreInputVAE, TripleStreamsVAE

from logging import getLogger
logger = getLogger("helpers.eval_utils")
logger.setLevel("DEBUG")
from data import Groove2TripleStreams2BarDataset


# def batch_data_extractor(data_, device):
#     # Extract the data from the batch
#     input_grooves = data_[0].to(device) if data_[0].device.type != device else data_[0]
#     output_streams = data_[1].to(device) if data_[1].device.type != device else data_[1]
#     encoding_control1_tokens = data_[2].to(device) if data_[2].device.type != device else data_[2]
#     encoding_control2_tokens = data_[3].to(device) if data_[3].device.type != device else data_[3]
#     decoding_control1_tokens = data_[4].to(device) if data_[4].device.type != device else data_[4]
#     decoding_control2_tokens = data_[5].to(device) if data_[5].device.type != device else data_[5]
#     decoding_control3_tokens = data_[6].to(device) if data_[6].device.type != device else data_[6]
#     indices = data_[7]
#
#     return (input_grooves,
#             output_streams,
#             encoding_control1_tokens,
#             encoding_control2_tokens,
#             decoding_control1_tokens,
#             decoding_control2_tokens,
#             decoding_control3_tokens,
#             indices)
#
#
# def predict_using_batch_data(batch_data, model_=model_on_device, device):
#     (input_grooves,
#      output_streams,
#      encoding_control1_tokens,
#      encoding_control2_tokens,
#      decoding_control1_tokens,
#      decoding_control2_tokens,
#      decoding_control3_tokens,
#      indices) = batch_data_extractor(batch_data, device)
#
#     with torch.no_grad():
#         hvo, latent_z = model_.predict(
#             flat_hvo_groove=input_grooves,
#             encoding_control1_token=encoding_control1_tokens,
#             encoding_control2_token=encoding_control2_tokens,
#             decoding_control1_token=decoding_control1_tokens,
#             decoding_control2_token=decoding_control2_tokens,
#             decoding_control3_token=decoding_control3_tokens)
#
#     return hvo, latent_z

def generate_umap_for_wandb(
        config,
        subset_tag,
        use_cached,
        downsampled_size,
        predict_using_batch_data_method=None,
        tag_key="collection",
        previous_loaded_dataset=None,
):
    """
    Generate the umap for the given model and dataset setting.

    Args:
        :param config: config dictionary similiar to training config
        :param subset_tag: subset tag to use ('train'/'test'/'validation')
        :param use_cached: at the end of first call (usually epoch 0), will decide whether to use different samples or
                           allow reusing one that was cached in a previous run
        :param downsampled_size: Number of random samples to use for umap generation
        :param tag_key: coloring data based on this key (use metadata key or 'collection' here)
        :param previous_loaded_dataset: dataset loaded in a prior epoch

    Returns:
        dictionary ready to be logged by wandb {f"{subset_tag}_{umap}": wandb.Html}
    """

    start = time.time()

    if previous_loaded_dataset is None:
        test_dataset = Groove2TripleStreams2BarDataset(
            config=config,
            subset_tag=subset_tag,
            use_cached=use_cached,
            downsampled_size=downsampled_size,
            )
    else:
        test_dataset = previous_loaded_dataset

    if tag_key == "collection":
        tags_all = test_dataset.collection
    else:
        tags_all = test_dataset.collection[tag_key]

    tags_used = []

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    predict_using_batch_data = predict_using_batch_data_method if predict_using_batch_data_method is not None else predict_using_batch_data
    latents_z = None
    for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating UMAP"):
        _, z = predict_using_batch_data(batch_data=batch_data)

        tags_used.extend([tags_all[ix] for ix in batch_data[-1]])

        if latents_z is None:
            latents_z = z.detach().cpu().numpy()
        else:
            latents_z = np.concatenate((latents_z, z.detach().cpu().numpy()), axis=0)

    try:
        umapper = UMapper(subset_tag)
        umapper.fit(latents_z, tags_=tags_used)
        p = umapper.plot(show_plot=False, prepare_for_wandb=True)
        end = time.time()
        logger.info(f"UMAP Generation for {subset_tag} took {end - start} seconds")
        return {f"{subset_tag}_umap": p}, test_dataset

    except Exception as e:
        logger.info(f"UMAP failed for subset: {subset_tag}".format(subset_tag))
        return None, test_dataset




def get_pianoroll_for_wandb(
        config,
        subset_tag,
        use_cached,
        downsampled_size,
        predict_using_batch_data_method=None,
        tag_key="collection",
        previous_loaded_dataset=None,
        cached_folder="cached/Evaluators/templates/",
        divide_by_collection=True,   # use collection instead
        previous_evaluator=None,
        **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :param predict_using_batch_data: The function to be used for prediction
    :param dataset_setting_json_path: The path to the dataset setting json file
    :param subset_tag: The name of the subset to be evaluated
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
        evaluator = load_triple_streams_evaluator_template(
            config=config,
            subset_tag=subset_tag,
            downsampled_size=downsampled_size,
            cached_folder=cached_folder,
            use_cached=use_cached,
            divide_by_collection=divide_by_collection,
            use_input_in_hvo_sequences=True
        )
    batch_data = evaluator.dataset[:]
    metadatas = batch_data[-1]
    full_midi_filenames = [mtd["full_midi_filename"] for mtd in metadatas]
    predict_using_batch_data = predict_using_batch_data_method if predict_using_batch_data_method is not None \
        else predict_using_batch_data
    hvos_array, _ = predict_using_batch_data(batch_data=batch_data)
    hvos_array = stack_groove_with_outputs(input_hvos=batch_data[0], output_hvos=hvos_array)
    evaluator.add_unsorted_predictions(
        hvos_array.detach().cpu().numpy(),
        prediction_full_midi_filenames=full_midi_filenames,
    )

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
    logger.info(f"PianoRoll Generation for {subset_tag} took {end - start} seconds")

    return media, evaluator


def get_hit_scores(
        predict_using_batch_data, dataset_setting_json_path, subset_tag,
        down_sampled_ratio,
        cached_folder="cached/Evaluators/templates/",
        previous_evaluator=None,
        divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_tag))
    # and model is correct type

    start = time.time()

    if previous_evaluator is not None:
        evaluator = previous_evaluator
    else:
        # load the evaluator template (or create a new one if it does not exist)
        evaluator = load_evaluator_template(
            dataset_setting_json_path=dataset_setting_json_path,
            subset_tag=subset_tag,
            down_sampled_ratio=down_sampled_ratio,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre
        )

    print(f"evaluator = load_evaluator_template("
          f"dataset_setting_json_path={dataset_setting_json_path},"
          f"subset_tag={subset_tag},"
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


    for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating Hit Scores - {subset_tag}"):
        hvos_array, latent_z = predict_using_batch_data(batch_data=batch_data)
        predictions.append(hvos_array.detach().cpu().numpy())

    full_midi_filenames = [hvo_seq.metadata["full_midi_filename"] for hvo_seq in evaluator.dataset.hvo_sequences]
    evaluator.add_unsorted_predictions(np.concatenate(predictions), full_midi_filenames)

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"Hit_Scores/{key}_mean_{subset_tag}".replace(" ", "_").replace("-", "_"): float(value['mean']) for key, value
                  in sorted(hit_dict.items())}

    score_dict.update({f"Hit_Scores/{key}_std_{subset_tag}".replace(" ", "_").replace("-", "_"): float(value['std']) for key, value
                  in sorted(hit_dict.items())})

    end = time.time()
    logger.info(f"Hit Scores Generation for {subset_tag} took {end - start} seconds")
    return score_dict, evaluator



def generate_umap(
        model,
        device,
        test_dataset,
        subset_tag):
    """
    Generate the umap for the given model and dataset setting.
    Args:
        :param model_: The model to be evaluated
        :param dataset_setting_json_path: The path to the dataset setting json file
        :param subset_tag: The name of the subset to be evaluated
        :param previous_loaded_dataset: The previous loaded dataset to be used for evaluation (this optimizes the loading of the dataset - in the second epoch, pass the returned dataset from the first epoch)
        :param down_sampled_ratio: The ratio of the subset to be evaluated

    Returns:
        dictionary ready to be logged by wandb {f"{subset_tag}_{umap}": wandb.Html}
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
        umapper = UMapper(subset_tag)
        umapper.fit(latents_z, tags_=tags)
        p = umapper.plot(show_plot=False, prepare_for_wandb=False, save_plot=False)
        end = time.time()
        logger.info(f"UMAP Generation for {subset_tag} took {end - start} seconds")
        return p

    except Exception as e:
        logger.info("UMAP failed for subset: {}".format(subset_tag))
        return None