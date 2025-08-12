from eval.GrooveEvaluator import Evaluator
from eval.GrooveEvaluator import load_evaluator
import os
from data import Groove2Drum2BarDataset, Groove2TripleStreams2BarDataset
import logging
import numpy as np
from hvo_sequence.hvo_seq import HVO_Sequence
import torch

logger = logging.getLogger("eval.GrooveEvaluator.templates.main")
logger.setLevel("DEBUG")


def create_template(dataset, identifier,
                    cached_folder="cached/Evaluators/templates/", divide_by_genre=True):
    """
    Create a template for the given dataset and subset. The template will ALWAYS be saved in the cached_folder.
    :param dataset:                     The dataset object. (e.g. Groove2Drum2BarDataset)
    :param identifier:                 The name of the subset to be used. (e.g. "train", "test", "validation")
    :param cached_folder:               The folder to save the template.
    :param divide_by_genre:             Whether to divide the dataset by genre.
    :return:
    """

    # load data using the settings specified in the dataset_setting_json_path

    # create a list of filter dictionaries for each genre if divide_by_genre is True
    if divide_by_genre:
        list_of_filter_dicts_for_subsets = []
        if dataset is None:
            styles = [
                "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
                "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]
        else:
            styles = dataset.genre_tags

        for style in styles:
            list_of_filter_dicts_for_subsets.append(
                {"style_primary": [style]}  # , "beat_type": ["beat"], "time_signature": ["4-4"]}
            )

    else:
        list_of_filter_dicts_for_subsets = None

    _identifier = identifier

    # create the evaluator
    eval = Evaluator(
        dataset=dataset,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier=_identifier,
        n_samples_to_use=-1,
        max_hvo_shape=(32, 27),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=False,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        need_kl_oa=False,
        n_samples_to_synthesize_and_draw="all",
        disable_tqdm=False)

    eval.dump(path=cached_folder)

    logger.info(f"Template created and cached at {cached_folder}")
    return eval


def load_evaluator_template(dataset_setting_json_path, subset_name,
                            down_sampled_ratio, cached_folder="cached/Evaluators/templates/",
                            divide_by_genre=True, disable_logging=True):
    """
    Load a template for the given dataset and subset. If the template does not exist, it will be created and
    automatically saved in the cached_folder.

    :param dataset_setting_json_path:  The path to the json file that contains the dataset settings.

    :param subset_name:             The name of the subset to be used. (e.g. "train", "test", "validation")
    :param down_sampled_ratio:      The ratio of the down-sampled set. (e.g. 0.02)
    :param cached_folder:           The folder to save the template.
    :return:                        The evaluator template.
    """
    dataset_name = dataset_setting_json_path.split("/")[-1].split(".")[0]

    _identifier = f"_{down_sampled_ratio}_ratio_of_{dataset_name}_{subset_name}" \
        if down_sampled_ratio is not None else f"complete_set_of_{dataset_name}_{subset_name}"
    path = os.path.join(cached_folder, f"{_identifier}_evaluator.Eval.bz2")
    print (path)

    if os.path.exists(path):
        if not disable_logging:
            logger.info(f"Loading template from {path}")
        eval = load_evaluator(path)

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

        eval.dataset = test_dataset

        return eval
    else:
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

        return create_template(
            dataset=test_dataset,
            identifier=_identifier,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre)

def stack_groove_with_outputs(input_hvos, output_hvos):
    if len(input_hvos.shape) == 2:
        # add batch dim
        input_hvos = input_hvos.unsqueeze(0)
        output_hvo = output_hvos.unsqueeze(0)

    assert input_hvos.shape[0] == output_hvos.shape[0]     # batch
    assert input_hvos.shape[1] == output_hvos.shape[1]     # timesteps
    n_overall_voices = input_hvos.shape[-1] // 3 + output_hvos.shape[-1] // 3
    stacked_hvos = torch.zeros((output_hvos.shape[0], output_hvos.shape[1], n_overall_voices*3), dtype=input_hvos.dtype)

    stacked_hvos[:, :, 0] = input_hvos[:, :, 0]
    stacked_hvos[:, :, n_overall_voices] = input_hvos[:, :, 1]
    stacked_hvos[:, :, n_overall_voices * 2] = input_hvos[:, :, 2]

    n_outputs_voices = output_hvos.shape[-1] // 3
    for i in range(n_outputs_voices):
        stacked_hvos[:, :, 1 + i] = output_hvos[:, :, i]  # stream hit
        stacked_hvos[:, :, 1 + n_overall_voices + i] = output_hvos[:, :, i + n_outputs_voices]  # stream vel
        stacked_hvos[:, :, 1 + n_overall_voices * 2 + i] = output_hvos[:, :, i + 2 * n_outputs_voices]  # stream offset

    return stacked_hvos

def compile_into_list_of_hvo_seqs(output_hvos, metadatas, input_hvos=None, qpms=None):
    # input_hvos: list of arrays
    # output_hvos: list of arrays
    # metadata: list of dictionaries

    hvo_sequence_list = []

    drum_mapping = {"groove": [36]} if input_hvos is not None else {}

    for i in range(output_hvos.shape[0]):
        input_hvo = input_hvos[i, :, :] if input_hvos is not None else None
        output_hvo = output_hvos[i, :, :]
        metadata = metadatas[i]

        n_outputs_voices = output_hvo.shape[-1] // 3
        n_overall_voices = n_outputs_voices + 1 if input_hvo is not None else n_outputs_voices

        for i in range(n_outputs_voices):
            drum_mapping[f"stream_{i + 1}"] = [37 + i]

        temp_hvo_seq = HVO_Sequence(drum_mapping=drum_mapping, beat_division_factors=[4])

        qpm = 120
        if qpms is not None:
            qpm = qpms[i]

        temp_hvo_seq.add_tempo(0, qpm=120)
        temp_hvo_seq.add_time_signature(0, 4, 4)
        temp_hvo_seq.adjust_length(output_hvo.shape[0])

        g_dim_shift = 0
        if input_hvo is not None:
            temp_hvo_seq.hvo[:, 0] = input_hvo[:, 0]  # groove hit
            temp_hvo_seq.hvo[:, n_overall_voices] = input_hvo[:, 1]  # groove vel
            temp_hvo_seq.hvo[:, n_overall_voices * 2] = input_hvo[:, 2]  # groove offset
            g_dim_shift = 1

        for i in range(n_outputs_voices):
            temp_hvo_seq.hvo[:, g_dim_shift + i] = output_hvo[:, i]  # stream hit
            temp_hvo_seq.hvo[:, g_dim_shift + n_overall_voices + i] = output_hvo[:, i + n_outputs_voices]  # stream vel
            temp_hvo_seq.hvo[:, g_dim_shift + n_overall_voices * 2 + i] = output_hvo[:, i + 2 * n_outputs_voices]  # stream offset

        temp_hvo_seq.metadata.update(metadata)
        hvo_sequence_list.append(temp_hvo_seq)

    return hvo_sequence_list

def create_triple_streams_template(dataset, identifier,
                                   cached_folder,
                                   divide_by_collection=True,
                                   use_input_in_hvo_sequences=False):
    """
    Create a template for the given dataset and subset. The template will ALWAYS be saved in the cached_folder.
    :param dataset:                     The dataset object. (e.g. Groove2Drum2BarDataset)
    :param identifier:                 The name of the subset to be used. (e.g. "train", "test", "validation")
    :param cached_folder:               The folder to save the template.
    :param divide_by_collection:             Whether to divide the dataset by collection.
    :return:
    """

    # load data using the settings specified in the dataset_setting_json_path

    # create a list of filter dictionaries for each genre if divide_by_genre is True
    if divide_by_collection:
        list_of_filter_dicts_for_subsets = []
        if dataset is None:
            styles = ["Unknown"]
        else:
            styles = sorted(list(set(dataset.collection).union()))

        for style in styles:
            list_of_filter_dicts_for_subsets.append(
                {"style_primary": [style]}  # , "beat_type": ["beat"], "time_signature": ["4-4"]}
            )

    else:
        list_of_filter_dicts_for_subsets = None

    _identifier = identifier

    hvo_sequences = compile_into_list_of_hvo_seqs(
        output_hvos=dataset.output_streams,
        metadatas=dataset.metadata,
        input_hvos=dataset.input_grooves if use_input_in_hvo_sequences else None)
    # create the evaluator
    eval = Evaluator(
        dataset=dataset,
        hvo_sequences_list_= hvo_sequences,
        has_hvo_sequences_in_dataset=False,
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        _identifier=_identifier,
        n_samples_to_use=-1,
        max_hvo_shape=(32, 9) if not use_input_in_hvo_sequences else (32, 12),
        need_hit_scores=False,
        need_velocity_distributions=False,
        need_offset_distributions=False,
        need_rhythmic_distances=False,
        need_heatmap=False,
        need_global_features=False,
        need_audio=False,
        need_piano_roll=False,
        need_kl_oa=False,
        n_samples_to_synthesize_and_draw="all",
        disable_tqdm=False
    )

    # m1 = [hs.metadata for hs in hvo_sequences]

    eval.dump(path=cached_folder)

    logger.info(f"Template created and cached at {cached_folder}")
    return eval



def load_triple_streams_evaluator_template(
        config,
        subset_tag,
        use_input_in_hvo_sequences,
        cached_folder,
        downsampled_size=None,
        use_cached=False,
        divide_by_collection=True,
        disable_logging=True,):
    """
    Load a template for the given dataset and subset. If the template does not exist, it will be created and
    automatically saved in the cached_folder.

    :param dataset_setting_json_path:  The path to the json file that contains the dataset settings.

    :param subset_name:             The name of the subset to be used. (e.g. "train", "test", "validation")
    :param down_sampled_ratio:      The ratio of the down-sampled set. (e.g. 0.02)
    :param cached_folder:           The folder to save the template.
    :return:                        The evaluator template.
    """
    dataset_name = config['dataset_setting_json_path'].split("/")[-1].split(".")[0]

    _identifier = f"_{downsampled_size}_samples_from_{dataset_name}_{subset_tag}" \
        if downsampled_size is not None else f"complete_set_of_{dataset_name}_{subset_tag}"
    path = os.path.join(cached_folder, f"{_identifier}_evaluator.Eval.bz2")
    print (path)

    if os.path.exists(path) and use_cached:
        print(f"\n\n\n============= Using cached evaluator at {path}")
        if not disable_logging:
            logger.info(f"Loading template from {path}")
        eval = load_evaluator(path)

        test_dataset = Groove2TripleStreams2BarDataset(
            config=config,
            subset_tag=subset_tag,
            use_cached=use_cached,
            downsampled_size=downsampled_size,
            force_regenerate=False
        )

        eval.dataset = test_dataset

        return eval
    else:
        test_dataset = test_dataset = Groove2TripleStreams2BarDataset(
            config=config,
            subset_tag=subset_tag,
            use_cached=use_cached,
            downsampled_size=downsampled_size,
            )

        return create_triple_streams_template(
            dataset=test_dataset,
            identifier=_identifier,
            cached_folder=cached_folder,
            divide_by_collection=divide_by_collection,
            use_input_in_hvo_sequences=use_input_in_hvo_sequences)