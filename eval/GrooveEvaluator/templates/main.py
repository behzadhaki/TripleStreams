from eval.GrooveEvaluator import Evaluator
from eval.GrooveEvaluator import load_evaluator
import os
from data import Groove2Drum2BarDataset
import logging
import numpy as np


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