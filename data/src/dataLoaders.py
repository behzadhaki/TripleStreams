from data.src.utils import (get_data_directory_using_filters, get_drum_mapping_using_label,
                            load_original_gmd_dataset_pickle, extract_hvo_sequences_dict, pickle_hvo_dict)
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from math import ceil
import json
import logging
import random
from data import *

import hashlib
import base64
from typing import Literal

import tqdm
import os
import bz2
import pickle
import io
import numpy as np

logging.basicConfig(level=logging.DEBUG)
dataLoaderLogger = logging.getLogger("data.Base.dataLoaders")



# ------------ helpers: encode/decode arrays as .npy bytes ------------
def _ndarray_to_npy_bytes(arr, allow_pickle=False):
    buf = io.BytesIO()
    # .npy format is versioned & stable; this avoids numpy's pickle reducers
    np.save(buf, arr, allow_pickle=allow_pickle)
    return buf.getvalue()

def _npy_bytes_to_ndarray(b, allow_pickle=False):
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=allow_pickle)

def _encode_for_pickle(obj, allow_pickle_arrays=False):
    """
    Recursively convert ndarrays into a small marker dict with .npy bytes.
    Everything else is returned as-is (and will be pickled normally).
    """
    if isinstance(obj, np.ndarray):
        return {"__npy__": True, "data": _ndarray_to_npy_bytes(obj, allow_pickle=allow_pickle_arrays)}
    elif isinstance(obj, dict):
        return {k: _encode_for_pickle(v, allow_pickle_arrays) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_encode_for_pickle(v, allow_pickle_arrays) for v in obj)
    else:
        return obj

def _decode_after_unpickle(obj, allow_pickle_arrays=False):
    """
    Inverse of _encode_for_pickle: turn marker dicts back into ndarrays.
    """
    if isinstance(obj, dict):
        if obj.keys() == {"__npy__", "data"} and obj.get("__npy__") is True:
            return _npy_bytes_to_ndarray(obj["data"], allow_pickle=allow_pickle_arrays)
        else:
            return {k: _decode_after_unpickle(v, allow_pickle_arrays) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_decode_after_unpickle(v, allow_pickle_arrays) for v in obj)
    else:
        return obj

def load_pkl_bz2_dict(dataset_pkl_bz2, *, allow_pickle_arrays=False):
    """
    Load a compiled dataset saved by compile_data_for_multiple_datasets_pkl().
    """
    assert dataset_pkl_bz2.endswith(".pkl.bz2"), "Dataset name must end with .pkl.bz2"
    compiled_data_path = dataset_pkl_bz2
    if not os.path.exists(compiled_data_path):
        raise FileNotFoundError(f"No compiled dataset found at {compiled_data_path}")

    with bz2.BZ2File(compiled_data_path, 'rb') as f:
        raw = pickle.load(f)
    return _decode_after_unpickle(raw, allow_pickle_arrays=allow_pickle_arrays)


def hash_string_to_name(
    text: str,
    method: Literal['md5', 'sha1', 'sha256'] = 'md5',
    length: int = 8,
    prefix: str = '',
    encoding: Literal['hex', 'base64', 'base32'] = 'hex'
) -> str:
    """
    Create a reproducible hashed name from a long string.

    Args:
        text: Input string to hash
        method: Hash algorithm ('md5', 'sha1', 'sha256')
        length: Length of the hash to keep (after encoding)
        prefix: Optional prefix for the hashed name
        encoding: How to encode the hash ('hex', 'base64', 'base32')

    Returns:
        Reproducible hashed name
    """
    # Create hash object
    if method == 'md5':
        hasher = hashlib.md5()
    elif method == 'sha1':
        hasher = hashlib.sha1()
    elif method == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash method: {method}")

    # Hash the text
    hasher.update(text.encode('utf-8'))

    # Get the hash in different encodings
    if encoding == 'hex':
        hash_str = hasher.hexdigest()
    elif encoding == 'base64':
        hash_str = base64.b64encode(hasher.digest()).decode('ascii').rstrip('=')
    elif encoding == 'base32':
        hash_str = base64.b32encode(hasher.digest()).decode('ascii').rstrip('=').lower()
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    # Truncate to desired length
    hash_str = hash_str[:length]

    # Add prefix if provided
    return f"{prefix}{hash_str}" if prefix else hash_str

def deterministic_short_name(text: str, max_length: int = 12) -> str:
    """
    Create a deterministic short name that's somewhat readable.

    Args:
        text: Input string
        max_length: Maximum length of output

    Returns:
        Short, deterministic name
    """
    # If already short enough, clean and return
    if len(text) <= max_length:
        clean = ''.join(c.lower() if c.isalnum() else '_' for c in text)
        return clean[:max_length].strip('_')

    # Take first part + hash
    first_part = ''.join(c.lower() for c in text if c.isalnum())[:4]
    if not first_part:
        first_part = "str"

    # Calculate remaining space for hash
    remaining = max_length - len(first_part) - 1  # -1 for underscore
    if remaining < 4:
        remaining = 4
        first_part = first_part[:max_length - remaining - 1]

    hash_part = hash_string_to_name(text, length=remaining, encoding='hex')

    return f"{first_part}_{hash_part}"

def get_bin_bounds_for_voice_densities(voice_counts_per_sample: list, num_nonzero_bins=3):
    """
    Calculates the lower and upper bounds for the voice density bins

    category 0: no hits


    :param voice_counts_per_sample:
    :param num_nonzero_bins:
    :return: lower_bounds, upper_bounds
    """

    assert num_nonzero_bins > 0, "num_nonzero_bins should be greater than 0"

    non_zero_counts = sorted([count for count in voice_counts_per_sample if count > 0])

    samples_per_bin = len(non_zero_counts) // num_nonzero_bins

    grouped_bins = [non_zero_counts[i * samples_per_bin: (i + 1) * samples_per_bin] for i in range(num_nonzero_bins)]

    lower_bounds = [group[0] for group in grouped_bins]
    upper_bounds = [group[-1] for group in grouped_bins]
    upper_bounds[-1] = non_zero_counts[-1] + 1

    return lower_bounds, upper_bounds


def map_voice_densities_to_categorical(voice_counts, lower_bounds, upper_bounds):
    """
    Maps the voice counts to a categorical value based on the lower and upper bounds provided
    :param voice_counts:
    :param lower_bounds:
    :param upper_bounds:
    :return:
    """

    categories = []
    adjusted_upper_bounds = upper_bounds.copy()
    adjusted_upper_bounds[-1] = adjusted_upper_bounds[
                                    -1] + 1  # to ensure that the last bin is inclusive on the upper bound

    for count in voice_counts:
        if count == 0:
            categories.append(0)
        else:
            for idx, (low, high) in enumerate(zip(lower_bounds, adjusted_upper_bounds)):
                if low <= count < high:
                    categories.append(idx + 1)
                    break

    return categories

def map_tempo_to_categorical(tempo, n_tempo_bins=6):
    """
    Maps the tempo to a categorical value based on the following bins:
    0-60, 60-76, 76-108, 108-120, 120-168, 168-Above
    :param tempo:
    :param n_tempo_bins: [int] number of tempo bins to use (default is 6 and only 6 is supported at the moment)
    :return:
    """
    if n_tempo_bins != 6:
        raise NotImplementedError("Only 6 bins are supported for tempo mapping at the moment")

    if tempo < 60:
        return 0
    elif 60 <= tempo < 76:
        return 1
    elif 76 <= tempo < 108:
        return 2
    elif 108 <= tempo < 120:
        return 3
    elif 120 <= tempo < 168:
        return 4
    elif 168 <= tempo:
        return 5

def map_global_density_to_categorical(total_hits, max_hits, n_global_density_bins=8):
    """
    hit increase per bin = max_hits / n_global_density_bins

    :param total_hits:
    :param lower_bounds:
    :param upper_bounds:
    :return:
    """
    assert False, "This function is not used in the current implementation"

    step_res = max_hits / n_global_density_bins
    categories = []
    categories = [int(count / step_res) for count in total_hits]


    return categories

def map_value_to_bins(value, edges):
    """
    Maps a value to a bin based on the edges provided
    :param value:
    :param edges:
    :return:
    """
    for i in range(len(edges)+1):
        if i == 0:
            if value < edges[i]:
                return i
        elif i == len(edges):
            if value >= edges[-1]:
                return i
        else:
            if edges[i - 1] <= value < edges[i]:
                return i

    print("SHOULD NOT REACH HERE")

def TokenizeControls(
        control_array: np.ndarray,
        n_bins: int,
        low: float = 0,
        high: float = 1
) -> np.ndarray:
    """
    Map control values to integer bin indices using uniform binning.

    Args:
        control_array: Numpy array of control values to bin
        n_bins: Number of bins to create
        low: Lower bound of binning range (default 0)
        high: Upper bound of binning range (default 1)

    Returns:
        Numpy array of integer bin indices (0 to n_bins-1)

    Example:
        For n_bins=4, low=0, high=1:
        - values < 0.25 → bin 0
        - 0.25 ≤ values < 0.5 → bin 1
        - 0.5 ≤ values < 0.75 → bin 2
        - 0.75 ≤ values → bin 3
    """
    # Ensure scalar inputs
    n_bins = int(n_bins)
    low = float(low)
    high = float(high)

    # Create bin edges
    bin_edges = np.linspace(low, high, n_bins + 1)

    # Use digitize to assign bins (subtract 1 to get 0-based indexing)
    bin_indices = np.digitize(control_array, bin_edges) - 1

    # Clamp to valid range [0, n_bins-1]
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bin_indices


def map_drum_to_groove_hit_ratio_to_categorical(hit_ratios):
    # check bottomn of the file for the bin calculation
    _10_bins = [1.149999976158142, 1.2666666507720947, 1.3333333730697632, 1.4137930870056152, 1.4800000190734863,
                1.5357142686843872, 1.615384578704834, 1.7142857313156128, 1.8666666746139526]

    categories = []
    for hit_ratio in hit_ratios:
        categories.append(map_value_to_bins(hit_ratio, _10_bins))
    return categories

def load_bz2_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param force_regenerate:
    :return:
    a list of hvo_sequences loaded from all the datasets specified in the json file
    """

    # load settings
    dataset_setting_json = json.load(open(dataset_setting_json_path, "r"))

    # load datasets
    dataset_tags = [key for key in dataset_setting_json["settings"].keys()]

    loaded_samples = []

    for dataset_tag in dataset_tags:
        dataLoaderLogger.info(f"Loading {dataset_tag} dataset")
        raw_data_pickle_path = dataset_setting_json["raw_data_pickle_path"][dataset_tag]

        for path_prepend in ["./", "../", "../../"]:
            if os.path.exists(path_prepend + raw_data_pickle_path):
                raw_data_pickle_path = path_prepend + raw_data_pickle_path
                break
        assert os.path.exists(raw_data_pickle_path), "path to gmd dict pickle is incorrect --- " \
                                                "look into data/***/storedDicts/groove-*.bz2pickle"

        dir__ = get_data_directory_using_filters(dataset_tag, dataset_setting_json_path)
        beat_division_factor = dataset_setting_json["global"]["beat_division_factor"]
        drum_mapping_label = dataset_setting_json["global"]["drum_mapping_label"]

        if (not os.path.exists(dir__)) or force_regenerate is True:
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> No Cached Version Available Here: {dir__}. ")
            dataLoaderLogger.info(
                f"extracting data from raw pickled midi/note_sequence/metadata dictionaries at {raw_data_pickle_path}")
            gmd_dict = load_original_gmd_dataset_pickle(raw_data_pickle_path)
            drum_mapping = get_drum_mapping_using_label(drum_mapping_label)
            hvo_dict = extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping)
            pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> Cached Version available at {dir__}")
        else:
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> Loading Cached Version from: {dir__}")

        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        data = pickle.load(ifile)
        ifile.close()
        loaded_samples.extend(data)

    return loaded_samples


def collect_train_set_info(dataset_setting_json_path_, num_voice_density_bins, num_global_density_bins, max_len=32):
    """

    :param dataset_setting_json_path_:
    :param num_voice_density_bins:
    :param num_global_density_bins:
    :return:
     (kick_low_bound, kick_up_bound), (snare_low_bound, snare_up_bound), (hat_low_bound, hat_up_bound),
        (tom_low_bound, tom_up_bound), (cymbal_low_bound, cymbal_up_bound),
        (global_density_low_bound, global_density_up_bound), (complexity_low_bound, complexity_up_bound), genre_tags
    """
    train_set_genre_tags = []
    train_set_complexities = []
    train_set_kick_counts = []
    train_set_snare_counts = []
    train_set_hat_counts = []
    train_set_tom_counts = []
    train_set_cymbal_counts = []
    train_set_total_hits = []
    train_set_hvo_files = []
    training_set_ = load_bz2_hvo_sequences(dataset_setting_json_path_, "train", force_regenerate=False)

    for ix, hvo_sample in enumerate(
            tqdm(training_set_,
                 desc="collecting genre tags and Per Voice Density Bins from corresponding full TRAINING set")):
        hits = hvo_sample.hits
        if hits is not None:
            train_set_hvo_files.append(hvo_sample.metadata["full_midi_filename"])
            hits = hvo_sample.hvo[:, :9]
            if hits.sum() > 0:
                hvo_sample.adjust_length(max_len)
                if hvo_sample.metadata["style_primary"] not in train_set_genre_tags:  # collect genre tags from training set
                    train_set_genre_tags.append(hvo_sample.metadata["style_primary"])
                train_set_complexities.append(
                    hvo_sample.get_complexity_surprisal()[0])  # collect complexity surprisal from training set
                train_set_total_hits.append(hits.sum())
                train_set_kick_counts.append(hits[:, 0].sum())
                train_set_snare_counts.append(hits[:, 1].sum())
                train_set_hat_counts.append(hits[:, 2:4].sum())
                train_set_tom_counts.append(hits[:, 4:7].sum())
                train_set_cymbal_counts.append(hits[:, 7:].sum())


    # get pervoice density bins
    return (get_bin_bounds_for_voice_densities(train_set_kick_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_snare_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_hat_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_tom_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_cymbal_counts, num_voice_density_bins),
            None,
            (min(train_set_complexities), max(train_set_complexities)), sorted(train_set_genre_tags),
            train_set_total_hits, train_set_hvo_files)

class MonotonicGrooveDataset(Dataset):
    def __init__(self, dataset_setting_json_path, subset_tag, max_len, tapped_voice_idx=2,
                 collapse_tapped_sequence=False, load_as_tensor=True, sort_by_metadata_key=None,
                 down_sampled_ratio=None, move_all_to_gpu=False,
                 hit_loss_balancing_beta=0, genre_loss_balancing_beta=0):
        """

        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param load_as_tensor:      [bool] loads the data as a tensor of torch.float32 instead of a numpy array
        :param sort_by_metadata_key: [str] sorts the data by the metadata key provided (e.g. "tempo")
        :param down_sampled_ratio: [float] down samples the data by the ratio provided (e.g. 0.5)
        :param move_all_to_gpu: [bool] moves all the data to the gpu
        :param hit_loss_balancing_beta: [float] beta parameter for hit balancing
                            (if 0 or very small, no hit balancing weights are returned)
        :param genre_loss_balancing_beta: [float] beta parameter for genre balancing
                            (if 0 or very small, no genre balancing weights are returned)
                hit_loss_balancing_beta and genre_balancing_beta are used to balance the data
                according to the hit and genre distributions of the dataset
                (reference: https://arxiv.org/pdf/1901.05555.pdf)
        """

        # Get processed inputs, outputs and hvo sequences
        self.inputs = list()
        self.outputs = list()
        self.hvo_sequences = list()

        # load pre-stored hvo_sequences or
        #   a portion of them uniformly sampled if down_sampled_ratio is provided
        # ------------------------------------------------------------------------------------------
        if down_sampled_ratio is None:
            subset = load_bz2_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)
        else:
            if "gmd" in json.load(open(dataset_setting_json_path, "r"))["settings"]:
                subset = load_downsampled_mega_set_hvo_sequences(
                    dataset_setting_json_path=dataset_setting_json_path,
                    subset_tag=subset_tag,
                    force_regenerate=False,
                    down_sampled_ratio=down_sampled_ratio,
                    cache_down_sampled_set=True
                )
            elif "mega_set" in json.load(open(dataset_setting_json_path, "r"))["settings"]:
                subset = load_downsampled_mega_set_hvo_sequences(
                    dataset_setting_json_path=dataset_setting_json_path,
                    subset_tag=subset_tag,
                    force_regenerate=False,
                    down_sampled_ratio=down_sampled_ratio,
                    cache_down_sampled_set=True
                )
            else:
                raise NotImplementedError

        # Sort data by a given metadata key if provided (e.g. "style_primary")
        # ------------------------------------------------------------------------------------------
        if sort_by_metadata_key:
            if sort_by_metadata_key in subset[0].metadata[sort_by_metadata_key]:
                subset = sorted(subset, key=lambda x: x.metadata[sort_by_metadata_key])

        # collect input tensors, output tensors, and hvo_sequences
        # ------------------------------------------------------------------------------------------
        for idx, hvo_seq in enumerate(tqdm(subset)):
            if hvo_seq.hits is not None:
                hvo_seq.adjust_length(max_len)
                if np.any(hvo_seq.hits):
                    # Ensure all have a length of max_len
                    self.hvo_sequences.append(hvo_seq)
                    self.outputs.append(hvo_seq.hvo)
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                    self.inputs.append(flat_seq)

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        # Get hit balancing weights if a beta parameter is provided
        # ------------------------------------------------------------------------------------------
        # get the effective number of hits per step and voice
        hits = self.outputs[:, :, :self.outputs.shape[-1] // 3]
        total_hits = hits.sum(0) + 1e-6
        effective_num_hits = 1.0 - np.power(hit_loss_balancing_beta, total_hits)
        hit_balancing_weights = (1.0 - hit_loss_balancing_beta) / effective_num_hits
        # normalize
        num_classes = hit_balancing_weights.shape[0] * hit_balancing_weights.shape[1]
        hit_balancing_weights = hit_balancing_weights / hit_balancing_weights.sum() * num_classes
        self.hit_balancing_weights_per_sample = [hit_balancing_weights for _ in range(len(self.outputs))]

        # Get genre balancing weights if a beta parameter is provided
        # ------------------------------------------------------------------------------------------
        # get the effective number of genres
        genres_per_sample = [sample.metadata["style_primary"] for sample in self.hvo_sequences]
        genre_counts = {genre: genres_per_sample.count(genre) for genre in set(genres_per_sample)}
        effective_num_genres = 1.0 - np.power(genre_loss_balancing_beta, list(genre_counts.values()))
        genre_balancing_weights = (1.0 - genre_loss_balancing_beta) / effective_num_genres
        # normalize
        genre_balancing_weights = genre_balancing_weights / genre_balancing_weights.sum() * len(genre_counts)
        genre_balancing_weights = {genre: weight for genre, weight in
                                   zip(genre_counts.keys(), genre_balancing_weights)}
        t_steps = self.outputs.shape[1]
        n_voices = self.outputs.shape[2] // 3
        temp_row = np.ones((t_steps, n_voices))
        self.genre_balancing_weights_per_sample = np.array(
            [temp_row * genre_balancing_weights[sample.metadata["style_primary"]]
             for sample in self.hvo_sequences])

        # Load as tensor if requested
        # ------------------------------------------------------------------------------------------
        if load_as_tensor or move_all_to_gpu:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
            self.outputs = torch.tensor(self.outputs, dtype=torch.float32)
            if hit_loss_balancing_beta is not None:
                self.hit_balancing_weights_per_sample = torch.tensor(self.hit_balancing_weights_per_sample,
                                                                     dtype=torch.float32)
            if genre_loss_balancing_beta is not None:
                self.genre_balancing_weights_per_sample = torch.tensor(self.genre_balancing_weights_per_sample,
                                                                       dtype=torch.float32)

        # Move to GPU if requested and GPU is available
        # ------------------------------------------------------------------------------------------
        if move_all_to_gpu and torch.cuda.is_available():
            self.inputs = self.inputs.to('cuda')
            self.outputs = self.outputs.to('cuda')
            if hit_loss_balancing_beta is not None:
                self.hit_balancing_weights_per_sample = self.hit_balancing_weights_per_sample.to('cuda')
            if genre_loss_balancing_beta is not None:
                self.genre_balancing_weights_per_sample = self.genre_balancing_weights_per_sample.to('cuda')

        dataLoaderLogger.info(f"Loaded {len(self.inputs)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], \
               self.hit_balancing_weights_per_sample[idx], self.genre_balancing_weights_per_sample[idx], idx

    def get_hvo_sequences_at(self, idx):
        return self.hvo_sequences[idx]

    def get_inputs_at(self, idx):
        return self.inputs[idx]

    def get_outputs_at(self, idx):
        return self.outputs[idx]






# ---------------------------------------------------------------------------------------------- #
# loading a down sampled dataset
# ---------------------------------------------------------------------------------------------- #

def down_sample_mega_dataset(hvo_seq_list, down_sample_ratio):
    """
    Down sample the dataset by a given ratio, it does it per style
    :param hvo_seq_list:
    :param down_sample_ratio:
    :return:
    """

    down_sampled_set = []

    per_style_hvos = dict()
    for hvo_seq in hvo_seq_list:
        style = hvo_seq.metadata["style_primary"]
        if style not in per_style_hvos:
            per_style_hvos[style] = []
        per_style_hvos[style].append(hvo_seq)

    total_down_sampled_size = ceil(len(hvo_seq_list) * down_sample_ratio)
    # ensure that equal number of samples are taken from each style
    per_style_size = int(total_down_sampled_size / len(per_style_hvos))

    for style, hvo_seqs in per_style_hvos.items():
        if per_style_size > 0:
            size_ = min(per_style_size, len(hvo_seqs))
            down_sampled_set.extend(random.sample(hvo_seqs, size_))

    return down_sampled_set


def down_sample_gmd_dataset(hvo_seq_list, down_sample_ratio):
    """
    Down sample the dataset by a given ratio, the ratio of the performers and the ratio of the performances
    are kept the same as much as possible.
    :param hvo_seq_list:
    :param down_sample_ratio:
    :return:
    """
    down_sampled_size = ceil(len(hvo_seq_list) * down_sample_ratio)

    # =================================================================================================
    # Divide up the performances by performer
    # =================================================================================================
    per_performer_per_performance_data = dict()
    for hs in tqdm(hvo_seq_list):
        performer = hs.metadata["drummer"]
        performance_id = hs.metadata["master_id"]
        if performer not in per_performer_per_performance_data:
            per_performer_per_performance_data[performer] = {}
        if performance_id not in per_performer_per_performance_data[performer]:
            per_performer_per_performance_data[performer][performance_id] = []
        per_performer_per_performance_data[performer][performance_id].append(hs)

    # =================================================================================================
    # Figure out how many loops to grab from each performer
    # =================================================================================================
    def flatten(l):
        if isinstance(l[0], list):
            return [item for sublist in l for item in sublist]
        else:
            return l

    ratios_to_other_performers = dict()

    # All samples per performer
    existing_sample_ratios_by_performer = dict()
    for performer, performances in per_performer_per_performance_data.items():
        existing_sample_ratios_by_performer[performer] = \
            len(flatten([performances[p] for p in performances])) / len(hvo_seq_list)

    new_samples_per_performer = dict()
    for performer, ratio in existing_sample_ratios_by_performer.items():
        samples = ceil(down_sampled_size * ratio)
        if samples > 0:
            new_samples_per_performer[performer] = samples

    # =================================================================================================
    # Figure out for each performer, how many samples to grab from each performance
    # =================================================================================================
    num_loops_from_each_performance_compiled_for_all_performers = dict()
    for performer, performances in per_performer_per_performance_data.items():
        total_samples = len(flatten([performances[p] for p in performances]))
        if performer in new_samples_per_performer:
            needed_samples = new_samples_per_performer[performer]
            num_loops_from_each_performance = dict()
            for performance_id, hs_list in performances.items():
                samples_to_select = ceil(needed_samples * len(hs_list) / total_samples)
                if samples_to_select > 0:
                    num_loops_from_each_performance[performance_id] = samples_to_select
            if num_loops_from_each_performance:
                num_loops_from_each_performance_compiled_for_all_performers[performer] = \
                    num_loops_from_each_performance

    # =================================================================================================
    # Sample required number of loops from each performance
    # THE SELECTION IS DONE BY RANKING THE TOTAL NUMBER OF HITS / TOTAL NUMBER OF VOICES ACTIVE
    # then selecting N equally spaced samples from the ranked list
    # =================================================================================================
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            seqs_sorted = sorted(
                hvo_seqs,
                key=lambda x: x.hits.sum() / x.get_number_of_active_voices(), reverse=True)
            indices = np.linspace(
                0,
                len(seqs_sorted) - 1,
                num_loops_from_each_performance_compiled_for_all_performers[performer][performance],
                dtype=int)
            per_performer_per_performance_data[performer][performance] = [seqs_sorted[i] for i in indices]

    downsampled_hvo_sequences = []
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            downsampled_hvo_sequences.extend(hvo_seqs)

    return downsampled_hvo_sequences


def up_sample_to_ensure_genre_balance(hvo_seq_list):
    """
    Upsamples the dataset to ensure genre balance. Repeats the samples from each genre to match the size of the largest
    genre.

    :param hvo_seq_list:
    :return:
    """
    hvo_seq_per_genre = dict()
    for hvo_seq in hvo_seq_list:
        genre = hvo_seq.metadata["style_primary"]
        if genre not in hvo_seq_per_genre:
            hvo_seq_per_genre[genre] = []
        hvo_seq_per_genre[genre].append(hvo_seq)

    max_genre_size = max([len(hvo_seq_per_genre[genre]) for genre in hvo_seq_per_genre.keys()])
    upsampled_hvo_sequences = []

    for genre, hvo_seqs in hvo_seq_per_genre.items():
        # get number of repeats
        num_repeats = ceil(max_genre_size / len(hvo_seqs))
        tmp = hvo_seqs * num_repeats
        upsampled_hvo_sequences.extend(tmp[:max_genre_size])

    return upsampled_hvo_sequences

def load_downsampled_mega_set_hvo_sequences(
        dataset_setting_json_path, subset_tag, down_sampled_ratio, cache_down_sampled_set=True, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param down_sampled_ratio: [float] the ratio of the dataset to downsample to
    :param cache_downsampled_set: [bool] whether to cache the down sampled dataset
    :param force_regenerate: [bool] if True, will regenerate the hvo_sequences from the raw data regardless of cache
    :return:
    """
    dataset_tag = "mega_set"
    dir__ = get_data_directory_using_filters(dataset_tag,
                                             dataset_setting_json_path,
                                             down_sampled_ratio=down_sampled_ratio)
    if (not os.path.exists(dir__)) or force_regenerate is True or cache_down_sampled_set is False:
        dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
        dataLoaderLogger.info(f"Downsampling the dataset to {down_sampled_ratio} and saving to {dir__}.")

        down_sampled_dict = {}
        for subset_tag in ["train", "validation", "test"]:
            hvo_seq_set = load_bz2_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False)
            if down_sampled_ratio is not None:
                down_sampled_dict.update({subset_tag: down_sample_mega_dataset(hvo_seq_set, down_sampled_ratio)})
            else:
                down_sampled_dict.update({subset_tag: hvo_seq_set})

        # collect and dump samples that match filter
        if cache_down_sampled_set:
            # create directories if needed
            if not os.path.exists(dir__):
                os.makedirs(dir__)
            for set_key_, set_data_ in down_sampled_dict.items():
                ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
                pickle.dump(set_data_, ofile)
                ofile.close()

        dataLoaderLogger.info(f"Loaded {len(down_sampled_dict[subset_tag])} {subset_tag} samples from {dir__}")
        return down_sampled_dict[subset_tag]
    else:
        dataLoaderLogger.info(f"load_downsampled_mega_set_hvo_sequences() -> Loading Cached Version from: {dir__}")
        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        set_data_ = pickle.load(ifile)
        ifile.close()
        dataLoaderLogger.info(f"Loaded {len(set_data_)} {subset_tag} samples from {dir__}")
        return set_data_

def upsample_to_ensure_genre_balance(dataset_setting_json_path, subset_tag, cache_upsampled_set=True, force_regenerate=False):
    dataset_tag = list(json.load(open(dataset_setting_json_path, "r"))["settings"].keys())[0]
    dir__ = get_data_directory_using_filters(dataset_tag,
                                             dataset_setting_json_path,
                                             up_sampled_ratio=1)
    if (not os.path.exists(dir__)) or force_regenerate is True or cache_upsampled_set is False:
        dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
        dataLoaderLogger.info(f"Upsampling the dataset to ensure genre balance and saving to {dir__}.")

        up_sampled_dict = {}
        for subset_tag in ["train", "validation", "test"]:
            hvo_seq_set = load_bz2_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False)
            up_sampled_dict.update({subset_tag: up_sample_to_ensure_genre_balance(hvo_seq_set)})

        # collect and dump samples that match filter
        if cache_upsampled_set:
            # create directories if needed
            if not os.path.exists(dir__):
                os.makedirs(dir__)
            for set_key_, set_data_ in up_sampled_dict.items():
                ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
                pickle.dump(set_data_, ofile)
                ofile.close()

        dataLoaderLogger.info(f"Loaded {len(up_sampled_dict[subset_tag])} {subset_tag} samples from {dir__}")
        return up_sampled_dict[subset_tag]
    else:
        dataLoaderLogger.info(f"upsample_to_ensure_genre_balance() -> Loading Cached Version from: {dir__}")
        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        set_data_ = pickle.load(ifile)
        ifile.close()
        dataLoaderLogger.info(f"Loaded {len(set_data_)} {subset_tag} samples from {dir__}")
        return set_data_


class Groove2Drum2BarDataset(Dataset):
    def __init__(self, dataset_setting_json_path, subset_tag, max_len, tapped_voice_idx=2,
                 collapse_tapped_sequence=False,
                 down_sampled_ratio=None, move_all_to_gpu=False,
                 augment_dataset=False,
                 use_cached=True,
                 num_voice_density_bins=None,
                 num_tempo_bins=None,
                 num_global_density_bins=None,
                 force_regenerate=False):

        """
        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param sort_by_metadata_key: [str] sorts the data by the metadata key provided (e.g. "tempo")
        :param down_sampled_ratio: [float] down samples the data by the ratio provided (e.g. 0.5)
        :param move_all_to_gpu: [bool] moves all the data to the gpu
        :param augment_dataset: [bool] if True, will augment the dataset by appending the bar-swapped version of each sequence
        :param use_cached: [bool] if True, will load the cached version of the dataset if available
        :param num_voice_density_bins: [int] number of bins to use for voice density (if None, will be set to 3)
        :param num_tempo_bins: [int] number of bins to use for tempo (if None, will be set to 6)
        :param num_global_density_bins: [int] number of bins to use for global density (if None, will be set to 8)
        :param force_regenerate: [bool] if True, will regenerate the cached version of the dataset
        """
        self.dataset_setting_json_path = dataset_setting_json_path

        if num_voice_density_bins is None:
            num_voice_density_bins = 3

        if num_tempo_bins is None:
            num_tempo_bins = 6

        if num_global_density_bins is None:
            num_global_density_bins = 8

        if down_sampled_ratio == 1:
            down_sampled_ratio = None

        def get_cached_filepath():
            dir_ = "cached/TorchDatasets"
            filename = (f"{dataset_setting_json_path.split('/')[-1]}_{subset_tag}_{max_len}_{tapped_voice_idx}"
                        f"_{collapse_tapped_sequence}_{down_sampled_ratio}_{augment_dataset}_{num_voice_density_bins}_{num_tempo_bins}_{num_global_density_bins}.bz2pickle")
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            return os.path.join(dir_, filename)

        # check if cached version exists
        # ------------------------------------------------------------------------------------------
        if use_cached and not force_regenerate:
            if os.path.exists(get_cached_filepath()):
                dataLoaderLogger.info(f"Groove2Drum2BarDataset Constructor --> Loading Cached Version from: {get_cached_filepath()}")
                ifile = bz2.BZ2File(get_cached_filepath(), 'rb')
                data = pickle.load(ifile)
                ifile.close()
                self.input_grooves = torch.tensor(data["input_grooves"], dtype=torch.float32)
                self.output_grooves = torch.tensor(data["output_grooves"], dtype=torch.float32)
                self.genre_encodings = torch.tensor(data["genre_encodings"], dtype=torch.float32)
                self.genre_mapping = data["genre_mapping"]
                self.genre_targets = torch.tensor(data["genre_targets"], dtype=torch.long)
                self.min_complexity = data["min_complexity"]
                self.max_complexity = data["max_complexity"]
                self.complexities = torch.tensor(data["complexities"], dtype=torch.float32)
                self.tempos = torch.tensor(data["tempos"], dtype=torch.long)
                self.num_tempo_bins = data["num_tempo_bins"]
                self.tempo_bins = torch.tensor(data["tempo_bins"], dtype=torch.long)
                self.hvo_sequences = data["hvo_sequences"]
                self.genre_tags = sorted(list(self.genre_mapping.keys()))
                self.num_genres = len(self.genre_tags) + 1
                self.kick_counts = data["kick_counts"]
                self.kick_density_bins = torch.tensor(data["kick_density_bins"], dtype=torch.long)
                self.kick_is_muted = torch.tensor(data["kick_is_muted"], dtype=torch.long)
                self.snare_counts = data["snare_counts"]
                self.snare_density_bins = torch.tensor(data["snare_density_bins"], dtype=torch.long)
                self.snare_is_muted = torch.tensor(data["snare_is_muted"], dtype=torch.long)
                self.hat_counts = data["hat_counts"]
                self.hat_density_bins = torch.tensor(data["hat_density_bins"], dtype=torch.long)
                self.hat_is_muted = torch.tensor(data["hat_is_muted"], dtype=torch.long)
                self.tom_counts = data["tom_counts"]
                self.tom_density_bins = torch.tensor(data["tom_density_bins"], dtype=torch.long)
                self.tom_is_muted = torch.tensor(data["tom_is_muted"], dtype=torch.long)
                self.cymbal_counts = data["cymbal_counts"]
                self.cymbal_density_bins = torch.tensor(data["cymbal_density_bins"], dtype=torch.long)
                self.cymbal_is_muted = torch.tensor(data["cymbal_is_muted"], dtype=torch.long)
                self.kick_lowBound_full_trnSet = data["kick_lowBound_full_trnSet"]
                self.kick_highBound_full_trnSet = data["kick_highBound_full_trnSet"]
                self.snare_lowBound_full_trnSet = data["snare_lowBound_full_trnSet"]
                self.snare_highBound_full_trnSet = data["snare_highBound_full_trnSet"]
                self.hat_lowBound_full_trnSet = data["hat_lowBound_full_trnSet"]
                self.hat_highBound_full_trnSet = data["hat_highBound_full_trnSet"]
                self.tom_lowBound_full_trnSet = data["tom_lowBound_full_trnSet"]
                self.tom_highBound_full_trnSet = data["tom_highBound_full_trnSet"]
                self.cymbal_lowBound_full_trnSet = data["cymbal_lowBound_full_trnSet"]
                self.cymbal_highBound_full_trnSet = data["cymbal_highBound_full_trnSet"]
                self.global_density_bins = torch.tensor(data["global_density_bins"], dtype=torch.long)
                process_data = False
            else:
                dataLoaderLogger.info(f"No Cached Version Available Here: {get_cached_filepath()}. ")
                process_data = True
        else:
            process_data = True

        if process_data:
            # Get processed inputs, outputs and hvo sequences
            self.input_grooves = list()
            self.output_grooves = list()
            self.complexities = list()
            self.genre_encodings = list()
            self.genre_targets = list()
            self.hvo_sequences = list()
            self.tempos = list()
            self.tempo_bins = list()

            self.kick_counts = list()               # voice 0
            self.snare_counts = list()              # voice 1
            self.hat_counts = list()                # voices 2, 3
            self.tom_counts = list()                # voices 4, 5, 6
            self.cymbal_counts = list()             # voices 7, 8
            self.hit_counts = list()                # all voices
            self.kick_is_muted = list()             # voice 0
            self.snare_is_muted = list()            # voice 1
            self.hat_is_muted = list()              # voices 2, 3
            self.tom_is_muted = list()              # voices 4, 5, 6
            self.cymbal_is_muted = list()           # voices 7, 8

            # up sample to ensure genre balance
            # ------------------------------------------------------------------------------------------
            # load pre-stored hvo_sequences or
            #   a portion of them uniformly sampled if down_sampled_ratio is provided
            # ------------------------------------------------------------------------------------------
            if down_sampled_ratio is None:
                subset = load_bz2_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False)
            else:
                subset = load_downsampled_mega_set_hvo_sequences(
                    dataset_setting_json_path=dataset_setting_json_path,
                    subset_tag=subset_tag,
                    force_regenerate=force_regenerate,
                    down_sampled_ratio=down_sampled_ratio,
                    cache_down_sampled_set=True
                )


            # collect genre map using the full TRAINING set only!!!
            # ------------------------------------------------------------------------------------------
            train_set_info = collect_train_set_info(dataset_setting_json_path,
                                                    num_voice_density_bins=num_voice_density_bins,
                                                    num_global_density_bins=num_global_density_bins,
                                                    max_len=max_len)
            self.kick_lowBound_full_trnSet, self.kick_highBound_full_trnSet = train_set_info[0]
            self.snare_lowBound_full_trnSet, self.snare_highBound_full_trnSet = train_set_info[1]
            self.hat_lowBound_full_trnSet, self.hat_highBound_full_trnSet = train_set_info[2]
            self.tom_lowBound_full_trnSet, self.tom_highBound_full_trnSet = train_set_info[3]
            self.cymbal_lowBound_full_trnSet, self.cymbal_highBound_full_trnSet = train_set_info[4]
            self.min_complexity, self.max_complexity = train_set_info[6]
            self.genre_tags = train_set_info[7]

            self.num_genres = len(self.genre_tags)
            # create a one-hot encoding for each genre
            self.genre_mapping = {genre: np.eye(self.num_genres)[idx].tolist() for idx, genre in enumerate(self.genre_tags)}

            # collect input tensors, output tensors, and hvo_sequences
            # ------------------------------------------------------------------------------------------
            for idx, hvo_seq in enumerate(tqdm(subset, desc="loading data and extracting input output tensors")):
                if hvo_seq.hits is not None:
                    hvo_seq.adjust_length(max_len)
                    if np.any(hvo_seq.hits):
                        # Ensure all have a length of max_len
                        self.hvo_sequences.append(hvo_seq)
                        self.output_grooves.append(hvo_seq.hvo)
                        flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx, reduce_dim=collapse_tapped_sequence)
                        self.input_grooves.append(flat_seq)
                        self.genre_encodings.append(self.map_genre_to_one_hot(hvo_seq.metadata["style_primary"]))
                        self.complexities.append(hvo_seq.get_complexity_surprisal()[0])
                        self.tempos.append(hvo_seq.tempos[0].qpm)
                        self.tempo_bins.append(map_tempo_to_categorical(hvo_seq.tempos[0].qpm))

                        hits = hvo_seq.hits
                        self.hit_counts.append(np.sum(hits))
                        self.kick_counts.append(hits[:, 0].sum())
                        self.snare_counts.append(hits[:, 1].sum())
                        self.hat_counts.append(hits[:, 2:4].sum())
                        self.tom_counts.append(hits[:, 4:7].sum())
                        self.cymbal_counts.append(hits[:, 7:].sum())

                        self.kick_is_muted.append(1 if self.kick_counts[-1] == 0 else 0)
                        self.snare_is_muted.append(1 if self.snare_counts[-1] == 0 else 0)
                        self.hat_is_muted.append(1 if self.hat_counts[-1] == 0 else 0)
                        self.tom_is_muted.append(1 if self.tom_counts[-1] == 0 else 0)
                        self.cymbal_is_muted.append(1 if self.cymbal_counts[-1] == 0 else 0)

            self.tempos = torch.tensor(np.array(self.tempos), dtype=torch.long)
            self.kick_density_bins = map_voice_densities_to_categorical(self.kick_counts,
                                                                        self.kick_lowBound_full_trnSet.copy(),
                                                                        self.kick_highBound_full_trnSet.copy())
            self.snare_density_bins = map_voice_densities_to_categorical(self.snare_counts,
                                                                         self.snare_lowBound_full_trnSet.copy(),
                                                                         self.snare_highBound_full_trnSet.copy())
            self.hat_density_bins = map_voice_densities_to_categorical(self.hat_counts,
                                                                       self.hat_lowBound_full_trnSet.copy(),
                                                                       self.hat_highBound_full_trnSet.copy())
            self.tom_density_bins = map_voice_densities_to_categorical(self.tom_counts,
                                                                       self.tom_lowBound_full_trnSet.copy(),
                                                                       self.tom_highBound_full_trnSet.copy())
            self.cymbal_density_bins = map_voice_densities_to_categorical(self.cymbal_counts,
                                                                          self.cymbal_lowBound_full_trnSet.copy(),
                                                                          self.cymbal_highBound_full_trnSet.copy())


            # Load as tensors
            self.input_grooves = torch.tensor(np.array(self.input_grooves), dtype=torch.float32)
            self.output_grooves = torch.tensor(np.array(self.output_grooves), dtype=torch.float32)
            self.genre_encodings = torch.tensor(np.array(self.genre_encodings), dtype=torch.float32)
            self.genre_targets = torch.argmax(self.genre_encodings, dim=1)

            groove_hit_counts = self.input_grooves[:, :, 0].sum(dim=1)
            output_groove_counts = self.output_grooves[:, :, :9].sum(dim=1).sum(dim=1)
            ratio = output_groove_counts / groove_hit_counts
            ratio = ratio.numpy()
            print(f"Max Ratio of output_groove_counts / groove_hit_counts: {ratio.max()}, shape: {ratio.shape}")
            # self.global_density_bins = map_global_density_to_categorical(self.hit_counts,
            #                                                              max_hits=max(self.hit_counts),
            #                                                              n_global_density_bins=num_global_density_bins)
            self.global_density_bins = torch.tensor(np.array(map_drum_to_groove_hit_ratio_to_categorical(ratio)), dtype=torch.long)
            self.complexities = torch.tensor(np.array(self.complexities), dtype=torch.float32)
            self.complexities = torch.clamp((self.complexities - self.min_complexity) / (self.max_complexity - self.min_complexity), 0, 1)

            self.tempo_bins = torch.tensor(np.array(self.tempo_bins), dtype=torch.long)
            self.kick_density_bins = torch.tensor(np.array(self.kick_density_bins), dtype=torch.long)
            self.snare_density_bins = torch.tensor(np.array(self.snare_density_bins), dtype=torch.long)
            self.hat_density_bins = torch.tensor(np.array(self.hat_density_bins), dtype=torch.long)
            self.tom_density_bins = torch.tensor(np.array(self.tom_density_bins), dtype=torch.long)
            self.cymbal_density_bins = torch.tensor(np.array(self.cymbal_density_bins), dtype=torch.long)
            self.global_density_bins = torch.tensor(np.array(self.global_density_bins), dtype=torch.long)

            self.kick_is_muted = torch.tensor(np.array(self.kick_is_muted), dtype=torch.long)
            self.snare_is_muted = torch.tensor(np.array(self.snare_is_muted), dtype=torch.long)
            self.hat_is_muted = torch.tensor(np.array(self.hat_is_muted), dtype=torch.long)
            self.tom_is_muted = torch.tensor(np.array(self.tom_is_muted), dtype=torch.long)
            self.cymbal_is_muted = torch.tensor(np.array(self.cymbal_is_muted), dtype=torch.long)

            if augment_dataset:
                # augment the dataset by appending the bar-swapped version of each sequence
                self.hvo_sequences = self.hvo_sequences * 2

                # inputs (N, 32, 27)
                # append with rotated 16 steps
                self.input_grooves = torch.cat([self.input_grooves, torch.roll(self.input_grooves, 16, dims=1)], dim=0)
                self.output_grooves = torch.cat([self.output_grooves, torch.roll(self.output_grooves, 16, dims=1)], dim=0)
                self.complexities = torch.cat([self.complexities, self.complexities], dim=0)
                self.genre_encodings = torch.cat([self.genre_encodings, self.genre_encodings], dim=0)
                self.genre_targets = torch.cat([self.genre_targets, self.genre_targets], dim=0)
                self.tempos = torch.cat([self.tempos, self.tempos], dim=0)
                self.tempo_bins = torch.cat([self.tempo_bins, self.tempo_bins], dim=0)
                self.kick_density_bins = torch.cat([self.kick_density_bins, self.kick_density_bins], dim=0)
                self.snare_density_bins = torch.cat([self.snare_density_bins, self.snare_density_bins], dim=0)
                self.hat_density_bins = torch.cat([self.hat_density_bins, self.hat_density_bins], dim=0)
                self.tom_density_bins = torch.cat([self.tom_density_bins, self.tom_density_bins], dim=0)
                self.cymbal_density_bins = torch.cat([self.cymbal_density_bins, self.cymbal_density_bins], dim=0)
                self.global_density_bins = torch.cat([self.global_density_bins, self.global_density_bins], dim=0)
                self.kick_is_muted = torch.cat([self.kick_is_muted, self.kick_is_muted], dim=0)
                self.snare_is_muted = torch.cat([self.snare_is_muted, self.snare_is_muted], dim=0)
                self.hat_is_muted = torch.cat([self.hat_is_muted, self.hat_is_muted], dim=0)
                self.tom_is_muted = torch.cat([self.tom_is_muted, self.tom_is_muted], dim=0)
                self.cymbal_is_muted = torch.cat([self.cymbal_is_muted, self.cymbal_is_muted], dim=0)



                # add the rolled groove to the second half of the hvo_sequences
                for idx, hvo_seq in enumerate(self.hvo_sequences[len(self.hvo_sequences) // 2:]):
                    hvo_seq.hvo = self.output_grooves[idx].numpy()

            # cache the processed data
            # ------------------------------------------------------------------------------------------
            if use_cached:
                data_to_dump = {
                    "input_grooves": self.input_grooves.numpy(),
                    "output_grooves": self.output_grooves.numpy(),
                    "genre_encodings": self.genre_encodings.numpy(),
                    "genre_mapping": self.genre_mapping,
                    "genre_targets": self.genre_targets.numpy(),
                    "min_complexity": self.min_complexity,
                    "max_complexity": self.max_complexity,
                    "complexities": self.complexities.numpy(),
                    "tempos": self.tempos.numpy(),
                    "tempo_bins": self.tempo_bins.numpy(),
                    "num_tempo_bins": num_tempo_bins,
                    "n_density_bins": num_voice_density_bins,  # "n_density_bins": "num_voice_density_bins",
                    "hvo_sequences": self.hvo_sequences,
                    "kick_counts": self.kick_counts,
                    "kick_density_bins": self.kick_density_bins.numpy(),
                    "snare_counts": self.snare_counts,
                    "snare_density_bins": self.snare_density_bins.numpy(),
                    "hat_counts": self.hat_counts,
                    "hat_density_bins": self.hat_density_bins.numpy(),
                    "tom_counts": self.tom_counts,
                    "tom_density_bins": self.tom_density_bins.numpy(),
                    "cymbal_counts": self.cymbal_counts,
                    "cymbal_density_bins": self.cymbal_density_bins.numpy(),
                    "kick_lowBound_full_trnSet": self.kick_lowBound_full_trnSet,
                    "kick_highBound_full_trnSet": self.kick_highBound_full_trnSet,
                    "snare_lowBound_full_trnSet": self.snare_lowBound_full_trnSet,
                    "snare_highBound_full_trnSet": self.snare_highBound_full_trnSet,
                    "hat_lowBound_full_trnSet": self.hat_lowBound_full_trnSet,
                    "hat_highBound_full_trnSet": self.hat_highBound_full_trnSet,
                    "tom_lowBound_full_trnSet": self.tom_lowBound_full_trnSet,
                    "tom_highBound_full_trnSet": self.tom_highBound_full_trnSet,
                    "cymbal_lowBound_full_trnSet": self.cymbal_lowBound_full_trnSet,
                    "cymbal_highBound_full_trnSet": self.cymbal_highBound_full_trnSet,
                    "global_density_bins": self.global_density_bins.numpy(),
                    "kick_is_muted": self.kick_is_muted.numpy(),
                    "snare_is_muted": self.snare_is_muted.numpy(),
                    "hat_is_muted": self.hat_is_muted.numpy(),
                    "tom_is_muted": self.tom_is_muted.numpy(),
                    "cymbal_is_muted": self.cymbal_is_muted.numpy()
                }
                ofile = bz2.BZ2File(get_cached_filepath(), 'wb')
                pickle.dump(data_to_dump, ofile)
                ofile.close()

        # Move to GPU if requested and GPU is available
        # ------------------------------------------------------------------------------------------
        if move_all_to_gpu and torch.cuda.is_available():
            self.input_grooves = self.input_grooves.to('cuda')
            self.output_grooves = self.output_grooves.to('cuda')
            # self.genre_encodings = self.genre_encodings.to('cuda')
            self.genre_targets = self.genre_targets.to('cuda')
            # self.complexities = self.complexities.to('cuda')
            # self.tempo_bins = self.tempo_bins.to('cuda')
            # self.kick_density_bins = self.kick_density_bins.to('cuda')
            # self.snare_density_bins = self.snare_density_bins.to('cuda')
            # self.hat_density_bins = self.hat_density_bins.to('cuda')
            # self.tom_density_bins = self.tom_density_bins.to('cuda')
            # self.cymbal_density_bins = self.cymbal_density_bins.to('cuda')
            self.global_density_bins = self.global_density_bins.to('cuda')
            self.kick_is_muted = self.kick_is_muted.to('cuda')
            self.snare_is_muted = self.snare_is_muted.to('cuda')
            self.hat_is_muted = self.hat_is_muted.to('cuda')
            self.tom_is_muted = self.tom_is_muted.to('cuda')
            self.cymbal_is_muted = self.cymbal_is_muted.to('cuda')

        dataLoaderLogger.info(f"Loaded {len(self.input_grooves)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return (self.input_grooves[idx],     # 0: input_groove
                self.complexities[idx],     # 1: complexity
                self.tempos[idx],           # 2: tempo (real value - not binned)
                self.genre_encodings[idx],  # 3: genre_encoding (one-hot)
                self.genre_targets[idx],    # 4: genre_target (categorical)
                self.output_grooves[idx],   # 5: output_groove (drum hvo)
                self.global_density_bins[idx],  # 6: global_density_bins
                self.tempo_bins[idx],       # 7: tempo_bin (categorical)
                self.kick_is_muted[idx],    # 8: kick_is_muted
                self.snare_is_muted[idx],   # 9: snare_is_muted
                self.hat_is_muted[idx],     # 10: hat_is_muted
                self.tom_is_muted[idx],     # 11: tom_is_muted
                self.cymbal_is_muted[idx])  # 12: cymbal_is_muted

    def __repr__(self):
        text =  f"    -------------------------------------\n"
        text += "Dataset Loaded using json file: \n"
        text += f"    {self.dataset_setting_json_path}\n"
        text += f"    ------------ __getitem__ returns a tuple of:\n"
        text += f"    input_groove: {self.input_grooves.shape}\n"
        text += f"    kick_density_bins: {self.kick_density_bins.shape}\n"
        text += f"    snare_density_bins: {self.snare_density_bins.shape}\n"
        text += f"    hat_density_bins: {self.hat_density_bins.shape}\n"
        text += f"    tom_density_bins: {self.tom_density_bins.shape}\n"
        text += f"    cymbal_density_bins: {self.cymbal_density_bins.shape}\n"
        text += f"    global_density_bins: {self.global_density_bins.shape}\n"
        text += f"    complexity: {self.complexities.shape}\n"
        text += f"    tempo: {self.tempos.shape}\n"
        text += f"    genre_encoding: {self.genre_encodings.shape}\n"
        text += f"    genre_target: {self.genre_targets.shape}\n"
        text += f"    output_groove: {self.output_grooves.shape}\n"
        text += f"    idx: {self.output_grooves.shape}\n"
        text += f"    ---------- Genre INFO ----------\n"
        text += f"    genre_mapping: {self.genre_mapping}\n"
        text += f"    num_genres: {self.num_genres}\n"
        text += f"    genre_tags: {self.genre_tags}\n"
        text += f"    -------- Complexity INFO -------\n"
        text += f"    min_complexity: {self.min_complexity}\n"
        text += f"    max_complexity: {self.max_complexity}\n"
        text += f"    -------- HVO INFO -------\n"
        text += f"    hvo_sequences: {len(self.hvo_sequences)}\n"
        text += f"    -------------------------------------\n"
        return text



    def get_hvo_sequences_at(self, idx):
        return self.hvo_sequences[idx]

    def get_inputs_at(self, idx):
        return self.input_grooves[idx]

    def get_outputs_at(self, idx):
        return self.output_grooves[idx]

    def get_genre_labels_for_all(self):
        return [self.genre_tags[int(tg_val)] for tg_val in self.genre_targets.detach().cpu().numpy()]

    def get_num_genres(self):
        return self.num_genres

    def get_genre_mapping(self):
        return self.genre_mapping

    def map_genre_to_one_hot(self, genre):
        # check if lowercase version of genre matches lowercase version of genre in genre_mapping
        for g in self.genre_mapping.keys():
            if genre.lower() == g.lower():
                return self.genre_mapping[g]

        assert False, f"Genre {genre} not found in genre_mapping"

    def get_genre_histogram(self):
        genre_histogram = {genre: 0 for genre in self.genre_mapping.keys()}
        for genre in self.genre_targets:
            genre_histogram[self.genre_tags[int(genre.item())]] += 1
        return genre_histogram

    def get_complexities_per_genre(self):
        '''
        Returns the complexities per genre
        :return:
        '''

        complexity_histogram_per_genre = {genre: [] for genre in self.genre_mapping.keys()}
        complexity_histogram_per_genre["all"] = []

        for idx, hvo_seq in enumerate(self.hvo_sequences):
            genre = self.genre_targets[idx]
            complexity_histogram_per_genre[self.genre_tags[int(genre.item())]].append(self.complexities[idx].item())

        return complexity_histogram_per_genre

    def get_tempos_per_genre(self):
        '''
        Returns the tempos per genre
        :return:
        '''

        tempo_histogram_per_genre = {genre: [] for genre in self.genre_mapping.keys()}
        tempo_histogram_per_genre["all"] = []

        for idx, hvo_seq in enumerate(self.hvo_sequences):
            genre = self.genre_targets[idx]
            tempo_histogram_per_genre[self.genre_tags[int(genre.item())]].append(self.tempos[idx].item())
            tempo_histogram_per_genre["all"].append(self.tempos[idx].item())

        return tempo_histogram_per_genre


class Groove2TripleStreams2BarDataset(Dataset):
    def __init__(self,
                 config,
                 subset_tag,            # pass "train" or "validation" or "test"
                 use_cached=True,
                 down_sampled_size=None,
                 force_regenerate=False):

        """
        :param dataset_setting_json_path:   path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
        :param subset_tag:                [str] whether to load the train/test/validation set
        :param max_len:              [int] maximum length of the sequences to be loaded
        :param tapped_voice_idx:    [int] index of the voice to be tapped (default is 2 which is usually closed hat)
        :param collapse_tapped_sequence:  [bool] returns a Tx3 tensor instead of a Tx(3xNumVoices) tensor
        :param sort_by_metadata_key: [str] sorts the data by the metadata key provided (e.g. "tempo")
        :param down_sampled_ratio: [float] down samples the data by the ratio provided (e.g. 0.5)
        :param move_all_to_gpu: [bool] moves all the data to the gpu
        :param augment_dataset: [bool] if True, will augment the dataset by appending the bar-swapped version of each sequence
        :param use_cached: [bool] if True, will load the cached version of the dataset if available
        :param num_voice_density_bins: [int] number of bins to use for voice density (if None, will be set to 3)
        :param num_tempo_bins: [int] number of bins to use for tempo (if None, will be set to 6)
        :param num_global_density_bins: [int] number of bins to use for global density (if None, will be set to 8)
        :param force_regenerate: [bool] if True, will regenerate the cached version of the dataset
        """

        if down_sampled_size is not None:
            if down_sampled_size >= 1:
                down_sampled_size = None

        self.dataset_setting_json_path = config["dataset_setting_json_path"]
        self.subset_tag = subset_tag
        self.max_len = config["max_len"]
        self.n_encoding_control1_tokens = config["n_encoding_control1_tokens"]
        self.encoding_control1_key = config["encoding_control1_key"]
        self.n_encoding_control2_tokens = config["n_encoding_control2_tokens"]
        self.encoding_control2_key = config["encoding_control2_key"]
        self.n_decoding_control1_tokens = config["n_decoding_control1_tokens"]
        self.decoding_control1_key = config["decoding_control1_key"]
        self.n_decoding_control2_tokens = config["n_decoding_control2_tokens"]
        self.decoding_control2_key = config["decoding_control2_key"]
        self.n_decoding_control3_tokens = config["n_decoding_control3_tokens"]
        self.decoding_control3_key = config["decoding_control3_key"]
        self.down_sampled_size = down_sampled_size
        with open(self.dataset_setting_json_path, "r") as f:
            self.json = json.load(f)

        def get_source_compiled_data_dictionary_path():
            return os.path.join(self.json["compiled_pkl_bz2_path"], self.subset_tag + ".pkl.bz2")

        def get_cached_filepath():
            dir_ = os.path.join("cached/TorchDatasets", self.json["tag"], self.subset_tag)
            os.makedirs(dir_, exist_ok=True)
            filename = f"{self.subset_tag}_{self.max_len}_{self.down_sampled_size}" \
            f"_{self.n_encoding_control1_tokens}_{self.encoding_control1_key}" \
            f"_{self.n_encoding_control2_tokens}_{self.encoding_control2_key}" \
            f"_{self.n_decoding_control1_tokens}_{self.decoding_control1_key}" \
            f"_{self.n_decoding_control2_tokens}_{self.decoding_control2_key}" \
            f"_{self.n_decoding_control3_tokens}_{self.decoding_control3_key}"

            filename = hash_string_to_name(filename) + ".bz2pickle"
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            return os.path.join(dir_, filename)

        # check if cached version exists
        # ------------------------------------------------------------------------------------------
        process_data = True

        if use_cached and not force_regenerate:
            if os.path.exists(get_cached_filepath()):
                dataLoaderLogger.info(f"Groove2TripleStreams2BarDataset Constructor --> Loading Cached Version from: {get_cached_filepath()}")
                ifile = bz2.BZ2File(get_cached_filepath(), 'rb')
                data = pickle.load(ifile)
                ifile.close()

                self.input_grooves = torch.tensor(data["input_grooves"], dtype=torch.float32)
                self.output_streams = torch.tensor(data["output_streams"], dtype=torch.float32)
                self.flat_output_streams = torch.tensor(data["flat_output_streams"], dtype=torch.float32)
                self.encoding_control1_tokens = torch.tensor(data["encoding_control1_tokens"], dtype=torch.long)
                self.encoding_control2_tokens = torch.tensor(data["encoding_control2_tokens"], dtype=torch.long)
                self.decoding_control1_tokens = torch.tensor(data["decoding_control1_tokens"], dtype=torch.long)
                self.decoding_control2_tokens = torch.tensor(data["decoding_control2_tokens"], dtype=torch.long)
                self.decoding_control3_tokens = torch.tensor(data["decoding_control3_tokens"], dtype=torch.long)
                self.metadata = data["metadata"]
                self.tempos = data["tempos"]
                self.collection = data["collection"]

                process_data = False

        if process_data:
            # ------------------------------------------------------------------------------------------
            # load pre-stored hvo_sequences or
            #   a portion of them uniformly sampled if down_sampled_ratio is provided
            # ------------------------------------------------------------------------------------------
            print("\n\n LOADING THE DICTIONARY CONTAINING RAW DATA")
            loaded_data_dictionary = load_pkl_bz2_dict(get_source_compiled_data_dictionary_path(), allow_pickle_arrays=True)

            # check if only a subset of the data is needed
            if self.down_sampled_size is not None:
                n_samples = len(loaded_data_dictionary[loaded_data_dictionary.keys()[0]])
                sampled_indices = np.random.choice(n_samples, self.down_sampled_size, replace=False)
                for k, v in loaded_data_dictionary.items():
                    loaded_data_dictionary[k] = v[sampled_indices]


            # Populate already available fields
            # ------------------------------------------------------------------------------------------
            self.input_grooves = np.array(loaded_data_dictionary["input_hvos"])
            self.output_streams = np.array(loaded_data_dictionary["output_hvos"])
            self.flat_output_streams = np.array(loaded_data_dictionary["flat_out_hvos"])
            self.metadata = loaded_data_dictionary["metadata"]
            self.tempos = loaded_data_dictionary["qpm"]
            self.collection = loaded_data_dictionary["collection"]

            # Populate control tokens
            # ------------------------------------------------------------------------------------------

            self.encoding_control1_tokens = TokenizeControls(
                control_array=np.round(loaded_data_dictionary[self.encoding_control1_key], 5),
                n_bins=self.n_encoding_control1_tokens, low=0, high=1)
            self.encoding_control2_tokens = TokenizeControls(
                control_array=np.round(loaded_data_dictionary[self.encoding_control2_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control1_tokens = TokenizeControls(
                control_array=np.round(loaded_data_dictionary[self.decoding_control1_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control2_tokens = TokenizeControls(
                control_array=np.round(loaded_data_dictionary[self.decoding_control2_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control3_tokens = TokenizeControls(
                control_array=np.round(loaded_data_dictionary[self.decoding_control3_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)

            # cache the processed data
            # ------------------------------------------------------------------------------------------
            if use_cached:
                dataLoaderLogger.info(f"Caching at {get_cached_filepath()}")
                data_to_dump = {
                    "input_grooves": self.input_grooves,
                    "output_streams": self.output_streams,
                    "flat_output_streams": self.flat_output_streams,
                    "metadata": self.metadata,
                    "tempos": self.tempos,
                    "collection": self.collection,
                    "encoding_control1_tokens": self.encoding_control1_tokens,
                    "encoding_control2_tokens": self.encoding_control2_tokens,
                    "decoding_control1_tokens": self.decoding_control1_tokens,
                    "decoding_control2_tokens": self.decoding_control2_tokens,
                    "decoding_control3_tokens": self.decoding_control3_tokens
                }

                ofile = bz2.BZ2File(get_cached_filepath(), 'wb')
                pickle.dump(data_to_dump, ofile)
                ofile.close()

                # Conver to tensors (patterns as float32 and controls as long)
                # ------------------------------------------------------------------------------------------
                self.input_grooves = torch.tensor(self.input_grooves, dtype=torch.float32)
                self.output_streams = torch.tensor(self.output_streams, dtype=torch.float32)
                self.flat_output_streams = torch.tensor(self.flat_output_streams, dtype=torch.float32)
                self.encoding_control1_tokens = torch.tensor(self.encoding_control1_tokens, dtype=torch.long)
                self.encoding_control2_tokens = torch.tensor(self.encoding_control2_tokens, dtype=torch.long)
                self.decoding_control1_tokens = torch.tensor(self.decoding_control1_tokens, dtype=torch.long)
                self.decoding_control2_tokens = torch.tensor(self.decoding_control2_tokens, dtype=torch.long)
                self.decoding_control3_tokens = torch.tensor(self.decoding_control3_tokens, dtype=torch.long)

        dataLoaderLogger.info(f"Loaded {len(self.input_grooves)} sequences")

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return (self.input_grooves[idx],
                self.output_streams[idx],
                self.encoding_control1_tokens[idx],
                self.encoding_control2_tokens[idx],
                self.decoding_control1_tokens[idx],
                self.decoding_control2_tokens[idx],
                self.decoding_control3_tokens[idx],
                idx
                )

    def __repr__(self):
        text =  f"    -------------------------------------\n"
        text += "Dataset Loaded using json file: \n"
        text += f"    {self.dataset_setting_json_path}\n"
        return text



if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run demos/data/demo.py to test")


    #
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset
    from data import Groove2TripleStreams2BarDataset

    config = {
        'dataset_setting_json_path': "data/dataset_json_settings/TripleStreams0_75_Accent.json",

        'encoding_control1_key': "Flat Out Vs. Input | Hits | Hamming",
        'encoding_control2_key': "Flat Out Vs. Input | Accent | Hamming",
        'decoding_control1_key': "Stream 1 Vs. Flat Out | Hits | Hamming",
        'decoding_control2_key': "Stream 2 Vs. Flat Out | Hits | Hamming",
        'decoding_control3_key': "Stream 3 Vs. Flat Out | Hits | Hamming",

        'n_encoding_control1_tokens': 33,
        'n_encoding_control2_tokens': 10,
        'n_decoding_control1_tokens': 12,
        'n_decoding_control2_tokens': 12,
        'n_decoding_control3_tokens': 12,

        'd_model': 128,
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

        'device': 'cpu'
    }

    # load dataset as torch.utils.data.Dataset
    training_dataset = Groove2TripleStreams2BarDataset(
        config=config,
        subset_tag="train",
        use_cached=True,
        force_regenerate=False
    )

    from model import TripleStreamsVAE

    '''
    return (input_grooves,     
                output_streams,     
                flat_output_streams,           
                encoding_control1_tokens,
                encoding_control2_tokens,
                decoding_control1_tokens,
                decoding_control2_tokens,
                decoding_control3_tokens,
                tempos,
                collection,
                self.metadata
                )
                '''

    model = TripleStreamsVAE(config)

    import tqdm
    for i in tqdm.trange(1000):
        start = i * 100
        end = (i + 1) * 100

        batch_data = training_dataset[start:end]
        input_grooves = batch_data[0]
        output_streams = batch_data[1]
        encoding_control1_tokens = batch_data[2]
        encoding_control2_tokens = batch_data[3]
        decoding_control1_tokens = batch_data[4]
        decoding_control2_tokens = batch_data[5]
        decoding_control3_tokens = batch_data[6]
        indices = batch_data[7]

        model.forward(
            flat_hvo_groove=input_grooves,
            encoding_control1_token=encoding_control1_tokens,
            encoding_control2_token=encoding_control2_tokens,
            decoding_control1_token=decoding_control1_tokens,
            decoding_control2_token=decoding_control2_tokens,
            decoding_control3_token=decoding_control3_tokens,
        )
