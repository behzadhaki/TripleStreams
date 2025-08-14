from data.src.utils import (get_data_directory_using_filters, get_drum_mapping_using_label,
                            load_original_gmd_dataset_pickle, extract_hvo_sequences_dict, pickle_hvo_dict)
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
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
dataLoaderLogger = logging.getLogger("dloader")



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


# ---------------------------------------------------------------------------------------------- #
# loading a down sampled dataset
# ---------------------------------------------------------------------------------------------- #

class Groove2TripleStream2BarDataset(Dataset):

    def __init__(self,
                 config,
                 subset_tag,            # pass "train" or "validation" or "test"
                 use_cached=True,
                 downsampled_size=None,
                 force_regenerate=False,
                 move_all_to_cuda=False):

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



        self.dataset_root_path = config["dataset_root_path"]
        self.dataset_files = config["dataset_files"]
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
        # with open(self.dataset_setting_json_path, "r") as f:
        #     self.json = json.load(f)

        def get_source_compiled_data_dictionary_path():
            return os.path.join(self.dataset_root_path, self.subset_tag)

        def get_cached_filepath():
            dir_ = os.path.join("cached/TorchDatasets/", self.dataset_root_path.replace("/", "-"), self.subset_tag)
            os.makedirs(dir_, exist_ok=True)
            filename = "".join([df.split("_")[0] for df in self.dataset_files])
            filename += f"_{self.max_len}{downsampled_size}" \
            f"_{self.n_encoding_control1_tokens}{self.n_encoding_control2_tokens}{self.n_decoding_control1_tokens}{self.n_decoding_control2_tokens}{self.n_decoding_control3_tokens}"
            filename = filename.replace(" ", "_").replace("|", "_").replace("/", "_").replace("\\", "_").replace("__", "_").replace("__", "_")

            filename = filename + ".bz2pickle"
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            return os.path.join(dir_, filename)

        # check if cached version exists
        # ------------------------------------------------------------------------------------------
        process_data = True

        if use_cached and not force_regenerate:
            if os.path.exists(get_cached_filepath()):
                # dataLoaderLogger.info(f"Groove2TripleStreams2BarDataset Constructor --> Loading Cached Version from: {get_cached_filepath()}")
                ifile = bz2.BZ2File(get_cached_filepath(), 'rb')
                data = pickle.load(ifile)
                ifile.close()

                self.input_grooves = data["input_grooves"]
                self.output_streams = data["output_streams"]
                self.flat_output_streams = data["flat_output_streams"]
                self.encoding_control1_tokens = data["encoding_control1_tokens"]
                self.encoding_control2_tokens = data["encoding_control2_tokens"]
                self.decoding_control1_tokens = data["decoding_control1_tokens"]
                self.decoding_control2_tokens = data["decoding_control2_tokens"]
                self.decoding_control3_tokens = data["decoding_control3_tokens"]
                self.metadata = data["metadata"]
                self.tempos = data["tempos"]
                self.collection = data["collection"]

                process_data = False

        if process_data:
            # ------------------------------------------------------------------------------------------
            # load pre-stored hvo_sequences or
            #   a portion of them uniformly sampled if down_sampled_ratio is provided
            # ------------------------------------------------------------------------------------------
            n_samples = 0
            loaded_data_dictionary = {}

            pbar = tqdm.tqdm(self.dataset_files, desc="Loading data files") if len(self.dataset_files) > 1 else self.dataset_files
            for dataset_file in pbar:
                if not isinstance(pbar, list):
                    # Update description with current filename
                    pbar.set_description(f"Loading: {dataset_file}")
                temp = load_pkl_bz2_dict(
                    os.path.join(get_source_compiled_data_dictionary_path(), dataset_file),
                    allow_pickle_arrays=True)
                for k, v in temp.items():
                    if k not in loaded_data_dictionary:
                        loaded_data_dictionary[k] = []
                    if isinstance(v, np.ndarray):
                        loaded_data_dictionary[k].extend(v.tolist())
                    else:
                        loaded_data_dictionary[k].extend(v)
                n_samples += len(temp["metadata"])
            
            if downsampled_size is not None:
                if downsampled_size >= n_samples:
                    downsampled_size = None
                else:
                    downsampled_size = downsampled_size
            else:
                downsampled_size = downsampled_size

            # check if only a subset of the data is needed
            if downsampled_size is not None:
                sampled_indices = np.random.choice(n_samples, downsampled_size, replace=False)
                # dataLoaderLogger.info(f"Downsizing by selecting {downsampled_size} from {n_samples} samples")
                for k, v in loaded_data_dictionary.items():
                    loaded_data_dictionary[k] = [v[ix] for ix in sampled_indices]


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
                # dataLoaderLogger.info(f"Caching at {get_cached_filepath()}")
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

        # Safety checks
        # ------------------------------------------------------------------------------------------
        def get_invalid_indices(hvo):
            n_voices = hvo.shape[-1] // 3
            hits = hvo[:, :, :n_voices]
            velocities = hvo[:, :, n_voices:2*n_voices]
            offsets = hvo[:, :, 2*n_voices:3*n_voices]
            h_invalid_sample_ix, _, _ = np.where((hits > 1) | (hits < 0.0))
            v_invalid_sample_ix, _, _ = np.where((velocities > 1) | (velocities < 0.0))
            o_invalid_sample_ix, _, _ = np.where((offsets > 0.5) | (offsets < -0.5))
            # get the union of all invalid sample indices
            invalid_sample_ix = set(h_invalid_sample_ix).union(set(v_invalid_sample_ix)).union(set(o_invalid_sample_ix))
            return invalid_sample_ix

        invalid_indices_input = get_invalid_indices(self.input_grooves)
        invalid_indices_output = get_invalid_indices(self.output_streams)
        all_invalid_indices = set(invalid_indices_input).union(set(invalid_indices_output))

        # remove invalid samples
        if len(all_invalid_indices) > 0:
            print(f"Found {len(all_invalid_indices)} invalid samples in input grooves. Removing them.")
            print("Size before removing invalid samples: ", self.input_grooves.shape[0])
            self.input_grooves = np.delete(self.input_grooves, list(all_invalid_indices), axis=0)
            self.output_streams = np.delete(self.output_streams, list(all_invalid_indices), axis=0)
            self.flat_output_streams = np.delete(self.flat_output_streams, list(all_invalid_indices), axis=0)
            self.encoding_control1_tokens = np.delete(self.encoding_control1_tokens, list(all_invalid_indices), axis=0)
            self.encoding_control2_tokens = np.delete(self.encoding_control2_tokens, list(all_invalid_indices), axis=0)
            self.decoding_control1_tokens = np.delete(self.decoding_control1_tokens, list(all_invalid_indices), axis=0)
            self.decoding_control2_tokens = np.delete(self.decoding_control2_tokens, list(all_invalid_indices), axis=0)
            self.decoding_control3_tokens = np.delete(self.decoding_control3_tokens, list(all_invalid_indices), axis=0)
            self.metadata = [self.metadata[ix] for ix in range(len(self.metadata)) if ix not in all_invalid_indices]
            self.tempos = [self.tempos[ix] for ix in range(len(self.tempos)) if ix not in all_invalid_indices]
            self.collection = [self.collection[ix] for ix in range(len(self.collection)) if ix not in all_invalid_indices]
            print("Size after removing invalid samples: ", self.input_grooves.shape[0])

        # Convert to tensors (patterns as float32 and controls as long)
        # ------------------------------------------------------------------------------------------
        self.indices = list(range(len(self.input_grooves)))
        self.input_grooves = torch.tensor(self.input_grooves, dtype=torch.float32)
        self.output_streams = torch.tensor(self.output_streams, dtype=torch.float32)
        self.flat_output_streams = torch.tensor(self.flat_output_streams, dtype=torch.float32)
        self.encoding_control1_tokens = torch.tensor(self.encoding_control1_tokens, dtype=torch.long)
        self.encoding_control2_tokens = torch.tensor(self.encoding_control2_tokens, dtype=torch.long)
        self.decoding_control1_tokens = torch.tensor(self.decoding_control1_tokens, dtype=torch.long)
        self.decoding_control2_tokens = torch.tensor(self.decoding_control2_tokens, dtype=torch.long)
        self.decoding_control3_tokens = torch.tensor(self.decoding_control3_tokens, dtype=torch.long)
        
        # move_all_to_cuda 
        # ------------------------------------------------------------------------------------------
        if move_all_to_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            self.input_grooves = self.input_grooves.to(device)
            self.output_streams = self.output_streams.to(device)
            self.flat_output_streams = self.flat_output_streams.to(device)
            self.encoding_control1_tokens = self.encoding_control1_tokens.to(device)
            self.encoding_control2_tokens = self.encoding_control2_tokens.to(device)
            self.decoding_control1_tokens = self.decoding_control1_tokens.to(device)
            self.decoding_control2_tokens = self.decoding_control2_tokens.to(device)
            self.decoding_control3_tokens = self.decoding_control3_tokens.to(device)
            
        self.indices = list(range(len(self.metadata)))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return (self.input_grooves[idx],
                self.output_streams[idx],
                self.encoding_control1_tokens[idx],
                self.encoding_control2_tokens[idx],
                self.decoding_control1_tokens[idx],
                self.decoding_control2_tokens[idx],
                self.decoding_control3_tokens[idx],
                self.metadata[idx],
                self.indices[idx]
                )

    def __repr__(self):
        text =  f"    -------------------------------------\n"
        text += "Dataset Loaded using json file: \n"
        text += f"    {self.dataset_setting_json_path}\n"
        return text

    @classmethod
    def from_concatenated_datasets(cls,
                                   config,
                                   subset_tag,
                                   use_cached=True,
                                   downsampled_size=None,
                                   force_regenerate=False,
                                   move_all_to_cuda=False):
        """
        Alternative constructor that loads multiple dataset files and concatenates them
        into a single dataset instance, avoiding ConcatDataset issues.

        This preserves device location (GPU/CPU) of tensors after concatenation.
        Individual datasets are cached, but the concatenated result is not cached.
        """

        dataLoaderLogger.info(f"Creating concatenated dataset from {len(config['dataset_files'])} files")

        individual_downsampled_size = int(
            downsampled_size / len(config["dataset_files"])) if downsampled_size is not None else None

        # Load all individual datasets (these can be cached)
        datasets = []
        pbar = tqdm.tqdm(config["dataset_files"], desc="Loading dataset files for concatenation")
        for dataset_file in pbar:
            pbar.set_description(f"Loading: {dataset_file}")
            new_config = config.copy()
            new_config["dataset_files"] = [dataset_file]

            dataset = cls(
                config=new_config,
                subset_tag=subset_tag,
                use_cached=use_cached,
                downsampled_size=individual_downsampled_size,
                force_regenerate=force_regenerate,
                move_all_to_cuda=False  # Don't move individual datasets to CUDA yet
            )
            datasets.append(dataset)


        common_keys = set.intersection(*(set(ds.metadata[0].keys()) for ds in datasets))
        dataLoaderLogger.info(f"Loaded {len(datasets)} datasets:")
        for ix, ds in enumerate(datasets):
            ds.metadata = [{k: v for k, v in sample.items() if k in common_keys} for sample in ds.metadata]
            dataLoaderLogger.info(f"\t {ix} : {config['dataset_files'][ix]} --> {len(ds)} samples")

        # Concatenate all data
        dataLoaderLogger.info("Concatenating datasets...")

        # Concatenate tensors
        input_grooves = torch.cat([ds.input_grooves for ds in datasets], dim=0)
        output_streams = torch.cat([ds.output_streams for ds in datasets], dim=0)
        flat_output_streams = torch.cat([ds.flat_output_streams for ds in datasets], dim=0)
        encoding_control1_tokens = torch.cat([ds.encoding_control1_tokens for ds in datasets], dim=0)
        encoding_control2_tokens = torch.cat([ds.encoding_control2_tokens for ds in datasets], dim=0)
        decoding_control1_tokens = torch.cat([ds.decoding_control1_tokens for ds in datasets], dim=0)
        decoding_control2_tokens = torch.cat([ds.decoding_control2_tokens for ds in datasets], dim=0)
        decoding_control3_tokens = torch.cat([ds.decoding_control3_tokens for ds in datasets], dim=0)

        # Concatenate lists (metadata already cleaned above)
        metadata = []
        tempos = []
        collection = []
        for ds in datasets:
            metadata.extend(ds.metadata)
            tempos.extend(ds.tempos)
            collection.extend(ds.collection)

        # Create new instance
        instance = cls.__new__(cls)
        instance.dataset_root_path = config["dataset_root_path"]
        instance.dataset_files = config["dataset_files"]
        instance.subset_tag = subset_tag
        instance.max_len = config["max_len"]
        instance.n_encoding_control1_tokens = config["n_encoding_control1_tokens"]
        instance.encoding_control1_key = config["encoding_control1_key"]
        instance.n_encoding_control2_tokens = config["n_encoding_control2_tokens"]
        instance.encoding_control2_key = config["encoding_control2_key"]
        instance.n_decoding_control1_tokens = config["n_decoding_control1_tokens"]
        instance.decoding_control1_key = config["decoding_control1_key"]
        instance.n_decoding_control2_tokens = config["n_decoding_control2_tokens"]
        instance.decoding_control2_key = config["decoding_control2_key"]
        instance.n_decoding_control3_tokens = config["n_decoding_control3_tokens"]
        instance.decoding_control3_key = config["decoding_control3_key"]

        # Assign concatenated data
        instance.input_grooves = input_grooves
        instance.output_streams = output_streams
        instance.flat_output_streams = flat_output_streams
        instance.encoding_control1_tokens = encoding_control1_tokens
        instance.encoding_control2_tokens = encoding_control2_tokens
        instance.decoding_control1_tokens = decoding_control1_tokens
        instance.decoding_control2_tokens = decoding_control2_tokens
        instance.decoding_control3_tokens = decoding_control3_tokens
        instance.metadata = metadata
        instance.tempos = tempos
        instance.collection = collection
        instance.indices = list(range(len(metadata)))

        # Move to GPU (CUDA/MPS) if requested
        if move_all_to_cuda:
            # Try CUDA first, then MPS, fallback to CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                dataLoaderLogger.info("Moving concatenated dataset to CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                dataLoaderLogger.info("Moving concatenated dataset to MPS")
            else:
                device = torch.device("cpu")
                dataLoaderLogger.warning("CUDA and MPS not available, keeping dataset on CPU")

            instance.input_grooves = instance.input_grooves.to(device)
            instance.output_streams = instance.output_streams.to(device)
            instance.flat_output_streams = instance.flat_output_streams.to(device)
            instance.encoding_control1_tokens = instance.encoding_control1_tokens.to(device)
            instance.encoding_control2_tokens = instance.encoding_control2_tokens.to(device)
            instance.decoding_control1_tokens = instance.decoding_control1_tokens.to(device)
            instance.decoding_control2_tokens = instance.decoding_control2_tokens.to(device)
            instance.decoding_control3_tokens = instance.decoding_control3_tokens.to(device)

        dataLoaderLogger.info(f"Concatenated dataset created with {len(instance)} samples")
        return instance

def get_triplestream_dataset(
        config,
        subset_tag,  # pass "train" or "validation" or "test"
        use_cached=True,
        downsampled_size=None,
        force_regenerate=False,
        move_all_to_cuda=False):
    """
    Alternative to get_triplestream_dataset that returns a single concatenated dataset
    instead of using ConcatDataset, preserving device location.
    """

    try:
        cfg_dict = config.as_dict()
    except:
        cfg_dict = config

    return Groove2TripleStream2BarDataset.from_concatenated_datasets(
        config=cfg_dict,
        subset_tag=subset_tag,
        use_cached=use_cached,
        downsampled_size=downsampled_size,
        force_regenerate=force_regenerate,
        move_all_to_cuda=move_all_to_cuda
    )

if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run demos/data/demo.py to test")


    #
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset

    import yaml
    from data import Groove2TripleStream2BarDataset
    # load dataset as torch.utils.data.Dataset
    training_dataset = Groove2TripleStream2BarDataset.from_concatenated_datasets(
        config=yaml.load(open("helpers/configs/TripleStreams_beta_0.5_test.yaml", "r"), Loader=yaml.FullLoader),
        subset_tag="train",
        use_cached=True,
        downsampled_size=None,
        force_regenerate=False,
        move_all_to_cuda=True
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        training_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # <—
        drop_last=False
    )

    for batch in loader:
        for item in batch[:-2]:
            print(item.device)
        break

