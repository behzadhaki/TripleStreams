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
from scipy.stats import wasserstein_distance

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

# --------------- helpers: Center of Mass Extractiors ---------------

def hvo_to_polar_center_of_mass(hvo, use_velocity=False, use_microtiming=False):
    """
    Convert HVO sequence to polar coordinates and find center of mass (vectorized).

    Args:
        hvo: np.array of shape [batch_size, 32, 3] where:
             - [:, :, 0] = onsets (binary)
             - [:, :, 1] = velocities (0-1)
             - [:, :, 2] = microtiming (-0.5 to 0.5)
        use_velocity: bool, if True use velocity as radius (normalized to max), else use 1
        use_microtiming: bool, if True adjust timing with microtiming offset

    Returns:
        magnitude: np.array of shape [batch_size] - magnitude of center of mass (0-1)
        angle: np.array of shape [batch_size] - angle of center of mass (0-1, representing 0 to 2π)
    """
    # Create timestep array for all batches: shape [batch_size, 32]
    timesteps = np.arange(32)[np.newaxis, :].repeat(hvo.shape[0], axis=0)

    # Calculate actual timing positions
    actual_timing = timesteps.astype(float)
    if use_microtiming:
        actual_timing = actual_timing + hvo[:, :, 2]
        # Handle wraparound for step 0 with negative microtiming
        step_0_mask = (timesteps == 0) & (hvo[:, :, 2] < 0)
        actual_timing[step_0_mask] = 32 + hvo[:, :, 2][step_0_mask]
        # Ensure timing stays within valid range
        actual_timing = actual_timing % 32

    # Convert timing to angles (0 to 2π): shape [batch_size, 32]
    angles = (actual_timing / 32.0) * 2 * np.pi

    # Determine radius for each point: shape [batch_size, 32]
    if use_velocity:
        # Normalize velocity to max value per batch
        max_velocity = np.max(hvo[:, :, 1], axis=1, keepdims=True)  # shape [batch_size, 1]
        # Handle case where max velocity is zero
        max_velocity_safe = np.where(max_velocity > 0, max_velocity, 1.0)
        radius = hvo[:, :, 1] / max_velocity_safe  # normalized velocity as radius
    else:
        radius = np.ones_like(hvo[:, :, 1])  # unit radius

    # Convert to Cartesian coordinates: shape [batch_size, 32]
    x_all = radius * np.cos(angles)
    y_all = radius * np.sin(angles)

    # Weight by onset strength: shape [batch_size, 32]
    weights = hvo[:, :, 0]

    # Apply weights to coordinates
    x_weighted = x_all * weights
    y_weighted = y_all * weights

    # Calculate center of mass for each batch
    total_weights = np.sum(weights, axis=1)  # shape [batch_size]

    # Handle case where no onsets exist (avoid division by zero)
    total_weights_safe = np.where(total_weights > 0, total_weights, 1.0)

    x_center = np.sum(x_weighted, axis=1) / total_weights_safe  # shape [batch_size]
    y_center = np.sum(y_weighted, axis=1) / total_weights_safe  # shape [batch_size]

    # Set center to (0,0) for batches with no onsets
    no_onsets_mask = total_weights == 0
    x_center[no_onsets_mask] = 0.0
    y_center[no_onsets_mask] = 0.0

    # Convert back to polar coordinates
    magnitude = np.sqrt(x_center ** 2 + y_center ** 2)
    angle_radians = np.arctan2(y_center, x_center)

    # Handle floating-point precision errors: if magnitude is essentially zero, set to exactly zero
    tolerance = 1e-10
    near_zero_mask = magnitude < tolerance
    magnitude[near_zero_mask] = 0.0

    # For near-zero magnitudes, set angle to 0 (arbitrary but consistent)
    angle_radians[near_zero_mask] = 0.0

    # Normalize angle to 0-1 range (0 to 2π)
    angle_normalized = (angle_radians + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

    return magnitude, angle_normalized


def hvo_polar_comparison(hvo1, hvo2):
    """
    Compare two HVO sequences using polar coordinate differences.

    Args:
        hvo1: np.array of shape [batch_size, 32, 3] - first HVO sequence
        hvo2: np.array of shape [batch_size, 32, 3] - second HVO sequence

    Returns:
        timing_mag_diff: magnitude difference for timing (hvo1 - hvo2, -1 to +1)
        timing_ang_diff: angle difference for timing (hvo1 - hvo2, -0.5 to +0.5)
        velocity_mag_diff: magnitude difference for velocity (hvo1 - hvo2, -1 to +1)
        velocity_ang_diff: angle difference for velocity (hvo1 - hvo2, -0.5 to +0.5)

    Expected Behavior Examples:

    TIMING DIFFERENCES (onset + microtiming, unit radius):
    - Single onset vs scattered onsets:
      timing_mag_diff ≈ +0.8 (hvo1 more concentrated)
    - Scattered onsets vs single onset:
      timing_mag_diff ≈ -0.8 (hvo1 less concentrated)
    - Onset at step 0 vs onset at step 8:
      timing_ang_diff = -0.25 (hvo1 is 90° behind hvo2)
    - Onset at step 8 vs onset at step 0:
      timing_ang_diff = +0.25 (hvo1 is 90° ahead of hvo2)
    - Onsets at opposite positions (step 0 vs step 16):
      timing_ang_diff = ±0.5 (180° apart, sign depends on shortest path)
    - Same timing pattern: timing_mag_diff ≈ 0, timing_ang_diff ≈ 0

    VELOCITY DIFFERENCES (onset + velocity weighting, no microtiming):
    - High velocity (0.8) vs low velocity (0.3) at same position:
      velocity_mag_diff ≈ +0.5 (hvo1 stronger velocity profile)
    - Single strong onset vs multiple weak onsets:
      velocity_mag_diff > 0 (hvo1 more concentrated velocity)
    - Velocity pattern at step 0 vs step 12:
      velocity_ang_diff ≈ -0.375 (hvo1 is 135° behind hvo2)
    - Same velocity pattern: velocity_mag_diff ≈ 0, velocity_ang_diff ≈ 0

    EDGE CASES:
    - No onsets vs single onset: mag_diff = -1.0
    - Single onset vs no onsets: mag_diff = +1.0
    - Floating-point precision near zero: handled with tolerance (returns 0.0)

    INTERPRETATION:
    - Positive mag_diff: hvo1 has more concentrated/focused pattern
    - Negative mag_diff: hvo1 has more spread out/diffuse pattern
    - Positive ang_diff: hvo1's center is counterclockwise from hvo2
    - Negative ang_diff: hvo1's center is clockwise from hvo2
    """
    # Get timing polar coordinates (onset + microtiming, no velocity weighting)
    mag1_timing, ang1_timing = hvo_to_polar_center_of_mass(hvo1, use_velocity=False, use_microtiming=True)
    mag2_timing, ang2_timing = hvo_to_polar_center_of_mass(hvo2, use_velocity=False, use_microtiming=True)

    # Get velocity polar coordinates (onset + velocity, no microtiming)
    mag1_velocity, ang1_velocity = hvo_to_polar_center_of_mass(hvo1, use_velocity=True, use_microtiming=False)
    mag2_velocity, ang2_velocity = hvo_to_polar_center_of_mass(hvo2, use_velocity=True, use_microtiming=False)

    # Calculate timing differences
    timing_mag_diff = mag1_timing - mag2_timing
    timing_ang_diff = ang1_timing - ang2_timing
    # Handle circular angle difference (shortest path)
    timing_ang_diff = np.where(timing_ang_diff > 0.5, timing_ang_diff - 1.0, timing_ang_diff)
    timing_ang_diff = np.where(timing_ang_diff < -0.5, timing_ang_diff + 1.0, timing_ang_diff)

    # Calculate velocity differences
    velocity_mag_diff = mag1_velocity - mag2_velocity
    velocity_ang_diff = ang1_velocity - ang2_velocity
    # Handle circular angle difference (shortest path)
    velocity_ang_diff = np.where(velocity_ang_diff > 0.5, velocity_ang_diff - 1.0, velocity_ang_diff)
    velocity_ang_diff = np.where(velocity_ang_diff < -0.5, velocity_ang_diff + 1.0, velocity_ang_diff)

    return timing_mag_diff, timing_ang_diff, velocity_mag_diff, velocity_ang_diff


def hvo_combined_polar_center_of_mass(hvo1, hvo2, use_velocity=True, use_microtiming=True):
    """
    Combine two HVO sequences on a single z-plane and find their combined center of mass.

    Args:
        hvo1: np.array of shape [batch_size, 32, 3] - first HVO sequence
        hvo2: np.array of shape [batch_size, 32, 3] - second HVO sequence
        use_velocity: bool, if True use velocity as radius (normalized to max), else use 1
        use_microtiming: bool, if True adjust timing with microtiming offset

    Returns:
        magnitude: np.array [batch_size] - magnitude of combined center of mass (0-1)
        angle: np.array [batch_size] - angle of combined center of mass (0-1, representing 0 to 2π)
    """
    import numpy as np

    # Get individual polar coordinates for both HVOs
    mag1, ang1 = hvo_to_polar_center_of_mass(hvo1, use_velocity=use_velocity, use_microtiming=use_microtiming)
    mag2, ang2 = hvo_to_polar_center_of_mass(hvo2, use_velocity=use_velocity, use_microtiming=use_microtiming)

    # Convert to Cartesian coordinates
    ang1_rad = ang1 * 2 * np.pi
    ang2_rad = ang2 * 2 * np.pi

    x1 = mag1 * np.cos(ang1_rad)
    y1 = mag1 * np.sin(ang1_rad)
    x2 = mag2 * np.cos(ang2_rad)
    y2 = mag2 * np.sin(ang2_rad)

    # Calculate combined center of mass (simple average of the two centers)
    x_combined = (x1 + x2) / 2.0
    y_combined = (y1 + y2) / 2.0

    # Convert back to polar coordinates
    magnitude = np.sqrt(x_combined ** 2 + y_combined ** 2)
    angle_radians = np.arctan2(y_combined, x_combined)

    # Handle floating-point precision errors
    tolerance = 1e-10
    near_zero_mask = magnitude < tolerance
    magnitude[near_zero_mask] = 0.0
    angle_radians[near_zero_mask] = 0.0

    # Normalize angle to 0-1 range (0 to 2π)
    angle_normalized = (angle_radians + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

    return magnitude, angle_normalized

# ---------- Tokenization of control values into bins ----------
def tokenize_control_feature_array(
        control_array: np.array,
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

# ----------- Legacy Dataset Class for Groove2TripleStream2BarDataset -----------

class Groove2TripleStream2BarDataset(Dataset):

    def __init__(self,
                 config,
                 subset_tag,            # pass "train" or "validation" or "test"
                 use_cached=True,
                 downsampled_size=None,
                 force_regenerate=False,
                 move_all_to_cuda=False,
                 print_logs=False):

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
                self.collection = [self.dataset_files[0] for _ in range(len(self.metadata))]

                process_data = False

        if process_data:
            # ------------------------------------------------------------------------------------------
            # load pre-stored hvo_sequences or
            #   a portion of them uniformly sampled if down_sampled_ratio is provided
            # ------------------------------------------------------------------------------------------
            n_samples = 0
            loaded_data_dictionary = {}

            if print_logs:
                pbar = tqdm.tqdm(self.dataset_files, desc="Loading data files") if len(self.dataset_files) > 1 else self.dataset_files
            else:
                pbar = self.dataset_files

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
            self.collection = [self.dataset_files[0] for _ in range(len(self.metadata))]

            # Populate control tokens
            # ------------------------------------------------------------------------------------------

            self.encoding_control1_tokens = tokenize_control_feature_array(
                control_array=np.round(loaded_data_dictionary[self.encoding_control1_key], 5),
                n_bins=self.n_encoding_control1_tokens, low=0, high=1)
            self.encoding_control2_tokens = tokenize_control_feature_array(
                control_array=np.round(loaded_data_dictionary[self.encoding_control2_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control1_tokens = tokenize_control_feature_array(
                control_array=np.round(loaded_data_dictionary[self.decoding_control1_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control2_tokens = tokenize_control_feature_array(
                control_array=np.round(loaded_data_dictionary[self.decoding_control2_key], 5),
                n_bins=self.n_encoding_control2_tokens, low=0, high=0.85)
            self.decoding_control3_tokens = tokenize_control_feature_array(
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
            if print_logs:
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
            if print_logs:
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
            self.collection = [self.dataset_files[0] for _ in range(len(self.metadata))]
            
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
                                   move_all_to_cuda=False,
                                   print_logs=False):
        """
        Alternative constructor that loads multiple dataset files and concatenates them
        into a single dataset instance, avoiding ConcatDataset issues.

        This preserves device location (GPU/CPU) of tensors after concatenation.
        Individual datasets are cached, but the concatenated result is not cached.
        """
        dataLoaderLogger.info(
            f"Creating concatenated dataset from {len(config['dataset_files'])} files") if print_logs else None

        individual_downsampled_size = int(
            downsampled_size / len(config["dataset_files"])) if downsampled_size is not None else None

        # Load all individual datasets (these can be cached)
        datasets = []
        pbar = tqdm.tqdm(config["dataset_files"], desc="Loading dataset files for concatenation") if print_logs else config["dataset_files"]
        for dataset_file in pbar:
            if print_logs:
                # Update description with current filename
                pbar.set_description(f"Loading: {dataset_file}")
            new_config = config.copy()
            new_config["dataset_files"] = [dataset_file]

            dataset = cls(
                config=new_config,
                subset_tag=subset_tag,
                use_cached=use_cached,
                downsampled_size=individual_downsampled_size,
                force_regenerate=force_regenerate,
                move_all_to_cuda=False,  # Don't move individual datasets to CUDA yet
                print_logs=print_logs
            )
            datasets.append(dataset)


        common_keys = set.intersection(*(set(ds.metadata[0].keys()) for ds in datasets))
        dataLoaderLogger.info(f"Loaded {len(datasets)} datasets:") if print_logs else None
        for ix, ds in enumerate(datasets):
            ds.metadata = [{k: v for k, v in sample.items() if k in common_keys} for sample in ds.metadata]
            dataLoaderLogger.info(f"\t {ix} : {config['dataset_files'][ix]} --> {len(ds)} samples") if print_logs else None

        # Concatenate all data
        dataLoaderLogger.info("Concatenating datasets...") if print_logs else None

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
                dataLoaderLogger.info("Moving concatenated dataset to CUDA") if print_logs else None
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                dataLoaderLogger.info("Moving concatenated dataset to MPS") if print_logs else None
            else:
                device = torch.device("cpu")
                dataLoaderLogger.warning("CUDA and MPS not available, keeping dataset on CPU") if print_logs else None

            instance.input_grooves = instance.input_grooves.to(device)
            instance.output_streams = instance.output_streams.to(device)
            instance.flat_output_streams = instance.flat_output_streams.to(device)
            instance.encoding_control1_tokens = instance.encoding_control1_tokens.to(device)
            instance.encoding_control2_tokens = instance.encoding_control2_tokens.to(device)
            instance.decoding_control1_tokens = instance.decoding_control1_tokens.to(device)
            instance.decoding_control2_tokens = instance.decoding_control2_tokens.to(device)
            instance.decoding_control3_tokens = instance.decoding_control3_tokens.to(device)

        dataLoaderLogger.info(f"Concatenated dataset created with {len(instance)} samples") if print_logs else None
        return instance

def get_triplestream_dataset(
        config,
        subset_tag,  # pass "train" or "validation" or "test"
        use_cached=True,
        downsampled_size=None,
        force_regenerate=False,
        move_all_to_cuda=False,
        print_logs=False):
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
        move_all_to_cuda=move_all_to_cuda,
        print_logs=print_logs
    )


# ---------- Latest Dataset Class for flexible management of control values/tokens ----------
class FlexControlGroove2TripleStream2BarDataset(Dataset):
    """
    Dataset class for FlexControlTripleStreamsVAE that supports flexible control token configurations.
    Features are cached separately from control tokenization for better flexibility.
    """

    def __init__(self,
                 config,
                 subset_tag,  # pass "train" or "validation" or "test"
                 use_cached=True,
                 downsampled_size=None,
                 force_regenerate=False,
                 move_all_to_cuda=False,
                 print_logs=False):

        self.dataset_root_path = config["dataset_root_path"]
        self.dataset_files = config["dataset_files"]
        self.subset_tag = subset_tag
        self.max_len = config["max_len"]

        # Flexible control configuration - handle empty lists
        self.n_encoding_control_tokens = config.get("n_encoding_control_tokens", [])
        self.encoding_control_keys = config.get("encoding_control_keys", [])
        self.n_decoding_control_tokens = config.get("n_decoding_control_tokens", [])
        self.decoding_control_keys = config.get("decoding_control_keys", [])

        # Validate control configuration consistency
        if len(self.n_encoding_control_tokens) != len(self.encoding_control_keys):
            raise ValueError(
                f"Mismatch: {len(self.n_encoding_control_tokens)} encoding control tokens vs {len(self.encoding_control_keys)} keys")

        if len(self.n_decoding_control_tokens) != len(self.decoding_control_keys):
            raise ValueError(
                f"Mismatch: {len(self.n_decoding_control_tokens)} decoding control tokens vs {len(self.decoding_control_keys)} keys")

        def get_source_compiled_data_dictionary_path():
            return os.path.join(self.dataset_root_path, self.subset_tag)

        def get_cached_features_filepath():
            """Cache path for features only (not control tokens)"""
            dir_ = os.path.join("cached/TorchDatasets/", self.dataset_root_path.replace("/", "-"), self.subset_tag)
            os.makedirs(dir_, exist_ok=True)
            filename = "".join([df.split("_")[0] for df in self.dataset_files])
            filename += f"_features_{self.max_len}_{downsampled_size}"
            filename = filename.replace(" ", "_").replace("|", "_").replace("/", "_").replace("\\", "_").replace("__",
                                                                                                                 "_").replace(
                "__", "_")
            filename = filename + "_features.bz2pickle"
            return os.path.join(dir_, filename)

        # Load or generate base data and features
        # ------------------------------------------------------------------------------------------
        process_data = True
        cache_needs_update = False

        if use_cached and not force_regenerate:
            cached_features_path = get_cached_features_filepath()
            if os.path.exists(cached_features_path):
                if print_logs:
                    print(f"Loading cached features from: {cached_features_path}")

                ifile = bz2.BZ2File(cached_features_path, 'rb')
                data = pickle.load(ifile)
                ifile.close()

                self.input_grooves = data["input_grooves"]
                self.output_streams = data["output_streams"]
                self.flat_output_streams = data["flat_output_streams"]
                self.metadata = data["metadata"]
                self.tempos = data["tempos"]
                self.collection = [self.dataset_files[0] for _ in range(len(self.metadata))]
                features = data["features"]

                process_data = False

                # Check if we need to extract any new features
                all_control_keys = set(self.encoding_control_keys + self.decoding_control_keys)
                missing_features = all_control_keys - set(features.keys())

                if missing_features:
                    if print_logs:
                        print(f"Missing features detected: {missing_features}")
                        print("Extracting missing features...")

                    # Create temporary data dict for feature extraction
                    temp_data_dict = {
                        "input_hvos": self.input_grooves,
                        "output_hvos": self.output_streams,
                        "flat_out_hvos": self.flat_output_streams
                    }

                    # Extract missing features
                    new_features = self.extract_features_dict(temp_data_dict)

                    # Add only the missing features
                    for key in missing_features:
                        if key in new_features:
                            features[key] = new_features[key]
                            if print_logs:
                                print(f"  Added feature: {key}")

                    cache_needs_update = True

        if process_data:
            if print_logs:
                print("Processing data from source files...")

            # Load data from source files
            n_samples = 0
            loaded_data_dictionary = {}

            pbar = tqdm.tqdm(self.dataset_files, desc="Loading data files") if print_logs and len(
                self.dataset_files) > 1 else self.dataset_files

            for dataset_file in pbar:
                if not isinstance(pbar, list):
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

            # Downsample if needed
            if downsampled_size is not None:
                if downsampled_size >= n_samples:
                    downsampled_size = None
                else:
                    sampled_indices = np.random.choice(n_samples, downsampled_size, replace=False)
                    if print_logs:
                        print(f"Downsizing by selecting {downsampled_size} from {n_samples} samples")
                    for k, v in loaded_data_dictionary.items():
                        loaded_data_dictionary[k] = [v[ix] for ix in sampled_indices]

            # Populate basic fields
            self.input_grooves = np.array(loaded_data_dictionary["input_hvos"])
            self.output_streams = np.array(loaded_data_dictionary["output_hvos"])
            self.flat_output_streams = np.array(loaded_data_dictionary["flat_out_hvos"])
            self.metadata = loaded_data_dictionary["metadata"]
            self.tempos = loaded_data_dictionary["qpm"]
            self.collection = [self.dataset_files[0] for _ in range(len(self.metadata))]

            # Extract all features
            if print_logs:
                print("Extracting features...")
            features = self.extract_features_dict(loaded_data_dictionary)

            cache_needs_update = True

        # Update cache if needed
        # ------------------------------------------------------------------------------------------
        if use_cached and cache_needs_update:
            cached_features_path = get_cached_features_filepath()
            if print_logs:
                print(f"Updating features cache at: {cached_features_path}")

            data_to_cache = {
                "input_grooves": self.input_grooves,
                "output_streams": self.output_streams,
                "flat_output_streams": self.flat_output_streams,
                "metadata": self.metadata,
                "tempos": self.tempos,
                "features": features
            }

            ofile = bz2.BZ2File(cached_features_path, 'wb')
            pickle.dump(data_to_cache, ofile)
            ofile.close()

        # Now tokenize controls based on current configuration
        # ------------------------------------------------------------------------------------------
        if print_logs:
            print("Tokenizing controls for current configuration...")

        n_samples = len(self.input_grooves)

        # Create encoding control tokens tensor - handle empty case
        if len(self.encoding_control_keys) == 0:
            self.encoding_control_values = np.empty((n_samples, 0), dtype=np.float32)
            self.encoding_controls = np.empty((n_samples, 0), dtype=np.float32)
            if print_logs:
                print("No encoding controls specified - using empty tensors")
        else:
            encoding_control_values_list = []
            encoding_tokens_list = []
            for i, (key, n_tokens) in enumerate(zip(self.encoding_control_keys, self.n_encoding_control_tokens)):
                if key not in features:
                    raise KeyError(
                        f"Encoding control key '{key}' not found in features. Available: {list(features.keys())}")
                tokens_or_controls, control_array = self.tokenize(features, key, n_tokens)
                encoding_tokens_list.append(tokens_or_controls)
                encoding_control_values_list.append(control_array)

            self.encoding_control_values = np.stack(encoding_control_values_list, axis=1)
            self.encoding_controls = np.stack(encoding_tokens_list, axis=1)

        # Create decoding control tokens tensor - handle empty case
        if len(self.decoding_control_keys) == 0:
            self.decoding_control_values = np.empty((n_samples, 0), dtype=np.float32)
            self.decoding_controls = np.empty((n_samples, 0), dtype=np.float32)
            if print_logs:
                print("No decoding controls specified - using empty tensors")
        else:
            decoding_control_values_list = []
            decoding_tokens_list = []
            for i, (key, n_tokens) in enumerate(zip(self.decoding_control_keys, self.n_decoding_control_tokens)):
                if key not in features:
                    raise KeyError(
                        f"Decoding control key '{key}' not found in features. Available: {list(features.keys())}")
                tokens_or_controls, control_array = self.tokenize(features, key, n_tokens)
                decoding_tokens_list.append(tokens_or_controls)
                decoding_control_values_list.append(control_array)

            self.decoding_control_values = np.stack(decoding_control_values_list, axis=1)
            self.decoding_controls = np.stack(decoding_tokens_list, axis=1)

        # Safety checks and cleanup
        # ------------------------------------------------------------------------------------------
        def get_invalid_indices(hvo):
            n_voices = hvo.shape[-1] // 3
            hits = hvo[:, :, :n_voices]
            velocities = hvo[:, :, n_voices:2 * n_voices]
            offsets = hvo[:, :, 2 * n_voices:3 * n_voices]
            h_invalid_sample_ix, _, _ = np.where((hits > 1) | (hits < 0.0))
            v_invalid_sample_ix, _, _ = np.where((velocities > 1) | (velocities < 0.0))
            o_invalid_sample_ix, _, _ = np.where((offsets > 0.5) | (offsets < -0.5))
            invalid_sample_ix = set(h_invalid_sample_ix).union(set(v_invalid_sample_ix)).union(set(o_invalid_sample_ix))
            return invalid_sample_ix

        def get_empty_indices(hvo):
            n_voices = hvo.shape[-1] // 3
            empty_indices = np.where(np.all(hvo[:, :, :n_voices] == 0, axis=(1, 2)))[0]
            return empty_indices

        invalid_indices_input = get_invalid_indices(self.input_grooves)
        invalid_indices_output = get_invalid_indices(self.output_streams)
        all_invalid_indices = set(invalid_indices_input).union(set(invalid_indices_output))
        empty_output_indices = set(get_empty_indices(self.output_streams))

        # Keep only 99% of empty output indices
        if len(empty_output_indices) > 0:
            empty_output_indices = set(
                np.random.choice(list(empty_output_indices), int(len(empty_output_indices) * 0.99),
                                 replace=False).tolist())
        all_invalid_indices = all_invalid_indices.union(empty_output_indices)

        # Remove invalid samples
        if len(all_invalid_indices) > 0:
            if print_logs:
                print(f"Found {len(all_invalid_indices)} invalid samples. Removing them.")
                print("Size before removing invalid samples: ", self.input_grooves.shape[0])

            self.input_grooves = np.delete(self.input_grooves, list(all_invalid_indices), axis=0)
            self.output_streams = np.delete(self.output_streams, list(all_invalid_indices), axis=0)
            self.flat_output_streams = np.delete(self.flat_output_streams, list(all_invalid_indices), axis=0)

            # Handle control arrays
            if self.encoding_controls.shape[1] > 0:
                self.encoding_controls = np.delete(self.encoding_controls, list(all_invalid_indices), axis=0)
                self.encoding_control_values = np.delete(self.encoding_control_values, list(all_invalid_indices),
                                                         axis=0)
            else:
                n_valid_samples = self.input_grooves.shape[0]
                self.encoding_controls = np.empty((n_valid_samples, 0), dtype=np.float32)
                self.encoding_control_values = np.empty((n_valid_samples, 0), dtype=np.float32)

            if self.decoding_controls.shape[1] > 0:
                self.decoding_controls = np.delete(self.decoding_controls, list(all_invalid_indices), axis=0)
                self.decoding_control_values = np.delete(self.decoding_control_values, list(all_invalid_indices),
                                                         axis=0)
            else:
                n_valid_samples = self.input_grooves.shape[0]
                self.decoding_controls = np.empty((n_valid_samples, 0), dtype=np.float32)
                self.decoding_control_values = np.empty((n_valid_samples, 0), dtype=np.float32)

            self.metadata = [self.metadata[ix] for ix in range(len(self.metadata)) if ix not in all_invalid_indices]
            self.tempos = [self.tempos[ix] for ix in range(len(self.tempos)) if ix not in all_invalid_indices]
            self.collection = [self.collection[ix] for ix in range(len(self.collection)) if
                               ix not in all_invalid_indices]

            if print_logs:
                print("Size after removing invalid samples: ", self.input_grooves.shape[0])

        # Convert to tensors
        # ------------------------------------------------------------------------------------------
        self.indices = list(range(len(self.input_grooves)))
        self.input_grooves = torch.tensor(self.input_grooves, dtype=torch.float32)
        self.output_streams = torch.tensor(self.output_streams, dtype=torch.float32)
        self.flat_output_streams = torch.tensor(self.flat_output_streams, dtype=torch.float32)
        self.encoding_controls = torch.tensor(self.encoding_controls, dtype=torch.float32)
        self.decoding_controls = torch.tensor(self.decoding_controls, dtype=torch.float32)
        self.decoding_control_values = torch.tensor(self.decoding_control_values, dtype=torch.float32)
        self.encoding_control_values = torch.tensor(self.encoding_control_values, dtype=torch.float32)

        # Validate and log tensor shapes
        if print_logs:
            print(f"Final tensor shapes:")
            print(f"  input_grooves: {self.input_grooves.shape}")
            print(f"  output_streams: {self.output_streams.shape}")
            print(f"  encoding_controls: {self.encoding_controls.shape}")
            print(f"  decoding_controls: {self.decoding_controls.shape}")

        # Move to CUDA if requested
        # ------------------------------------------------------------------------------------------
        if move_all_to_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            self.input_grooves = self.input_grooves.to(device)
            self.output_streams = self.output_streams.to(device)
            self.flat_output_streams = self.flat_output_streams.to(device)
            self.encoding_controls = self.encoding_controls.to(device)
            self.decoding_controls = self.decoding_controls.to(device)

        self.indices = list(range(len(self.metadata)))

    def tokenize(self, features, key, n_tokens):
        """Helper method for tokenizing control features"""
        if isinstance(n_tokens, str):
            if n_tokens.lower() == "none":
                n_tokens = None
            else:
                assert isinstance(n_tokens, int), f"n_tokens should be an int or 'None', got {n_tokens}"

        if key == "Flat Out Vs. Input | Hits | Hamming":
            low = 0.0
            high = 32.0
            control_array_ = np.round(features[key], 5)
        elif (key == "Flat Out Vs. Input | Accent | Hamming" or
              key == "Stream 1 Vs. Flat Out | Hits | Hamming" or
              key == "Stream 2 Vs. Flat Out | Hits | Hamming" or
              key == "Stream 3 Vs. Flat Out | Hits | Hamming"):
            low = 0.0
            high = 0.85
            control_array_ = np.round(features[key], 5)
        elif key == "Relative Density":
            low = 0.0
            high = 1.0
            control_array_ = np.round(features[key], 5)
        elif key == "Structural Similarity Distance":
            low = 0.0
            high = 1.0
            control_array_ = (np.round(features[key], 5) / 5.6568)
        elif key == "Total Out Hits":
            low = 0.0
            high = 96.0
            control_array_ = np.round(features[key], 5)
        elif key == "Output Step Density":
            low = 0.0
            high = 1.0
            control_array_ = np.clip((np.round(features[key], 5) - 1), 0, 3) / (3.0 - 1.0)
        elif (key == "Stream 1 Relative Density" or
              key == "Stream 2 Relative Density" or
              key == "Stream 3 Relative Density"):
            low = 0.0
            high = 1.0
            control_array_ = np.round(features[key], 5)
        elif "Center of Mass" in key:
            low = 0.0
            high = 1.0
            control_array_ = np.round(features[key], 5)
        else:
            available_keys = '\n'.join(features.keys())
            raise KeyError(f"Control key '{key}' not recognized - available keys: {available_keys}")

        if n_tokens is None:
            return control_array_, control_array_
        else:
            tokens = tokenize_control_feature_array(
                control_array=control_array_,
                n_bins=n_tokens,
                low=low,
                high=high
            )
            return tokens, control_array_

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return (self.input_grooves[idx],
                self.output_streams[idx],
                self.encoding_controls[idx],
                self.decoding_controls[idx],
                self.metadata[idx],
                self.indices[idx])

    @classmethod
    def from_concatenated_datasets(cls,
                                   config,
                                   subset_tag,
                                   use_cached=True,
                                   downsampled_size=None,
                                   force_regenerate=False,
                                   move_all_to_cuda=False,
                                   print_logs=False):
        """
        Alternative constructor that loads multiple dataset files and concatenates them
        into a single dataset instance for FlexControl configuration.
        """
        if print_logs:
            dataLoaderLogger.info(
                f"Creating concatenated FlexControl dataset from {len(config['dataset_files'])} files")

        individual_downsampled_size = int(
            downsampled_size / len(config["dataset_files"])) if downsampled_size is not None else None

        # Load all individual datasets
        datasets = []
        pbar = tqdm.tqdm(config["dataset_files"], desc="Loading FlexControl dataset files") if print_logs else config[
            "dataset_files"]
        for dataset_file in pbar:
            if print_logs:
                pbar.set_description(f"Loading: {dataset_file}")
            new_config = config.copy()
            new_config["dataset_files"] = [dataset_file]

            dataset = cls(
                config=new_config,
                subset_tag=subset_tag,
                use_cached=use_cached,
                downsampled_size=individual_downsampled_size,
                force_regenerate=force_regenerate,
                move_all_to_cuda=False,
                print_logs=print_logs
            )
            datasets.append(dataset)

        # Get common metadata keys
        common_keys = set.intersection(*(set(ds.metadata[0].keys()) for ds in datasets))
        if print_logs:
            dataLoaderLogger.info(f"Loaded {len(datasets)} FlexControl datasets:")
        for ix, ds in enumerate(datasets):
            ds.metadata = [{k: v for k, v in sample.items() if k in common_keys} for sample in ds.metadata]
            if print_logs:
                dataLoaderLogger.info(f"\t {ix} : {config['dataset_files'][ix]} --> {len(ds)} samples")

        # Concatenate all data
        if print_logs:
            dataLoaderLogger.info("Concatenating FlexControl datasets...")

        # Concatenate tensors
        input_grooves = torch.cat([ds.input_grooves for ds in datasets], dim=0)
        output_streams = torch.cat([ds.output_streams for ds in datasets], dim=0)
        flat_output_streams = torch.cat([ds.flat_output_streams for ds in datasets], dim=0)
        encoding_controls = torch.cat([ds.encoding_controls for ds in datasets], dim=0)
        decoding_controls = torch.cat([ds.decoding_controls for ds in datasets], dim=0)
        encoding_control_values = torch.cat([ds.encoding_control_values for ds in datasets], dim=0)
        decoding_control_values = torch.cat([ds.decoding_control_values for ds in datasets], dim=0)

        # Concatenate lists
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
        instance.n_encoding_control_tokens = config["n_encoding_control_tokens"]
        instance.encoding_control_keys = config["encoding_control_keys"]
        instance.n_decoding_control_tokens = config["n_decoding_control_tokens"]
        instance.decoding_control_keys = config["decoding_control_keys"]

        # Assign concatenated data
        instance.input_grooves = input_grooves
        instance.output_streams = output_streams
        instance.flat_output_streams = flat_output_streams
        instance.encoding_controls = encoding_controls
        instance.decoding_controls = decoding_controls
        instance.encoding_control_values = encoding_control_values
        instance.decoding_control_values = decoding_control_values

        instance.metadata = metadata
        instance.tempos = tempos
        instance.collection = collection
        instance.indices = list(range(len(metadata)))

        # Move to GPU if requested
        if move_all_to_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if print_logs:
                    dataLoaderLogger.info("Moving concatenated FlexControl dataset to CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                if print_logs:
                    dataLoaderLogger.info("Moving concatenated FlexControl dataset to MPS")
            else:
                device = torch.device("cpu")
                if print_logs:
                    dataLoaderLogger.warning("CUDA and MPS not available, keeping FlexControl dataset on CPU")

            instance.input_grooves = instance.input_grooves.to(device)
            instance.output_streams = instance.output_streams.to(device)
            instance.flat_output_streams = instance.flat_output_streams.to(device)
            instance.encoding_controls = instance.encoding_controls.to(device)
            instance.decoding_controls = instance.decoding_controls.to(device)

        if print_logs:
            dataLoaderLogger.info(f"Concatenated FlexControl dataset created with {len(instance)} samples")
        return instance

    @classmethod
    def flatten_triple_streams(cls, hvos):
        assert len(hvos.shape) == 3, "Expected shape of hvos is (n_samples, n_steps, n_voices*3)"
        flat_out_hvos = np.zeros((hvos.shape[0], hvos.shape[1], 3))

        # get hits
        flat_out_hvos[:, :, 0] = np.clip(np.sum(hvos[:, :, :3], axis=-1), 0, 1)
        # get indices of max velocities
        max_vel_indices = np.argmax(hvos[:, :, 3:6], axis=-1)
        # set flat out velocities based on max velocity indices
        flat_out_hvos[:, :, 1] = np.take_along_axis(hvos[:, :, 3:6], max_vel_indices[..., None], axis=-1).squeeze(-1)
        # use same for offsets
        flat_out_hvos[:, :, 2] = np.take_along_axis(hvos[:, :, 6:9], max_vel_indices[..., None], axis=-1).squeeze(-1)
        return flat_out_hvos

    @classmethod
    def extract_features_dict(cls, data_dict, normalize=False):

        assert ("input_hvos" in data_dict and
                "output_hvos" in data_dict), \
            "input_hvos, output_hvos must be present in the loaded data dictionary"

        if "flat_out_hvos" not in data_dict:
            # flatten the output_hvos to get flat_out_hvos
            data_dict["flat_out_hvos"] = cls.flatten_triple_streams(data_dict["output_hvos"])

        features_extracted = {}

        # Basic features that are always extracted
        basic_features = {
            "Flat Out Vs. Input | Hits | Hamming",
            "Flat Out Vs. Input | Accent | Hamming",
            "Stream 1 Vs. Flat Out | Hits | Hamming",
            "Stream 2 Vs. Flat Out | Hits | Hamming",
            "Stream 3 Vs. Flat Out | Hits | Hamming"
        }

        # Add basic features if they exist in the data dict
        for feature in basic_features:
            if feature in data_dict:
                features_extracted[feature] = data_dict[feature]

        # Check if relative hit density is present
        if "Relative Density" not in data_dict or "Total Out Hits" not in data_dict:
            stream_1_hits = np.sum(np.array(data_dict["output_hvos"])[:, :, 0], axis=-1)
            stream_2_hits = np.sum(np.array(data_dict["output_hvos"])[:, :, 1], axis=-1)
            stream_3_hits = np.sum(np.array(data_dict["output_hvos"])[:, :, 2], axis=-1)

            total_hits = (stream_1_hits + stream_2_hits + stream_3_hits)

            if normalize:
                features_extracted.update({"Total Out Hits": (total_hits / 96.0).tolist()})
            else:
                features_extracted.update({"Total Out Hits": (total_hits).tolist()})

            # replace zeros with 1
            total_hits[total_hits == 0] = 1

            features_extracted.update({"Stream 1 Relative Density": (stream_1_hits / total_hits).tolist()})
            features_extracted.update({"Stream 2 Relative Density": (stream_2_hits / total_hits).tolist()})
            features_extracted.update({"Stream 3 Relative Density": (stream_3_hits / total_hits).tolist()})

        if "Output Step Density" not in data_dict:
            # total onsets divided by the number of steps with at least one onset
            flat_out_hits = np.clip(np.array(data_dict["flat_out_hvos"])[:, :, 0], 0, 1)
            steps_with_at_least_one_onset = flat_out_hits.sum(axis=-1)
            steps_with_at_least_one_onset[steps_with_at_least_one_onset == 0] = 1  # avoid division by zero

            total_out_hits = np.sum(np.array(data_dict["output_hvos"])[:, :, :3], axis=-1).sum(axis=-1)
            features_extracted.update(
                {"Output Step Density": (total_out_hits / steps_with_at_least_one_onset).tolist()})

            if normalize:
                features_extracted["Output Step Density"] = np.clip(
                    (np.round(features_extracted["Output Step Density"], 5) - 1), 0, 3) / (3.0 - 1.0)

        if "Structural Similarity Distance" not in data_dict:
            # reference: https://github.com/fredbru/GrooveToolbox/blob/c73ecfc7bcc7f7bdb69372ea3532fa613c38665a/SimilarityMetrics.py#L108-L119
            flat_in_vels = np.clip(np.array(data_dict["input_hvos"])[:, :, 1], 0, 1)
            flat_out_vels = np.clip(np.array(data_dict["flat_out_hvos"])[:, :, 1], 0, 1)
            struct_sim = np.clip(np.sqrt(np.sum((flat_in_vels - flat_out_vels) ** 2, axis=-1)), 0, 10)
            if normalize:
                features_extracted["Structural Similarity Distance"] = (struct_sim / 5.6568).tolist()
            else:
                features_extracted["Structural Similarity Distance"] = struct_sim.tolist()

        if "Center of Mass" not in data_dict:
            # input CoM
            input_com_mag, input_com_angle = hvo_to_polar_center_of_mass(
                hvo=np.array(data_dict["input_hvos"]),
                use_velocity=True,
                use_microtiming=True
            )

            output_com_mag, output_com_angle = hvo_to_polar_center_of_mass(
                hvo=np.array(data_dict["flat_out_hvos"]),
                use_velocity=True,
                use_microtiming=True
            )

            input_with_output_com_mag, input_with_output_com_angle = hvo_combined_polar_center_of_mass(
                hvo1=np.array(data_dict["input_hvos"]),
                hvo2=np.array(data_dict["flat_out_hvos"]),
                use_velocity=True,
                use_microtiming=True
            )

            features_extracted.update({
                "Center of Mass | Input | Magnitude": input_com_mag,
                "Center of Mass | Input | Angle": input_com_angle,
                "Center of Mass | Output | Magnitude": output_com_mag,
                "Center of Mass | Output | Angle": output_com_angle,
                "Center of Mass | Input + Output | Magnitude": input_with_output_com_mag,
                "Center of Mass | Input + Output | Angle": input_with_output_com_angle
            })

        return features_extracted


def get_flexcontrol_triplestream_dataset(
        config,
        subset_tag,  # pass "train" or "validation" or "test"
        use_cached=True,
        downsampled_size=None,
        force_regenerate=False,
        move_all_to_cuda=False,
        print_logs=False, ):
    """
    Get FlexControl dataset that returns control tokens as tensors instead of individual tokens.
    Now supports empty control configurations for no-control experiments.
    Features are cached separately from control tokenization for better flexibility.
    """

    try:
        cfg_dict = config.as_dict()
    except:
        cfg_dict = config

    # Ensure control configuration keys exist with defaults
    cfg_dict.setdefault("n_encoding_control_tokens", [])
    cfg_dict.setdefault("encoding_control_keys", [])
    cfg_dict.setdefault("n_decoding_control_tokens", [])
    cfg_dict.setdefault("decoding_control_keys", [])

    # Validate that empty lists are consistent
    if len(cfg_dict["n_encoding_control_tokens"]) != len(cfg_dict["encoding_control_keys"]):
        if len(cfg_dict["n_encoding_control_tokens"]) == 0 and len(cfg_dict["encoding_control_keys"]) == 0:
            pass  # Both empty, OK
        else:
            raise ValueError(
                f"Encoding control mismatch: {len(cfg_dict['n_encoding_control_tokens'])} tokens vs {len(cfg_dict['encoding_control_keys'])} keys")

    if len(cfg_dict["n_decoding_control_tokens"]) != len(cfg_dict["decoding_control_keys"]):
        if len(cfg_dict["n_decoding_control_tokens"]) == 0 and len(cfg_dict["decoding_control_keys"]) == 0:
            pass  # Both empty, OK
        else:
            raise ValueError(
                f"Decoding control mismatch: {len(cfg_dict['n_decoding_control_tokens'])} tokens vs {len(cfg_dict['decoding_control_keys'])} keys")

    if print_logs:
        n_enc = len(cfg_dict["n_encoding_control_tokens"])
        n_dec = len(cfg_dict["n_decoding_control_tokens"])
        print(f"Loading dataset with {n_enc} encoding controls and {n_dec} decoding controls")
        if n_enc == 0 and n_dec == 0:
            print("  → NO CONTROLS MODE: Traditional VAE behavior")
        elif n_enc == 0:
            print("  → NO ENCODING CONTROLS: Only decoding controls active")
        elif n_dec == 0:
            print("  → NO DECODING CONTROLS: Only encoding controls active")

    return FlexControlGroove2TripleStream2BarDataset.from_concatenated_datasets(
        config=cfg_dict,
        subset_tag=subset_tag,
        use_cached=use_cached,
        downsampled_size=downsampled_size,
        force_regenerate=force_regenerate,
        move_all_to_cuda=move_all_to_cuda,
        print_logs=print_logs
    )

if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run demos/data/demo.py to test")


    #
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset

    import yaml
    from torch.utils.data import DataLoader


    # # load dataset as torch.utils.data.Dataset
    # from data import Groove2TripleStream2BarDataset
    #
    # training_dataset = Groove2TripleStream2BarDataset.from_concatenated_datasets(
    #     config=yaml.load(open("helpers/configs/Testing.yaml", "r"), Loader=yaml.FullLoader),
    #     subset_tag="train",
    #     use_cached=False,
    #     downsampled_size=None,
    #     force_regenerate=False,
    #     move_all_to_cuda=False
    # )
    # max(training_dataset.encoding_control1_tokens)
    # max(training_dataset.encoding_control2_tokens)
    # max(training_dataset.decoding_control1_tokens)
    #
    # loader = DataLoader(
    #     training_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=0,  # <—
    #     drop_last=False
    # )
    #
    # for batch in loader:
    #     for item in batch[:-2]:
    #         print(item.device)
    #     break



    # FLEX Control Dataset
    from data import FlexControlGroove2TripleStream2BarDataset

    config = {
        'dataset_root_path': "data/triple_streams/model_ready/AccentAt0.75/",
        'max_len': 32,
        'dataset_files':
        [
            "01_candombe_four_voices.pkl.bz2",
            "02_elbg_both_flattened_left_right.pkl.bz2",
            # "03_groove_midi_crash_hhclosed_hhopen_ride.pkl.bz2",
            # "04_groove_midi_hh_kick_snare_toms.pkl.bz2",
            # "05_groove_midi_hi_lo_mid_ride.pkl.bz2",
            # "06_lmd_bass_brass_drum_percussion.pkl.bz2",
            # "07_lmd_bass_brass_drum_percussive.pkl.bz2",
            # "08_lmd_bass_brass_guitar_percussion.pkl.bz2",
            # "09_lmd_bass_brass_guitar_percussive.pkl.bz2",
            # "10_lmd_bass_brass_guitar_piano.pkl.bz2",
            # "11_lmd_bass_brass_percussion_percussive.pkl.bz2",
            # "12_lmd_bass_brass_percussion_piano.pkl.bz2",
            # "13_lmd_bass_brass_percussive_piano.pkl.bz2",
            # "14_lmd_bass_drum_guitar_percussion.pkl.bz2",
            # "15_lmd_bass_drum_guitar_percussive.pkl.bz2",
            # "16_lmd_bass_drum_percussion_percussive.pkl.bz2",
            # "17_lmd_bass_drum_percussion_piano.pkl.bz2",
            # "18_lmd_bass_drum_percussive_piano.pkl.bz2",
            # "19_lmd_bass_guitar_percussion_percussive.pkl.bz2",
            # "20_lmd_bass_guitar_percussion_piano.pkl.bz2",
            # "21_lmd_bass_guitar_percussive_piano.pkl.bz2",
            # "22_lmd_bass_percussion_percussive_piano.pkl.bz2",
            # "23_lmd_brass_drum_guitar_percussion.pkl.bz2",
            # "24_lmd_brass_drum_guitar_percussive.pkl.bz2",
            # "25_lmd_brass_drum_guitar_piano.pkl.bz2",
            # "26_lmd_brass_drum_percussion_percussive.pkl.bz2",
            # "27_lmd_brass_drum_percussion_piano.pkl.bz2",
            # "28_lmd_brass_drum_percussive_piano.pkl.bz2",
            # "29_lmd_brass_guitar_percussion_percussive.pkl.bz2",
            # "30_lmd_brass_guitar_percussion_piano.pkl.bz2",
            # "31_lmd_brass_guitar_percussive_piano.pkl.bz2",
            # "32_lmd_brass_percussion_percussive_piano.pkl.bz2",
            # "33_lmd_drum_guitar_percussion_percussive.pkl.bz2",
            # "34_lmd_drum_guitar_percussion_piano.pkl.bz2",
            # "35_lmd_drum_guitar_percussive_piano.pkl.bz2",
            # "36_lmd_drum_percussion_percussive_piano.pkl.bz2",
            # "37_lmd_guitar_percussion_percussive_piano.pkl.bz2",
            # "38_ttd_both-is-and_both_flattened_left_right.pkl.bz2",
            # "39_ttd_both-is-or_both_flattened_left_right.pkl.bz2",
        ],

        # Encoding Controls (converted from legacy encoding_control1/2)
        'n_encoding_control_tokens': [33, 5],  # Was: n_encoding_control1_tokens: 33, n_encoding_control2_tokens: 10
        'encoding_control_modes': ['prepend', 'prepend'],  # Strategic: first prepended, second added
        'encoding_control_keys':
            ["Structural Similarity Distance",  # Was: encoding_control1_key
            "Flat Out Vs. Input | Accent | Hamming"],  # Was: encoding_control2_key

        # Decoding Controls (converted from legacy decoding_control1/2/3)
        'n_decoding_control_tokens': [97, 10, 10, 10],  # Was: n_decoding_control1/2/3_tokens: 10
        'decoding_control_modes': ['prepend', 'prepend', 'prepend', 'prepend'],  # All prepended (legacy behavior)
        'decoding_control_keys':
            ["Total Out Hits",
            "Output Step Density",
             "Stream 2 Relative Density",
             "Stream 3 Relative Density"]
    }
    flex_dataset = FlexControlGroove2TripleStream2BarDataset.from_concatenated_datasets(
        config=config,
        subset_tag="train",
        use_cached=True,
        downsampled_size=None,
        force_regenerate=False,
        move_all_to_cuda=False,
        print_logs=True
    )

    # Stuctural Similarity Values
    structural_similarity_values = flex_dataset.encoding_control_values[:, 0]
    # get index of inf values

    inf_indices = np.where(np.isinf(structural_similarity_values))[0]
    structural_similarity_values.min(), structural_similarity_values.max(), len(inf_indices)

    # Onset Coincidence Rate
    output_step_density = flex_dataset.decoding_control_values[:, 1]
    inf_indices = np.where(np.isinf(output_step_density))[0]
    output_step_density.min(), output_step_density.max(), len(inf_indices)