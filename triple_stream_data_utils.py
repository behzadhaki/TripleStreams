import os
import pickle, bz2

import tqdm
from bokeh.io import output_notebook, show
import torch

try:
    import note_seq
    HAS_NOTE_SEQ = True
except:
    HAS_NOTE_SEQ = False


# loader
import bz2, pickle

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect legacy/private numpy path to the public one
        if module == "numpy._core":
            module = "numpy.core"
        return super().find_class(module, name)

def load_dataset(file_path):
    with bz2.BZ2File(file_path, 'rb') as f:
        return NumpyCompatUnpickler(f).load()


def print_dataset_structure(dataset_dict, name=None):
    first_level_keys = set()
    second_level_keys = set()
    third_level_keys = set()



    for key, value in dataset_dict.items():
        first_level_keys.add(key)
        if isinstance(value, dict):
            for sub_key in value.keys():
                second_level_keys.add(sub_key)
                if isinstance(value[sub_key], dict):
                    for sub_sub_key in value[sub_key].keys():
                        third_level_keys.add(sub_sub_key)

    if name is not None:
        print(f"Dataset Structure for Dataset: {name}")
    else:
        print("Dataset Structure:")


    print("----"*10)
    print("First Level Keys:")
    print(first_level_keys)
    print("----"*10)
    print("Second Level Keys:")
    print(list(second_level_keys)[:10])
    print("----"*10)
    print("Third Level Keys:")
    print(third_level_keys)
    print("----"*10)
    print("\n")

def print_all_datasets_structure(data_dir, dataset_pkls):
    for dataset_pkl in dataset_pkls:
        dataset_path = os.path.join(data_dir, dataset_pkl)
        dataset_dict = load_dataset(dataset_path)
        print_dataset_structure(dataset_dict, name=dataset_pkl)
        del dataset_dict

# get all split_n_bar_phrases from last key
def get_split_n_bar_phrases(dataset_path_list, data_dir):
    if not isinstance(dataset_path_list, list):
        dataset_path_list = [dataset_path_list]

    split_n_bar_phrases = list()

    for dataset_pkl_ in dataset_path_list:
        dataset_dict_ = load_dataset(os.path.join(data_dir, dataset_pkl_))
        for key, value in dataset_dict_.items():
            for sub_key, sub_value in value.items():
                hvo_splits = sub_value['split_n_bar_phrases']
                for hvo_split in hvo_splits:
                    hvo_split.metadata.update({'collection': dataset_pkl_.replace('.pkl.bz2', ''), 'sample_id': sub_key})

                split_n_bar_phrases.extend(sub_value['split_n_bar_phrases'])

    return split_n_bar_phrases

import random

def get_randome_phrases(split_n_bar_phrases, n=1):
    # returns hvos and their indexes in the split_n_bar_phrases list
    if not isinstance(split_n_bar_phrases, list):
        split_n_bar_phrases = [split_n_bar_phrases]
    sample_ix = random.sample(range(len(split_n_bar_phrases)), n)
    sample_hvo_phrases = [split_n_bar_phrases[ix] for ix in sample_ix]

    return sample_hvo_phrases, sample_ix


# version-safe imports
from bokeh.io import output_notebook, show
try:
    # Bokeh ≥ 3.0
    from bokeh.models import Tabs, TabPanel as Panel
    HAS_TABPANEL = True
except ImportError:
    # Bokeh 2.x (your case: 2.4.3)
    from bokeh.models.widgets import Tabs, Panel
    HAS_TABPANEL = False


import warnings
import logging
from bokeh.util.warnings import BokehUserWarning, BokehDeprecationWarning

def setup_quiet_bokeh():
    """Set up comprehensive bokeh warning suppression"""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=BokehUserWarning)
    warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*bokeh.*")

    # Suppress logging
    logging.getLogger('bokeh').setLevel(logging.ERROR)

    # Suppress other common warnings that might appear with bokeh
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

from  hvo_sequence.hvo_seq import HVO_Sequence

import itertools

def generate_stream_permutated_indices(start=0, end=3):
    """
    Generate all permutations of numbers from start to end inclusive.

    Parameters:
        start (int): Starting integer of the sequence
        end (int): Ending integer of the sequence (inclusive)

    Returns:
        list[list[int]]: List of permutations as lists
    """
    return [list(p) for p in itertools.permutations(range(start, end + 1))]

def list_permutations(lst):
    """
    Generate all permutations of the given list.

    Parameters:
        lst (list): The list to permute

    Returns:
        list[list]: All permutations as lists
    """
    return [list(p) for p in itertools.permutations(lst)]


def compile_into_list_of_hvo_seqs(input_hvos, output_hvos, metadatas, qpms=None):
    # input_hvos: list of arrays
    # output_hvos: list of arrays
    # metadata: list of dictionaries

    hvo_sequence_list = []

    drum_mapping = {
        "groove": [36]
    }

    for i in range(len(output_hvos)):
        input_hvo = input_hvos[i]
        output_hvo = output_hvos[i]
        metadata = metadatas[i]

        n_outputs_voices = output_hvo.shape[-1] // 3
        n_overall_voices = n_outputs_voices + 1

        for i in range(n_outputs_voices):
            drum_mapping[f"stream_{i + 1}"] = [37 + i]

        temp_hvo_seq = HVO_Sequence(drum_mapping=drum_mapping, beat_division_factors=[4])

        qpm = 120
        if qpms is not None:
            qpm = qpms[i]

        temp_hvo_seq.add_tempo(0, qpm=120)
        temp_hvo_seq.add_time_signature(0, 4, 4)

        temp_hvo_seq.adjust_length(input_hvo.shape[0])

        temp_hvo_seq.hvo[:, 0] = input_hvo[:, 0]  # groove hit
        temp_hvo_seq.hvo[:, n_overall_voices] = input_hvo[:, 1]  # groove vel
        temp_hvo_seq.hvo[:, n_overall_voices * 2] = input_hvo[:, 2]  # groove offset

        for i in range(n_outputs_voices):
            temp_hvo_seq.hvo[:, 1 + i] = output_hvo[:, i]  # stream hit
            temp_hvo_seq.hvo[:, 1 + n_overall_voices + i] = output_hvo[:, i + n_outputs_voices]  # stream vel
            temp_hvo_seq.hvo[:, 1 + n_overall_voices * 2 + i] = output_hvo[:, i + 2 * n_outputs_voices]  # stream offset

        temp_hvo_seq.metadata.update(metadata)
        hvo_sequence_list.append(temp_hvo_seq)

    return hvo_sequence_list


def create_multitab_from_HVO_Sequences(hvos,  tab_titles=None, show_tabs= True):
    """
    Create a multitab visualizer from already-generated bokeh figures.

    Parameters:
    -----------
    figures : List
        List of bokeh figure objects
    tab_titles : List[str]
        List of titles for each tab
    show_tabs : bool, default=True
        Whether to immediately show the tabs

    Returns:
    --------
    Tabs
        Bokeh Tabs object containing all the figures
    """
    setup_quiet_bokeh()
    output_notebook()

    figures = []

    generate_titles = False
    if tab_titles is None:
        tab_titles = []
        generate_titles = True

    if not isinstance(hvos, list):
        hvos = [hvos]
    for ix, hvo in enumerate(hvos):
        figure = hvo.to_html_plot(
            filename='',
            save_figure=False,
            show_figure=False)
        figures.append(figure)
        if generate_titles:
            tab_titles.append(f"Tab {ix+1}")

    if not isinstance(tab_titles, list):
        tab_titles = [tab_titles]
    if not tab_titles:
        tab_titles = [f"Tab {i+1}" for i in range(len(figures))]
    if len(figures) == 0:
        raise ValueError("No figures provided to create tabs")

    if len(figures) != len(tab_titles):
        raise ValueError("Number of figures must match number of tab titles")

    panels = []

    for figure, title in zip(figures, tab_titles):
        panel = TabPanel(child=figure, title=title) if HAS_TABPANEL else Panel(child=figure, title=title)
        panels.append(panel)

    tabs = Tabs(tabs=panels)

    if show_tabs:
        show(tabs)

    return tabs



import numpy as np

try:
    import seaborn as sns

    HAVE_SEABORN = True
except ModuleNotFoundError:
    HAVE_SEABORN = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOT = True
except:
    HAS_MATPLOT = False


def get_accent_hits_from_velocities(velocity_flat, accent_thresh=0.6, compare_consecutives=False):
    """
    Extract accent hits from the velocity flat representation of HVO.
    :param velocity_flat: a (B, T, 1) or (T, 1) numpy array where the last  column represents the velocity of hits.
    :param use_median: if True, use the median velocity to determine accent hits, otherwise use 0.5
    :return:
    """
    assert velocity_flat.ndim == 1 or velocity_flat.shape[
        -1] == 1, "Velocity flat must have 1 columns (hits, velocity, offset)"

    return np.where(velocity_flat > accent_thresh, 1, 0)


def get_split_to_streams(hvo_sample, groove_dim=0):
    """
    Split the HVO sample into input groove, streams, and flat output HVO.

    Parameters:
    -----------
    hvo_sample : HVOSequence
        The HVO sample to be split.
    groove_dim : int, optional
        The index of the groove dimension in the HVO sample. Default is 0.

    Returns:
    --------
    input_hvo : HVOSequence
        The input groove extracted from the HVO sample.
    streams : list of HVOSequence
        List of streams extracted from the HVO sample, excluding the groove dimension.
    flat_out_hvo : HVOSequence
        The flat output HVO sequence with the groove dimension set to zero.
    """
    # get input groove
    n_streams = hvo_sample.hits.shape[1]

    input_hvo = hvo_sample.hvo[:, (groove_dim, groove_dim + n_streams, groove_dim + 2 * n_streams)]

    # get flat of rest
    flat_hvo = hvo_sample.copy()
    flat_hvo.hvo[:, groove_dim] = 0  # remove groove hit
    flat_hvo.hvo[:, (groove_dim + n_streams)] = 0  # remove groove vel
    flat_hvo.hvo[:, (groove_dim + n_streams * 2)] = 0  # remove groove offset
    flat_out_hvo = flat_hvo.flatten_voices(reduce_dim=True)

    # streams
    streams = []
    for i in range(n_streams):
        if i != groove_dim:
            streams.append(hvo_sample.hvo[:, (i, i + n_streams, i + 2 * n_streams)])

    return input_hvo, streams, flat_out_hvo


def Jaccard_similarity(a, b):
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return (intersection / union)


def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Sequences must be of equal length")
    return sum(x != y for x, y in zip(a, b)) / len(a)


from hvo_sequence.utils import fuzzy_Hamming_distance


def extract_features_from_sets(hvo_sample, groove_dim):
    """
    Extract control features from a set of HVO samples.
    :param hvo_sample: HVOSequence
        The HVO sample to extract control features from.
    :param groove_dim: int
        The index of the groove dimension in the HVO sample.
    :return: dict
        A dictionary containing the control features.
    """
    input_hvo, out_streams, flat_out_hvo = get_split_to_streams(hvo_sample, groove_dim=groove_dim)

    input_hits = input_hvo[:, 0]
    flat_out_hits = flat_out_hvo[:, 0]
    analysis_dict = {
        "groove_to_output_hit_hamming_distance": float(hamming_distance(input_hits, flat_out_hits)),
    }

    input_velocities = input_hvo[:, 1]
    input_accent_hits = get_accent_hits_from_velocities(input_velocities)
    flat_out_velocities = flat_out_hvo[:, 1]
    flat_out_accent_hits = get_accent_hits_from_velocities(flat_out_velocities)
    analysis_dict.update({
        "groove_to_output_accent_hamming_distance": float(hamming_distance(input_accent_hits, flat_out_accent_hits)),
    })

    # calculate hit
    analysis_dict.update(
        {f"out_stream_{i}_jaccard_sim_with_flat_out": float(Jaccard_similarity(out_streams[i][:, 0], flat_out_hits)) for
         i in range(len(out_streams))})
    analysis_dict.update({f"out_stream_{i}_accent_hamming_distance_with_flat_out": float(
        hamming_distance(get_accent_hits_from_velocities(out_streams[i][:, 1]), flat_out_accent_hits)) for i in
                          range(len(out_streams))})

    return analysis_dict


# plot violin plots of the control features

def plot_control_features_violin(control_features_df):
    """
    Plot violin plots of the control features.
    :param control_features_df: pd.DataFrame
        DataFrame containing the control features.
    """

    plt.figure(figsize=(6, 8))
    sns.violinplot(data=control_features_df.drop(columns=["sample_id", "collection"]))
    # 90 deg rotate with wrapp
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Control Features Violin Plots")
    plt.tight_layout()

    plt.show()


# plot the  scatter of groove to output hit hamming distance and groove to output accent hamming distance
def plot_scatter_input_output_distribution(control_features_df, use_normalized_accents=False):
    if not HAS_MATPLOT: return
    plt.figure(figsize=(6, 4))
    x = control_features_df["groove_to_output_hit_hamming_distance"]
    y = control_features_df["groove_to_output_accent_hamming_distance"] if not use_normalized_accents else \
    control_features_df["accent_hamming_values_per_hit_hamming_normalized"]
    plt.scatter(x, y, alpha=0.005)
    plt.xlabel("Groove to Output Hit Hamming Distance")
    plt.ylabel("Groove to Output Accent Hamming Distance")
    if use_normalized_accents:
        plt.title("Scatter Plot of Groove to Output Hit and Binned Accent Hamming Distances")
    else:
        plt.title("Scatter Plot of Groove to Output Hit and Accent Hamming Distances")
    plt.tight_layout()
    plt.show()



import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def create_heatmap_histogram_from_lists(
    feat1, feat2,
    xlabel="X", ylabel="Y",
    title=None, show_zeros=False,
    figsize=(12, 6),
    clip_counts_at=None,
    saturate_colors_only=False,  # if True: colors capped, text shows true counts
):
    if len(feat1) != len(feat2):
        raise ValueError("feat1 and feat2 must have the same length.")

    # --- Vectorized unique + indexing ---
    x_arr = np.asarray(feat1, dtype=object)
    y_arr = np.asarray(feat2, dtype=object)

    # Sorted unique values + inverse indices (maps each element to its code)
    x_vals, x_inv = np.unique(x_arr, return_inverse=True)
    y_vals, y_inv = np.unique(y_arr, return_inverse=True)

    nx, ny = x_vals.size, y_vals.size

    # --- Vectorized counting (very fast) ---
    # Flatten the 2D (y,x) code pairs into 1D bins, then bincount
    flat_idx = np.ravel_multi_index((y_inv, x_inv), dims=(ny, nx))
    counts_true = np.bincount(flat_idx, minlength=ny * nx).reshape(ny, nx).astype(np.int32)

    # --- Color matrix (optionally clipped) ---
    counts_for_color = np.minimum(counts_true, clip_counts_at) if clip_counts_at is not None else counts_true

    # --- Text annotations (vectorized) ---
    if saturate_colors_only:
        text_base = counts_true
    else:
        text_base = counts_for_color

    if show_zeros:
        text_arr = text_base.astype(str)
    else:
        text_arr = np.where(text_base > 0, text_base.astype(str), "")

    x_ticks = x_vals.astype(str).tolist()
    y_ticks = y_vals.astype(str).tolist()

    if title is None:
        title = f"Heatmap: {xlabel} vs {ylabel} (unique values)"

    # inches -> px (100 dpi)
    plot_width = int(figsize[0] * 100)
    plot_height = int(figsize[1] * 100)

    heatmap_kwargs = dict(
        z=counts_for_color,
        x=x_ticks,
        y=y_ticks,
        text=text_arr,
        texttemplate="%{text}",
        textfont={"size": 6},
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title="Count"),
        hovertemplate=f'{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<br>Count: %{{z}}<extra></extra>'
    )

    # Force color scale to capped range when clipping
    if clip_counts_at is not None:
        heatmap_kwargs.update(dict(zmin=0, zmax=clip_counts_at))

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=plot_width,
        height=plot_height,
    )
    return fig


import matplotlib.pyplot as plt

# plot input features
def plot_scatter_distribution(feat1, feat2, xlabel, ylabel, title=None, alpha=0.005, figsize=(6, 6),
                              xlim=(-.05, 1.05), ylim=(-.05, 1.05)):
    if not HAS_MATPLOT: return
    plt.figure(figsize=figsize)
    x = feat1
    y = feat2
    plt.scatter(x, y, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()