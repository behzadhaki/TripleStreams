from triple_stream_data_utils import  get_split_to_streams, Jaccard_similarity, hamming_distance, print_all_datasets_structure, get_split_n_bar_phrases, get_accent_hits_from_velocities, list_permutations

# LOAD DATASETS
import os, pickle, bz2
import numpy as np
import tqdm

import numpy as np

import numpy as np

def compare_sequences_masked(a, b, positive_thresh=0.0):
    """
    Compare two sequences in [0,1] only at indices where either value > positive_thresh.

    Returns rounded similarities:
      - pearson:               nearest 0.1   in [-1, 1] (NaN if undefined)
      - cosine:                nearest 0.05  in [-1, 1] (NaN if undefined)
      - euclidean_similarity:  nearest 0.05  in [0, 1]
      - mae_similarity:        nearest 0.05  in [0, 1]
      - dtw_similarity:        nearest 0.05  in (0, 1]
    """
    def _round_step(x, step, lo=None, hi=None):
        if not np.isfinite(x):
            return x
        r = round(x / step) * step
        if lo is not None: r = max(lo, r)
        if hi is not None: r = min(hi, r)
        return float(0.0 if abs(r) < 1e-12 else r)

    def _dtw_similarity(x, y):
        m, n = len(x), len(y)
        D = np.full((m + 1, n + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, m + 1):
            xi = x[i - 1]
            for j in range(1, n + 1):
                cost = abs(xi - y[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        dtw_dist = D[m, n]
        return 1.0 / (1.0 + dtw_dist)

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    mask = (a > positive_thresh) | (b > positive_thresh)
    a_m, b_m = a[mask], b[mask]
    n = a_m.size
    if n == 0:
        return {k: np.nan for k in ["pearson", "cosine", "euclidean_similarity", "mae_similarity", "dtw_similarity"]}

    # Pearson
    if n >= 2 and np.std(a_m) > 0 and np.std(b_m) > 0:
        pearson = np.corrcoef(a_m, b_m)[0, 1]
    else:
        pearson = np.nan

    # Cosine
    na, nb = np.linalg.norm(a_m), np.linalg.norm(b_m)
    cosine = np.dot(a_m, b_m) / (na * nb) if na > 0 and nb > 0 else np.nan

    # Euclidean similarity
    dist = np.linalg.norm(a_m - b_m)
    euclidean_similarity = 1.0 - dist / np.sqrt(n)

    # MAE similarity
    mae_similarity = 1.0 - np.mean(np.abs(a_m - b_m))

    # DTW similarity
    dtw_similarity = _dtw_similarity(a_m, b_m)

    # Clip and round
    results = {
        "pearson":              _round_step(np.clip(pearson, -1, 1) if np.isfinite(pearson) else np.nan, 0.1,  -1.0, 1.0),
        "cosine":               _round_step(np.clip(cosine, -1, 1) if np.isfinite(cosine) else np.nan, 0.05, -1.0, 1.0),
        "euclidean_similarity": _round_step(np.clip(euclidean_similarity, 0, 1), 0.05, 0.0, 1.0),
        "mae_similarity":       _round_step(np.clip(mae_similarity, 0, 1), 0.05, 0.0, 1.0),
        "dtw_similarity":       _round_step(np.clip(dtw_similarity, 0, 1), 0.05, 0.0, 1.0),
    }
    return results




def compile_data_for_a_single_dataset_pkl(data_dir, name):
    assert name.endswith('.pkl.bz2'), "Dataset name mustend with .pkl.bz2"

    # print the structure of each dataset
    print_all_datasets_structure(data_dir, [name])

    # load all loops from all datasets
    split_n_bar_phrases_unpermutated_outputstreams = get_split_n_bar_phrases([name], data_dir)

    # print the number of phrases loaded
    print(f"Number of split_n_bar_phrases_unpermutated_outputstreams loaded: {len(split_n_bar_phrases_unpermutated_outputstreams)}")

    # INSPECT SOME HVO SAMPLES
    # create_multitab_from_HVO_Sequences(split_n_bar_phrases_unpermutated_outputstreams[2000:2010], show_tabs=True)



    accent_v_thresh = 0.6

    # Extract control features for all samples
    dataset = {
        "input_hvos": [],    #  added √
        "output_hvos": [],   #  added √
        "flat_out_hvos": [], # added √
        # "full_hvo_sequences": [],
        "sample_id": [],        # added √
        "collection": [],       # added √
        "all_metadata": [],     # added √
        "qpm": [],              # added √
        "OF_Input Hamming": [],     # output flattened to input hamming distance         # added √
        "OF_Input Hamming Accent": [],       # added √
        "OS1_OF Jaccard": [],  # Stream 1 of Output's jaccard distance to flattened input       # added √
        "OS2_OF Jaccard": [],  # Stream 2 of Output's jaccard distance to flattened input       # added √
        "OS3_OF Jaccard": [],  # Stream 3 of Output's jaccard distance to flattened input       # added √
        "pearson": [],         # added √
        "cosine": [],          # added √
        "euclidean_similarity": [],  # added √
        "mae_similarity": [],   # added √
        "dtw_similarity": [],  # added √
    }

    def mix_streams_into_hvo(streams_list):
        n_streams = len(streams_list)
        n_steps = streams_list[0].shape[0]
        temp_hvo = np.zeros((n_steps, n_streams * 3), dtype=np.float32)
        for i, stream in enumerate(streams_list):
            temp_hvo[:, i] = stream[:, 0]
            temp_hvo[:, i + n_streams] = stream[:, 1]
            temp_hvo[:, i + n_streams * 2] = stream[:, 2]
        return temp_hvo

    for hvo_sample in tqdm.tqdm(split_n_bar_phrases_unpermutated_outputstreams):
        for groove_dim in range(4):
            # get info
            sample_id = hvo_sample.metadata["sample_id"]
            collection = hvo_sample.metadata["collection"]
            tempo = hvo_sample.tempos[0].qpm
            all_metadata = hvo_sample.metadata

            # get num of time steps
            t_steps = hvo_sample.hits.shape[0]

            # get groove and outputs
            input_hvo, streams, flat_out_hvo = get_split_to_streams(hvo_sample, groove_dim=groove_dim)

            i_fo_hamming = np.round(hamming_distance(input_hvo[:, 0], flat_out_hvo[:, 0]), 6)
            i_fo_accent_hamming = np.round(hamming_distance(
                get_accent_hits_from_velocities(input_hvo[:, 1], accent_thresh=accent_v_thresh),
                get_accent_hits_from_velocities(flat_out_hvo[:, 1], accent_thresh=accent_v_thresh),
            ), 6)

            v1 = input_hvo[:, 1]
            v2 = flat_out_hvo[:, 1]
            vel_correlations_dict = compare_sequences_masked(v1, v2, positive_thresh=0.0)

            for streams_permuation in list_permutations(streams):
                dataset["sample_id"].append(hvo_sample.metadata['sample_id'])
                dataset["collection"].append(hvo_sample.metadata['collection'])
                dataset["all_metadata"].append(hvo_sample.metadata)
                dataset["qpm"].append(tempo)
                dataset["input_hvos"].append(input_hvo)
                dataset["flat_out_hvos"].append(flat_out_hvo)
                dataset["output_hvos"].append(mix_streams_into_hvo(streams_permuation))
                dataset["OF_Input Hamming"].append(i_fo_hamming)
                dataset["OF_Input Hamming Accent"].append(i_fo_accent_hamming)
                dataset["OS1_OF Jaccard"].append(np.round(Jaccard_similarity(flat_out_hvo[:, 0], streams_permuation[0][:, 0]), 6))
                dataset["OS2_OF Jaccard"].append(np.round(Jaccard_similarity(flat_out_hvo[:, 0], streams_permuation[1][:, 0]), 6))
                dataset["OS3_OF Jaccard"].append(np.round(Jaccard_similarity(flat_out_hvo[:, 0], streams_permuation[2][:, 0]), 6))
                dataset["pearson"].append(vel_correlations_dict["pearson"])
                dataset["cosine"].append(vel_correlations_dict["cosine"])
                dataset["euclidean_similarity"].append(vel_correlations_dict["euclidean_similarity"])
                dataset["mae_similarity"].append(vel_correlations_dict["mae_similarity"])
                dataset["dtw_similarity"].append(vel_correlations_dict["dtw_similarity"])

    return dataset

def compile_data_for_multiple_datasets_pkl(data_dir,  dataset_pkls):
    """
    Compile data for multiple datasets.
    """
    # create a directory to save the compiled data
    compiled_data_dir = os.path.join(data_dir, "compiled_data")
    os.makedirs(compiled_data_dir, exist_ok=True)
    dataset_dict = {}

    pkl_fnames = []
    for dataset_pkl_fname in dataset_pkls:
        if dataset_pkl_fname.endswith(".pkl.bz2") and dataset_pkl_fname.split(".")[0].split("_")[0] not in pkl_fnames:
            pkl_fnames.append(dataset_pkl_fname.split(".")[0].split("_")[0])
    final_dict_fname = "_".join(pkl_fnames) + ".pkl.bz2"
    print("Final dictionary filename will be:", final_dict_fname)
    for name in (dataset_pkls):
        current_dataset = compile_data_for_a_single_dataset_pkl(data_dir, name)

        # add the dataset to the dictionary
        for key, val in current_dataset.items():
            if key not in dataset_dict:
                dataset_dict[key] = []
            dataset_dict[key].extend(val)

    # save the dataset
    compiled_data_path = os.path.join(compiled_data_dir, final_dict_fname)
    with bz2.BZ2File(compiled_data_path, 'wb') as f:
        pickle.dump(dataset_dict, f)

    # delete the dataset_dict to free memory
    del dataset_dict

if __name__ == "__main__":

    # NON LMD DATASETS
    data_dir = "data/triple_streams/split_2bars/rest"
    dataset_pkls = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl.bz2')])

    for dataset_pkl_fname in dataset_pkls:
        if dataset_pkl_fname.endswith(".pkl.bz2"):
            print(f"\n\n\n\n\n --------> Compiling dataset: {dataset_pkl_fname}")
            print("-" * 80)
            compile_data_for_multiple_datasets_pkl(data_dir, [dataset_pkl_fname])
            print("\n\n\n √√√√√√√√√√ Finished compiling dataset:", dataset_pkl_fname)

    # LMD DATASETS
    print("\n\n\n\n\nCompiling dataset: LMD Top Four Dataset...")
    print("-"*80)
    data_dir = "data/triple_streams/split_2bars/lmd_top_four"  # "data/triple_streams/split_2bars/lmd or rest"
    dataset_pkls = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl.bz2')])
    compile_data_for_multiple_datasets_pkl(data_dir, dataset_pkls)
    print("\n\n\n √√√√√√√√√ Finished compiling dataset: LMD Top Four Dataset")