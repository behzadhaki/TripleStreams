# LOAD DATASETS
import tqdm
import os
import bz2
import pickle
import io
import numpy as np


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

def load_compiled_dataset_pkl_bz2(dataset_pkl_bz2, *, allow_pickle_arrays=False):
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


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Subset data from compiled dataset.")
    parser.add_argument("--accent_v_thresh", type=float, default=0.75, help="Accent velocity threshold for accent hits.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Proportion of data to use for testing.")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Proportion of data to use for validation.")
    parser.add_argument("--output_dir", type=str, default="data/triple_streams/cached", help="Directory to save the subsets.")

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Prepare paths
    # --------------------------------------------------------------------------

    root_path = f"data/triple_streams/cached/AccentAt{args.accent_v_thresh}/separate_sets"
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"The specified root path does not exist: {root_path} \n\n "
                                f"Please compile the dataset first using ---> "
                                f"python CompileStreamsAndFeatures.py --accent_v_thresh {args.accent_v_thresh} ")

    save_path = os.path.join(args.output_dir, f"AccentAt{args.accent_v_thresh}/training_splits")
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------------------------
    # Find Valid .pkl.bz2 files (each containing a specific collection of streams)
    # --------------------------------------------------------------------------

    pkl_bz2_collections = [
        f for f in os.listdir(root_path) if f.endswith('.pkl.bz2')
    ]
    if not pkl_bz2_collections:
        raise FileNotFoundError(f"No .pkl.bz2 files found in {root_path}")

    print("\n Found the following .pkl.bz2 files in the root path:")
    for pkl_bz2_file in pkl_bz2_collections:
        print(f"\t\t{pkl_bz2_file}")

    # --------------------------------------------------------------------------
    # Load each collection and subset the data
    # --------------------------------------------------------------------------

    print("\n Loading and Subsetting data...")

    # NOTE 1:
    #   EACH COLLECTION IS A DICTIONARY WITH KEYS:
        # dict_keys([
        #   'input_hvos',
        #   'output_hvos',
        #   'flat_out_hvos',
        #   'sample_id',
        #   'collection',
        #   'metadata',
        #   'qpm',
        #   'Flat Out Vs. Input | Hits | Hamming',
        #   'Flat Out Vs. Input | Accent | Hamming',
        #   'Stream 1 Vs. Flat Out | Hits | Hamming',
        #   'Stream 2 Vs. Flat Out | Hits | Hamming',
        #   'Stream 3 Vs. Flat Out | Hits | Hamming',

    # NOTE 2:
        # EACH KEY CORRESPONDS TO A LIST, WHERE EVERY 24 ITEMS IN THE LIST CORRESPOND TO PERMUTATIONS OF A SINGLE SAMPLE

    train_subset_dict = {}
    test_subset_dict = {}
    validation_subset_dict = {}

    # Note 3
        # In addition to the subsets, we will also store all of 'Flat Out Vs. Input | Accent | Hamming' and
        # also same for Stream 1, Stream 2, and Stream 3
        # 'Flat Out Vs. Input | Accent | Hamming Normalized' in separate files (without splitting)
        # this will be useful for final binning of these features

    # Special features keys
    special_features_keys = [
        'Flat Out Vs. Input | Accent | Hamming',
        'Stream 1 Vs. Flat Out | Hits | Hamming',
        'Stream 2 Vs. Flat Out | Hits | Hamming',
        'Stream 3 Vs. Flat Out | Hits | Hamming'
    ]
    special_features_dict = {key: [] for key in special_features_keys}

    # Iterate through each pkl.bz2 file and load the data
    for pkl_bz2_file in tqdm.tqdm(pkl_bz2_collections):
        collection_path = os.path.join(root_path, pkl_bz2_file)
        collection_data = load_compiled_dataset_pkl_bz2(collection_path, allow_pickle_arrays=True)

        # Make sure no np.nan values in any fields of the collection_data
        for key, value in collection_data.items():
            if isinstance(value, list):
                if not isinstance(value[0], str) and not isinstance(value[0], dict):
                    if np.isnan(np.array(value)).any():
                        raise ValueError(f"Found NaN values in the collection data for key: {key}. Please check your dataset.")

        # Calculate the number of samples
        n_overal_samples = len(collection_data['sample_id'])

        # Check if the number of samples is a multiple of 24
        n_overal_samples % 24 == 0, "the number of samples should be a multiple of 24, as each sample has 24 permutations. Something went wrong during the compilation of the dataset."

        # Calculate the number of unique samples
        n_unpermutated_unique_samples = n_overal_samples // 24

        # get unpermutated unique sample IDs for training, validation, and testing
        n_test_samples = int(n_unpermutated_unique_samples * args.test_split)
        n_validation_samples = int(n_unpermutated_unique_samples * args.validation_split)
        n_train_samples = n_unpermutated_unique_samples - n_test_samples - n_validation_samples
        assert n_train_samples >= 0, "Not enough samples for training. Please adjust the splits."

        # Extend the lists for the special features
        for key in special_features_keys:
            if key in collection_data:
                special_features_dict[key].extend(collection_data[key])
            else:
                raise KeyError(f"Key '{key}' not found in the collection data. Please check the dataset or remove from special features.")

        # the following two are necessary fields for evaluators in the training pipeline
        for ix in range(len(collection_data['collection'])):
            collection_data['metadata'][ix].update({'full_midi_filename': f"{ix}_dummy.mid"})
            collection_data['metadata'][ix].update({'master_id': f"ID {ix}"})
            collection_data['metadata'][ix].update({'style_primary': f"{collection_data['metadata'][ix]['collection']}"})


        # Get the indices for training, validation, and testing
        indices = np.arange(n_unpermutated_unique_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train_samples]
        validation_indices = indices[n_train_samples:n_train_samples + n_validation_samples]
        test_indices = indices[n_train_samples + n_validation_samples:]

        # Create the subsets
        train_subset = {key: [] for key in collection_data.keys()}
        test_subset = {key: [] for key in collection_data.keys()}
        validation_subset = {key: [] for key in collection_data.keys()}

        for i in train_indices:
            is_valid = True
            # check that there is no NAN in the special features
            for key in special_features_keys:
                if key in collection_data and np.isnan(collection_data[key][i * 24:(i + 1) * 24]).any():
                    is_valid = False
                    break
            if not is_valid:
                continue
            for key in collection_data.keys():
                if isinstance(collection_data[key], list):
                    # For each key, we need to take 24 items for each sample
                    train_subset[key].extend(collection_data[key][i * 24:(i + 1) * 24])
                else:
                    # For non-list keys, just take the value
                    train_subset[key].append(collection_data[key])

            # Extend the special features
            for key in special_features_keys:
                if key in collection_data:
                    special_features_dict[key].extend(collection_data[key][i * 24:(i + 1) * 24])

        for i in test_indices:
            is_valid = True
            # check that there is no NAN in the special features
            for key in special_features_keys:
                if key in collection_data and np.isnan(collection_data[key][i * 24:(i + 1) * 24]).any():
                    is_valid = False
                    break
            if not is_valid:
                continue
            for key in collection_data.keys():
                if isinstance(collection_data[key], list):
                    # For each key, we need to take 24 items for each sample
                    test_subset[key].extend(collection_data[key][i * 24:(i + 1) * 24])
                else:
                    # For non-list keys, just take the value
                    test_subset[key].append(collection_data[key])

            # Extend the special features
            for key in special_features_keys:
                if key in collection_data:
                    special_features_dict[key].extend(collection_data[key][i * 24:(i + 1) * 24])

        for i in validation_indices:
            is_valid = True
            # check that there is no NAN in the special features
            for key in special_features_keys:
                if key in collection_data and np.isnan(collection_data[key][i * 24:(i + 1) * 24]).any():
                    is_valid = False
                    break
            if not is_valid:
                continue
            for key in collection_data.keys():
                if isinstance(collection_data[key], list):
                    # For each key, we need to take 24 items for each sample
                    validation_subset[key].extend(collection_data[key][i * 24:(i + 1) * 24])
                else:
                    # For non-list keys, just take the value
                    validation_subset[key].append(collection_data[key])

            # Extend the special features
            for key in special_features_keys:
                if key in collection_data:
                    special_features_dict[key].extend(collection_data[key][i * 24:(i + 1) * 24])

        # Add the subsets to the main dictionaries
        for key in collection_data.keys():
            if key not in train_subset_dict:
                train_subset_dict[key] = []
            if key not in test_subset_dict:
                test_subset_dict[key] = []
            if key not in validation_subset_dict:
                validation_subset_dict[key] = []
            train_subset_dict[key].extend(train_subset[key])
            test_subset_dict[key].extend(test_subset[key])
            validation_subset_dict[key].extend(validation_subset[key])

    # Store the subsets in the dictionaries as train.pkl.bz2, test.pkl.bz2, validation.pkl.bz2
    train_subset_path = os.path.join(save_path, f"train.pkl.bz2")
    test_subset_path = os.path.join(save_path, f"test.pkl.bz2")
    validation_subset_path = os.path.join(save_path, f"validation.pkl.bz2")

    # Store Subsets of the special features
    with bz2.BZ2File(train_subset_path, 'wb') as f:
        pickle.dump(_encode_for_pickle(train_subset_dict, allow_pickle_arrays=True), f)
        print(f"\n\nTrain subset saved to: {train_subset_path}")
    with bz2.BZ2File(test_subset_path, 'wb') as f:
        pickle.dump(_encode_for_pickle(test_subset_dict, allow_pickle_arrays=True), f)
        print(f"\n\nTest subset saved to: {test_subset_path}")
    with bz2.BZ2File(validation_subset_path, 'wb') as f:
        pickle.dump(_encode_for_pickle(validation_subset_dict, allow_pickle_arrays=True), f)
        print(f"\n\nValidation subset saved to: {validation_subset_path}")

    print("FINAL SET OF AVAILABLE KEYS IN THE SUBSETS:")
    for key in train_subset_dict.keys():
        print(f"\t\t\t\t|------> {key}")

    # Store the special features in separate files
    special_features_path = os.path.join(save_path, "special_features")
    os.makedirs(special_features_path, exist_ok=True)
    for key, values in special_features_dict.items():
        special_feature_path = os.path.join(special_features_path, f"{key}.pkl.bz2")
        with bz2.BZ2File(special_feature_path, 'wb') as f:
            pickle.dump(_encode_for_pickle(values, allow_pickle_arrays=True), f)
            print(f"\n\nSpecial feature '{key}' saved to: {special_feature_path}")


    print(f"\n --------------------------------------------------------"
          f"\n Subsetting completed with {len(train_subset_dict['sample_id'])} training samples, "
          f"\n {len(test_subset_dict['sample_id'])} testing samples, and {len(validation_subset_dict['sample_id'])} validation samples. \n"
          f" ----------------------------------------------------------\n")