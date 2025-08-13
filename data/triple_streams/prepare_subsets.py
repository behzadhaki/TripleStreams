# LOAD DATASETS
import tqdm
import os
import bz2
import pickle
import io
import numpy as np
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hvo_sequence.hvo_seq import HVO_Sequence

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


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


def save_dataset_as_hdf5(data_dict, filename):
    """
    Save dataset as HDF5 file - much more efficient than JSON+NPY for many arrays.
    HDF5 is cross-platform, language-agnostic, and handles mixed data types well.
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py not available. Install with: pip install h5py")

    with h5py.File(filename, 'w') as f:
        def save_recursive(group, data, path=""):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    # Save numpy arrays directly
                    group.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, list):
                    if value and isinstance(value[0], (int, float, np.number)):
                        # Save numeric lists as arrays
                        group.create_dataset(key, data=np.array(value), compression='gzip')
                    elif value and isinstance(value[0], dict):
                        # Handle list of dicts (like metadata)
                        subgroup = group.create_group(key)
                        for i, item in enumerate(value):
                            item_group = subgroup.create_group(str(i))
                            save_recursive(item_group, item, f"{path}/{key}/{i}")
                    else:
                        # Save as string array for other lists
                        str_array = np.array([str(v) for v in value], dtype=h5py.string_dtype())
                        group.create_dataset(key, data=str_array)
                elif isinstance(value, dict):
                    # Create subgroup for dictionaries
                    subgroup = group.create_group(key)
                    save_recursive(subgroup, value, f"{path}/{key}")
                else:
                    # Save scalars and strings
                    if isinstance(value, str):
                        group.attrs[key] = value
                    else:
                        group.create_dataset(key, data=value)

        save_recursive(f, data_dict)


def load_dataset_from_hdf5(filename):
    """
    Load dataset from HDF5 file.
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py not available. Install with: pip install h5py")

    def load_recursive(group):
        result = {}

        # Load attributes (scalars/strings)
        for key, value in group.attrs.items():
            result[key] = value

        # Load datasets and subgroups
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data = item[()]
                if item.dtype.kind == 'S':  # String array
                    result[key] = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in data]
                else:
                    result[key] = data
            elif isinstance(item, h5py.Group):
                # Check if this is a list of dicts (numeric keys)
                if all(k.isdigit() for k in item.keys()):
                    # Reconstruct list of dicts
                    result[key] = []
                    for i in sorted(item.keys(), key=int):
                        result[key].append(load_recursive(item[i]))
                else:
                    # Regular dictionary
                    result[key] = load_recursive(item)

        return result

    with h5py.File(filename, 'r') as f:
        return load_recursive(f)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Subset data from compiled dataset files separately.")
    parser.add_argument("--accent_v_thresh", type=float, default=0.75,
                        help="Accent velocity threshold for accent hits.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Proportion of data to use for testing.")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Proportion of data to use for validation.")
    parser.add_argument("--input_dir", type=str, default="with_features",
                        help="Base directory containing the separate sets.")
    parser.add_argument("--output_dir", type=str, default="model_ready",
                        help="Directory to save the subsets.")
    parser.add_argument("--use_hdf5", action="store_true",
                        help="Use HDF5 format instead of pickle (recommended for portability)")
    parser.add_argument("--use_json_format", action="store_true",
                        help="Use JSON+NPY format instead of pickle (creates many files, not recommended)")

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Prepare paths
    # --------------------------------------------------------------------------

    root_path = args.input_dir
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"The specified root path does not exist: {root_path} \n\n "
                                f"Please compile the dataset first using ---> "
                                f"python CompileStreamsAndFeatures.py --accent_v_thresh {args.accent_v_thresh} --input_dir {args.input_dir}")

    save_path = os.path.join(args.output_dir, f"AccentAt{args.accent_v_thresh}")
    os.makedirs(save_path, exist_ok=True)

    # Create subdirectories for each subset type
    train_dir = os.path.join(save_path, "train")
    test_dir = os.path.join(save_path, "test")
    validation_dir = os.path.join(save_path, "validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Find Valid .pkl.bz2 files and sort them alphabetically
    # --------------------------------------------------------------------------

    pkl_bz2_collections = [
        f for f in os.listdir(root_path) if f.endswith('.pkl.bz2')
    ]
    if not pkl_bz2_collections:
        raise FileNotFoundError(f"No .pkl.bz2 files found in {root_path}")

    # Sort files alphabetically
    pkl_bz2_collections.sort()

    print("\n Found the following .pkl.bz2 files in the root path (sorted alphabetically):")
    for i, pkl_bz2_file in enumerate(pkl_bz2_collections):
        print(f"\t\t{i + 1:02d}. {pkl_bz2_file}")

    # --------------------------------------------------------------------------
    # Load each collection and subset the data separately
    # --------------------------------------------------------------------------

    print("\n Loading and Subsetting data for each file separately...")

    # Iterate through each pkl.bz2 file and load the data
    for file_index, pkl_bz2_file in enumerate(tqdm.tqdm(pkl_bz2_collections)):
        collection_path = os.path.join(root_path, pkl_bz2_file)
        collection_data = load_compiled_dataset_pkl_bz2(collection_path, allow_pickle_arrays=True)

        # Get base filename without extension
        base_filename = pkl_bz2_file.replace('.pkl.bz2', '')

        # Create numbered prefix (1-indexed)
        file_prefix = f"{file_index + 1:02d}"

        # Make sure no np.nan values in any fields of the collection_data
        for key, value in collection_data.items():
            if isinstance(value, list):
                if value and not isinstance(value[0], str) and not isinstance(value[0], dict):
                    if np.isnan(np.array(value)).any():
                        raise ValueError(
                            f"Found NaN values in the collection data for key: {key}. Please check your dataset.")

        # Calculate the number of samples
        n_overal_samples = len(collection_data['sample_id'])

        # Check if the number of samples is a multiple of 24
        assert n_overal_samples % 24 == 0, "the number of samples should be a multiple of 24, as each sample has 24 permutations. Something went wrong during the compilation of the dataset."

        # Calculate the number of unique samples
        n_unpermutated_unique_samples = n_overal_samples // 24

        # get unpermutated unique sample IDs for training, validation, and testing
        n_test_samples = int(n_unpermutated_unique_samples * args.test_split)
        n_validation_samples = int(n_unpermutated_unique_samples * args.validation_split)
        n_train_samples = n_unpermutated_unique_samples - n_test_samples - n_validation_samples
        assert n_train_samples >= 0, "Not enough samples for training. Please adjust the splits."

        # Update metadata with required fields
        for ix in range(len(collection_data['collection'])):
            if 'full_midi_filename' not in collection_data['metadata'][ix]:
                collection_data['metadata'][ix]['full_midi_filename'] = f"{base_filename}_{ix}.mid"
            if 'master_id' not in collection_data['metadata'][ix]:
                collection_data['metadata'][ix]['master_id'] = f"{base_filename}_{ix}"
            # Keep style_primary as is (don't overwrite if it exists)
            if 'style_primary' not in collection_data['metadata'][ix]:
                collection_data['metadata'][ix]['style_primary'] = f"{collection_data['metadata'][ix]['collection']}"

        # Get the indices for training, validation, and testing
        indices = np.arange(n_unpermutated_unique_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train_samples]
        validation_indices = indices[n_train_samples:n_train_samples + n_validation_samples]
        test_indices = indices[n_train_samples + n_validation_samples:]

        # Create the subsets for this file
        train_subset = {key: [] for key in collection_data.keys()}
        test_subset = {key: [] for key in collection_data.keys()}
        validation_subset = {key: [] for key in collection_data.keys()}


        # Helper function to add samples to subset
        def add_samples_to_subset(subset_dict, sample_indices):
            for i in sample_indices:
                for key in collection_data.keys():
                    if isinstance(collection_data[key], list):
                        # For each key, we need to take 24 items for each sample
                        subset_dict[key].extend(collection_data[key][i * 24:(i + 1) * 24])


        add_samples_to_subset(train_subset, train_indices)
        add_samples_to_subset(test_subset, test_indices)
        add_samples_to_subset(validation_subset, validation_indices)

        # Save the subsets for this file with numbered prefix and in separate directories
        subsets = [
            (train_subset, 'train', train_dir),
            (test_subset, 'test', test_dir),
            (validation_subset, 'validation', validation_dir)
        ]

        for subset_data, subset_name, subset_dir in subsets:
            if args.use_hdf5:
                # Use HDF5 format (recommended for portability)
                hdf5_filename = os.path.join(subset_dir, f"{file_prefix}_{base_filename}.h5")
                save_dataset_as_hdf5(subset_data, hdf5_filename)
                print(f"{subset_name.capitalize()} subset for {base_filename} saved as HDF5: {hdf5_filename}")
            elif args.use_json_format:
                # Use JSON + NPY format (creates many files, not recommended)
                json_filename = os.path.join(subset_dir, f"{file_prefix}_{base_filename}.json")
                json_file, npy_files = save_dataset_as_json_npy(subset_data,
                                                                os.path.join(subset_dir,
                                                                             f"{file_prefix}_{base_filename}"))
                print(f"{subset_name.capitalize()} subset for {base_filename} saved as JSON: {json_filename}")
                print(f"  Associated numpy files: {len(npy_files)} files")
            else:
                # Use original pickle format
                subset_path = os.path.join(subset_dir, f"{file_prefix}_{base_filename}.pkl.bz2")
                with bz2.BZ2File(subset_path, 'wb') as f:
                    pickle.dump(_encode_for_pickle(subset_data, allow_pickle_arrays=True), f)
                print(f"{subset_name.capitalize()} subset for {base_filename} saved to: {subset_path}")

        print(f"Completed subsetting for {file_prefix}_{base_filename}: "
              f"{len(train_subset['sample_id'])} train, "
              f"{len(test_subset['sample_id'])} test, "
              f"{len(validation_subset['sample_id'])} validation samples")

    print(f"\n --------------------------------------------------------"
          f"\n All files have been subsetted separately!"
          f"\n Files are saved in: {save_path}"
          f"\n   - Training files: {train_dir}"
          f"\n   - Testing files: {test_dir}"
          f"\n   - Validation files: {validation_dir}"
          f"\n Format: {'HDF5' if args.use_hdf5 else 'JSON + NPY' if args.use_json_format else 'Pickle + BZ2'}"
          f"\n ----------------------------------------------------------\n")