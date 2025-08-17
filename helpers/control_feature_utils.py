import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

def Jaccard_similarity(a, b):
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return (intersection / union)

def get_timesteps_where_either_is_1(a, b):
    """
    Get indices where either a or b is 1.
    """
    return np.where((a == 1) | (b == 1))[0]


def n_steps_with_either_hit(a, b):
    """
    Get num steps where either a or b has a hit.
    """
    return (np.where((a == 1) | (b == 1))[0]).shape[0]

def hamming_distance(a, b, normalize=False):
    # normalize false divides by len(a),
    # otherwise, max edit uses num steps where either has a hit
    if len(a) != len(b):
        raise ValueError("Sequences must be of equal length")

    different_bits = sum(x != y for x, y in zip(a, b))

    if not normalize:
        return different_bits / len(a)
    else:
        n_hit_steps = n_steps_with_either_hit(a, b)
        return different_bits / n_hit_steps if n_hit_steps > 0 else 0.0

def get_accent_hits_from_velocities(velocity_flat, accent_thresh=0.6, compare_consecutives=False):
    """
    Extract accent hits from the velocity flat representation of HVO.
    :param velocity_flat: a (B, T, 1) or (T, 1) numpy array where the last  column represents the velocity of hits.
    :param use_median: if True, use the median velocity to determine accent hits, otherwise use 0.5
    :return:
    """

    return np.where(velocity_flat > accent_thresh, 1, 0)

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

def get_flat_from_streams(streams_hvo: np.ndarray) -> np.ndarray:
    """
    Convert HVO streams to flat representation.
    :param streams_hvo: (B, T, 9) numpy array of HVO streams
    :return: (B, T, 3) numpy array of flat representation
    """
    flat_hvo = np.zeros((streams_hvo.shape[0], streams_hvo.shape[1], 3))
    flat_hvo[:, :, 0] = streams_hvo[:, :, :3].sum(-1).clip(0, 1)  # hits
    # Get indices of max values along last dimension
    flat_hvo[:, :, 1] = streams_hvo[:, :, 3:6].max(-1).clip(0, 1) * flat_hvo[:, :, 0] # velocity
    max_indices = np.argmax(streams_hvo[:, :, 3:6], axis=-1)
    # Create a mask with True at max positions, False elsewhere
    mask = np.zeros_like(streams_hvo[:, :, 6:9], dtype=bool)
    np.put_along_axis(mask, max_indices[..., None], True, axis=-1)
    # Zero out all positions except the max ones
    flat_hvo[:, :, 2] = np.where(mask, streams_hvo[:, :, 6:9], 0).sum(axis=-1)* flat_hvo[:, :, 0]

    return flat_hvo

def run_inference(model, dataloader):
    model.eval()
    inputs_gt = []
    outputs_gt = []
    flat_outputs_gt = []
    metadata = []
    encoding_control1_tokens_gt = []
    encoding_control2_tokens_gt = []
    decoding_control1_tokens_gt = []
    decoding_control2_tokens_gt = []
    decoding_control3_tokens_gt = []
    outputs_pred = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_grooves_batch = batch[0]
            output_streams_gt_batch = batch[1]
            encoding_control1_tokens_batch = batch[2]
            encoding_control2_tokens_batch = batch[3]
            decoding_control1_tokens_batch = batch[4]
            decoding_control2_tokens_batch = batch[5]
            decoding_control3_tokens_batch = batch[6]
            indices_batch = batch[8]
            metadata.extend([dataloader.dataset.metadata[i] for i in indices_batch])

            # Store ground truth data
            inputs_gt.append(input_grooves_batch)
            outputs_gt.append(output_streams_gt_batch)
            encoding_control1_tokens_gt.append(encoding_control1_tokens_batch)
            encoding_control2_tokens_gt.append(encoding_control2_tokens_batch)
            decoding_control1_tokens_gt.append(decoding_control1_tokens_batch)
            decoding_control2_tokens_gt.append(decoding_control2_tokens_batch)
            decoding_control3_tokens_gt.append(decoding_control3_tokens_batch)

            # Forward pass
            hvo, latent_z = model.predict(
                flat_hvo_groove=input_grooves_batch,
                encoding_control1_token=encoding_control1_tokens_batch,
                encoding_control2_token=encoding_control2_tokens_batch,
                decoding_control1_token=decoding_control1_tokens_batch,
                decoding_control2_token=decoding_control2_tokens_batch,
                decoding_control3_token=decoding_control3_tokens_batch
            )
            # Store predicted data
            outputs_pred.append(hvo)

    # Convert to numpy arrays
    inputs_gt = torch.cat(inputs_gt, dim=0).cpu().numpy()
    outputs_gt = torch.cat(outputs_gt, dim=0).cpu().numpy()
    encoding_control1_tokens_gt = torch.cat(encoding_control1_tokens_gt, dim=0).cpu().numpy()
    encoding_control2_tokens_gt = torch.cat(encoding_control2_tokens_gt, dim=0).cpu().numpy()
    decoding_control1_tokens_gt = torch.cat(decoding_control1_tokens_gt, dim=0).cpu().numpy()
    decoding_control2_tokens_gt = torch.cat(decoding_control2_tokens_gt, dim=0).cpu().numpy()
    decoding_control3_tokens_gt = torch.cat(decoding_control3_tokens_gt, dim=0).cpu().numpy()
    outputs_pred = torch.cat(outputs_pred, dim=0).cpu().numpy()
    flat_outputs_gt = get_flat_from_streams(outputs_gt)
    flat_outputs_pred = get_flat_from_streams(outputs_pred)

    # Create a dictionary to hold all data
    data_dict = {
        'input_grooves_gt': inputs_gt,
        'output_streams_gt': outputs_gt,
        'flat_outputs_gt': flat_outputs_gt,
        'encoding_control1_tokens_gt': encoding_control1_tokens_gt,
        'encoding_control2_tokens_gt': encoding_control2_tokens_gt,
        'decoding_control1_tokens_gt': decoding_control1_tokens_gt,
        'decoding_control2_tokens_gt': decoding_control2_tokens_gt,
        'decoding_control3_tokens_gt': decoding_control3_tokens_gt,
        'output_streams_pred': outputs_pred,
        'flat_outputs_pred': flat_outputs_pred,
        'metadata': metadata
    }

    return data_dict

def extract_control_features(flat_input_hvo, flat_output_hvo, output_streams_hvo):
    """
    Extract control features from flat input HVO, flat output HVO, and output streams HVO.

    Args:
        flat_input_hvo: (B, T, 3) numpy array of flat input HVO
        flat_output_hvo: (B, T, 3) numpy array of flat output HVO
        output_streams_hvo: (B, T, 9) numpy array of output streams HVO

    Returns:
        Dictionary containing control features.
    """
    control_features = {}

    # Encoding Control 1: Hamming distance between groove and flat output hits
    ctrl1_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(flat_input_hvo[:, :, 0], flat_output_hvo[:, :, 0])])
    control_features['encoding_control1_tokens'] = TokenizeControls(ctrl1_untokenized, n_bins=33, low=0, high=1)

    # Encoding Control 2: Hamming distance between groove and flat output accented hits
    gt_accents = get_accent_hits_from_velocities(flat_input_hvo[:, :, 1], accent_thresh=0.75)
    pred_accents = get_accent_hits_from_velocities(flat_output_hvo[:, :, 1], accent_thresh=0.75)

    gt_pred_hit_union = np.array(
        [get_timesteps_where_either_is_1(a, b).shape[0] for a, b in zip(flat_input_hvo[:, :, 0], flat_output_hvo[:, :, 0])])
    # Normalize by the number of steps where either has a hit
    ctrl2_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(gt_accents, pred_accents)])

    control_features['decoding_control1_tokens'] = []
    for i, t in enumerate(gt_pred_hit_union):
        if t > 0:
            ctrl2_untokenized[i] = ctrl2_untokenized[i] / t * 32.0
        else:
            ctrl2_untokenized[i] = 0.0         # both silent

    control_features['encoding_control2_tokens'] = TokenizeControls(ctrl2_untokenized, n_bins=10, low=0, high=0.85)

    # Decoding Controls 1-3: Hamming distance between flat output and each output stream
    for i in range(3):
        out_stream_i = output_streams_hvo[:, :, i]
        ctrl_hammings_untokenized = np.array(
            [hamming_distance(a, b, normalize=True) for a, b in zip(flat_output_hvo[:, :, 0], out_stream_i)])
        control_features[f'decoding_control{i + 1}_tokens'] = TokenizeControls(
            ctrl_hammings_untokenized, n_bins=10, low=0, high=0.85
        )

    return control_features


def run_inference_and_extract_features(model, dataloader):
    model.eval()

    gt_data = {
        'input_grooves': [],
        'output_streams': [],
        'encoding_control1_tokens': [],
        'encoding_control2_tokens': [],
        'decoding_control1_tokens': [],
        'decoding_control2_tokens': [],
        'decoding_control3_tokens': []
    }

    pred_data = {
        'output_streams': [],
        'flat_outputs': [],
        'encoding_control1_tokens': [],  # hamming hit of groove to flat output
        'encoding_control2_tokens': [],  # Accent hits of groove to Accent hits of flat output
        'decoding_control1_tokens': [],
        'decoding_control2_tokens': [],
        'decoding_control3_tokens': []
    }

    metadata = []

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            input_grooves_batch = batch[0]
            output_streams_gt_batch = batch[1]
            encoding_control1_tokens_batch = batch[2]
            encoding_control2_tokens_batch = batch[3]
            decoding_control1_tokens_batch = batch[4]
            decoding_control2_tokens_batch = batch[5]
            decoding_control3_tokens_batch = batch[6]
            indices_batch = batch[8]
            metadata.extend([dataloader.dataset.metadata[i] for i in indices_batch])

            # Store ground truth data
            gt_data['input_grooves'].append(input_grooves_batch)
            gt_data['output_streams'].append(output_streams_gt_batch)
            gt_data['encoding_control1_tokens'].append(encoding_control1_tokens_batch)
            gt_data['encoding_control2_tokens'].append(encoding_control2_tokens_batch)
            gt_data['decoding_control1_tokens'].append(decoding_control1_tokens_batch)
            gt_data['decoding_control2_tokens'].append(decoding_control2_tokens_batch)
            gt_data['decoding_control3_tokens'].append(decoding_control3_tokens_batch)

            # Forward pass
            hvo, latent_z = model.predict(
                flat_hvo_groove=input_grooves_batch,
                encoding_control1_token=encoding_control1_tokens_batch,
                encoding_control2_token=encoding_control2_tokens_batch,
                decoding_control1_token=decoding_control1_tokens_batch,
                decoding_control2_token=decoding_control2_tokens_batch,
                decoding_control3_token=decoding_control3_tokens_batch
            )

            # Store predicted data
            pred_data['output_streams'].append(hvo)

    # convert to numpy    arrays
    gt_data = {k: torch.cat(v, dim=0).numpy() for k, v in gt_data.items()}
    pred_data['output_streams'] = torch.cat(pred_data['output_streams'], dim=0).numpy()
    pred_data['flat_outputs'] = get_flat_from_streams(pred_data['output_streams'])

    # Decoding control 1: Hamming distance between groove and flat output hits
    ctrl1_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(gt_data['input_grooves'][:, :, 0], pred_data['flat_outputs'][:, :, 0])])
    pred_data['encoding_control1_tokens'] = np.array(TokenizeControls(ctrl1_untokenized, n_bins=33, low=0, high=1))

    # Decoding control 2: Hamming distance between groove and flat output accented hits
    pred_accents = get_accent_hits_from_velocities(pred_data['flat_outputs'][:, :, 1], accent_thresh=0.75)
    gt_accents = get_accent_hits_from_velocities(gt_data['input_grooves'][:, :, 1], accent_thresh=0.75)
    ctr2_untokenized = np.array(
        [hamming_distance(a, b, normalize=True) for a, b in zip(gt_accents, pred_accents)])
    pred_data['encoding_control2_tokens'] = np.array(TokenizeControls(ctr2_untokenized, n_bins=10, low=0, high=0.85))

    # Decoding controls 1-3: Hamming distance between flat output and each output stream
    for i in range(3):
        out_stream_i = pred_data['output_streams'][:, :, i]
        ctrl_hammings_untokenized = np.array(
            [hamming_distance(a, b, normalize=True) for a, b in zip(pred_data['flat_outputs'][:, :, 0], pred_data['output_streams'][:, :, i])])
        pred_data[f'decoding_control{i + 1}_tokens'] = np.array(TokenizeControls(
            ctrl_hammings_untokenized, n_bins=10, low=0, high=0.85
        ))

    return gt_data, pred_data, metadata


def run_inference_and_extract_features_flex_model(model, dataloader):
    model.eval()

    gt_data = {
        'input_grooves': [],
        'output_streams': [],
        'encoding_control1_tokens': [],
        'encoding_control2_tokens': [],
        'decoding_control1_tokens': [],
        'decoding_control2_tokens': [],
        'decoding_control3_tokens': []
    }

    pred_data = {
        'output_streams': [],
        'flat_outputs': [],
        'encoding_control1_tokens': [],  # hamming hit of groove to flat output
        'encoding_control2_tokens': [],  # Accent hits of groove to Accent hits of flat output
        'decoding_control1_tokens': [],
        'decoding_control2_tokens': [],
        'decoding_control3_tokens': []
    }

    metadata = []

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            input_grooves_batch = batch[0]
            output_streams_gt_batch = batch[1]
            encoding_control_tokens = batch[2]
            decoding_control_tokens = batch[3]
            encoding_control1_tokens_batch = encoding_control_tokens[:, 0]
            encoding_control2_tokens_batch = encoding_control_tokens[:, 1]
            decoding_control1_tokens_batch = decoding_control_tokens[:, 0]
            decoding_control2_tokens_batch = decoding_control_tokens[:, 1]
            decoding_control3_tokens_batch = decoding_control_tokens[:, 2]
            indices_batch = batch[-1]
            metadata.extend([dataloader.dataset.metadata[i] for i in indices_batch])

            # Store ground truth data
            gt_data['input_grooves'].append(input_grooves_batch)
            gt_data['output_streams'].append(output_streams_gt_batch)
            gt_data['encoding_control1_tokens'].append(encoding_control1_tokens_batch)
            gt_data['encoding_control2_tokens'].append(encoding_control2_tokens_batch)
            gt_data['decoding_control1_tokens'].append(decoding_control1_tokens_batch)
            gt_data['decoding_control2_tokens'].append(decoding_control2_tokens_batch)
            gt_data['decoding_control3_tokens'].append(decoding_control3_tokens_batch)

            # Forward pass
            hvo, latent_z = model.predict(
                flat_hvo_groove=input_grooves_batch,
                encoding_control_tokens=encoding_control_tokens,
                decoding_control_tokens=decoding_control_tokens,
            )

            # Store predicted data
            pred_data['output_streams'].append(hvo)

    # convert to numpy    arrays
    gt_data = {k: torch.cat(v, dim=0).numpy() for k, v in gt_data.items()}
    pred_data['output_streams'] = torch.cat(pred_data['output_streams'], dim=0).numpy()
    pred_data['flat_outputs'] = get_flat_from_streams(pred_data['output_streams'])

    # Decoding control 1: Hamming distance between groove and flat output hits
    ctrl1_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(gt_data['input_grooves'][:, :, 0], pred_data['flat_outputs'][:, :, 0])])
    pred_data['encoding_control1_tokens'] = np.array(TokenizeControls(ctrl1_untokenized, n_bins=33, low=0, high=1))

    # Decoding control 2: Hamming distance between groove and flat output accented hits
    pred_accents = get_accent_hits_from_velocities(pred_data['flat_outputs'][:, :, 1], accent_thresh=0.75)
    gt_accents = get_accent_hits_from_velocities(gt_data['input_grooves'][:, :, 1], accent_thresh=0.75)
    ctr2_untokenized = np.array(
        [hamming_distance(a, b, normalize=True) for a, b in zip(gt_accents, pred_accents)])
    pred_data['encoding_control2_tokens'] = np.array(TokenizeControls(ctr2_untokenized, n_bins=10, low=0, high=0.85))

    # Decoding controls 1-3: Hamming distance between flat output and each output stream
    for i in range(3):
        out_stream_i = pred_data['output_streams'][:, :, i]
        ctrl_hammings_untokenized = np.array(
            [hamming_distance(a, b, normalize=True) for a, b in zip(pred_data['flat_outputs'][:, :, 0], pred_data['output_streams'][:, :, i])])
        pred_data[f'decoding_control{i + 1}_tokens'] = np.array(TokenizeControls(
            ctrl_hammings_untokenized, n_bins=10, low=0, high=0.85
        ))

    return gt_data, pred_data, metadata

def run_inference_flex_model(model, dataloader):
    model.eval()
    inputs_gt = []
    outputs_gt = []
    flat_outputs_gt = []
    metadata = []
    encoding_control1_tokens_gt = []
    encoding_control2_tokens_gt = []
    decoding_control1_tokens_gt = []
    decoding_control2_tokens_gt = []
    decoding_control3_tokens_gt = []
    outputs_pred = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_grooves_batch = batch[0]
            output_streams_gt_batch = batch[1]
            encoding_control_tokens = batch[2]
            decoding_control_tokens = batch[3]
            encoding_control1_tokens_batch = encoding_control_tokens[:, 0]
            encoding_control2_tokens_batch = encoding_control_tokens[:, 1]
            decoding_control1_tokens_batch = decoding_control_tokens[:, 0]
            decoding_control2_tokens_batch = decoding_control_tokens[:, 1]
            decoding_control3_tokens_batch = decoding_control_tokens[:, 2]
            indices_batch = batch[-1]
            metadata.extend([dataloader.dataset.metadata[i] for i in indices_batch])

            # Store ground truth data
            inputs_gt.append(input_grooves_batch)
            outputs_gt.append(output_streams_gt_batch)
            encoding_control1_tokens_gt.append(encoding_control1_tokens_batch)
            encoding_control2_tokens_gt.append(encoding_control2_tokens_batch)
            decoding_control1_tokens_gt.append(decoding_control1_tokens_batch)
            decoding_control2_tokens_gt.append(decoding_control2_tokens_batch)
            decoding_control3_tokens_gt.append(decoding_control3_tokens_batch)

            # Forward pass
            hvo, latent_z = model.predict(
                flat_hvo_groove=input_grooves_batch,
                encoding_control_tokens=encoding_control_tokens,
                decoding_control_tokens=decoding_control_tokens,
            )
            # Store predicted data
            outputs_pred.append(hvo)

    # Convert to numpy arrays
    inputs_gt = torch.cat(inputs_gt, dim=0).cpu().numpy()
    outputs_gt = torch.cat(outputs_gt, dim=0).cpu().numpy()
    encoding_control1_tokens_gt = torch.cat(encoding_control1_tokens_gt, dim=0).cpu().numpy()
    encoding_control2_tokens_gt = torch.cat(encoding_control2_tokens_gt, dim=0).cpu().numpy()
    decoding_control1_tokens_gt = torch.cat(decoding_control1_tokens_gt, dim=0).cpu().numpy()
    decoding_control2_tokens_gt = torch.cat(decoding_control2_tokens_gt, dim=0).cpu().numpy()
    decoding_control3_tokens_gt = torch.cat(decoding_control3_tokens_gt, dim=0).cpu().numpy()
    outputs_pred = torch.cat(outputs_pred, dim=0).cpu().numpy()
    flat_outputs_gt = get_flat_from_streams(outputs_gt)
    flat_outputs_pred = get_flat_from_streams(outputs_pred)

    # Create a dictionary to hold all data
    data_dict = {
        'input_grooves_gt': inputs_gt,
        'output_streams_gt': outputs_gt,
        'flat_outputs_gt': flat_outputs_gt,
        'encoding_control1_tokens_gt': encoding_control1_tokens_gt,
        'encoding_control2_tokens_gt': encoding_control2_tokens_gt,
        'decoding_control1_tokens_gt': decoding_control1_tokens_gt,
        'decoding_control2_tokens_gt': decoding_control2_tokens_gt,
        'decoding_control3_tokens_gt': decoding_control3_tokens_gt,
        'output_streams_pred': outputs_pred,
        'flat_outputs_pred': flat_outputs_pred,
        'metadata': metadata
    }

    return data_dict

def extract_control_features_flex_model(flat_input_hvo, flat_output_hvo, output_streams_hvo):
    """
    Extract control features from flat input HVO, flat output HVO, and output streams HVO.

    Args:
        flat_input_hvo: (B, T, 3) numpy array of flat input HVO
        flat_output_hvo: (B, T, 3) numpy array of flat output HVO
        output_streams_hvo: (B, T, 9) numpy array of output streams HVO

    Returns:
        Dictionary containing control features.
    """
    control_features = {}

    # Encoding Control 1: Hamming distance between groove and flat output hits
    ctrl1_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(flat_input_hvo[:, :, 0], flat_output_hvo[:, :, 0])])
    control_features['encoding_control1_tokens'] = TokenizeControls(ctrl1_untokenized, n_bins=33, low=0, high=1)

    # Encoding Control 2: Hamming distance between groove and flat output accented hits
    gt_accents = get_accent_hits_from_velocities(flat_input_hvo[:, :, 1], accent_thresh=0.75)
    pred_accents = get_accent_hits_from_velocities(flat_output_hvo[:, :, 1], accent_thresh=0.75)

    gt_pred_hit_union = np.array(
        [get_timesteps_where_either_is_1(a, b).shape[0] for a, b in zip(flat_input_hvo[:, :, 0], flat_output_hvo[:, :, 0])])
    # Normalize by the number of steps where either has a hit
    ctrl2_untokenized = np.array(
        [hamming_distance(a, b) for a, b in zip(gt_accents, pred_accents)])

    control_features['decoding_control1_tokens'] = []
    for i, t in enumerate(gt_pred_hit_union):
        if t > 0:
            ctrl2_untokenized[i] = ctrl2_untokenized[i] / t * 32.0
        else:
            ctrl2_untokenized[i] = 0.0         # both silent

    control_features['encoding_control2_tokens'] = TokenizeControls(ctrl2_untokenized, n_bins=5, low=0, high=0.85)

    # Decoding Controls 1-3: Hamming distance between flat output and each output stream
    for i in range(3):
        out_stream_i = output_streams_hvo[:, :, i]
        ctrl_hammings_untokenized = np.array(
            [hamming_distance(a, b, normalize=True) for a, b in zip(flat_output_hvo[:, :, 0], out_stream_i)])
        control_features[f'decoding_control{i + 1}_tokens'] = TokenizeControls(
            ctrl_hammings_untokenized, n_bins=6, low=0, high=0.85
        )

    return control_features
