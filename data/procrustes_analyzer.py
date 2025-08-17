import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes


def rhythm_procrustes_analysis(rhythm1, rhythm2, padding_method='silent', return_details=False):
    """
    Compare two rhythms using Procrustes analysis with interpretable measures.

    Parameters:
    -----------
    rhythm1, rhythm2 : list of tuples or 2D arrays
        Each rhythm as [(r1, theta1), (r2, theta2), ...] or array of shape (n, 2)
        where r is velocity/intensity and theta is timing in radians (0 to 4π for 2-bar loop)

    padding_method : str, default 'silent'
        'silent': Pad shorter rhythm with silent onsets (r=0) at center
        'repeat': Repeat points from shorter rhythm cyclically
        'reference': For <3 onsets, pad both rhythms to 3 points with (r=1, theta=0) reference points

    return_details : bool, default False
        If True, returns additional information about the analysis

    Returns:
    --------
    results : dict
        - 'rotation': Rotation angle in radians (timing shift)
        - 'scale': Scale factor (velocity/intensity scaling)
        - 'goodness_of_fit': How well rhythms match after alignment (0-1, higher is better)
        - 'comparable': Whether analysis is meaningful (False for degenerate cases)
        If return_details=True, also includes:
        - 'aligned_rhythm1', 'aligned_rhythm2': The processed rhythms used in analysis
        - 'transformation_matrix': The Procrustes transformation matrix
    """

    # Convert inputs to numpy arrays
    r1 = np.array([point[0] for point in rhythm1])
    theta1 = np.array([point[1] for point in rhythm1])
    r2 = np.array([point[0] for point in rhythm2])
    theta2 = np.array([point[1] for point in rhythm2])

    n1, n2 = len(r1), len(r2)

    # Handle degenerate cases
    if n1 == 0 and n2 == 0:
        return {'rotation': np.nan, 'scale': np.nan, 'goodness_of_fit': np.nan, 'comparable': False}

    if n1 == 0 or n2 == 0:
        return {'rotation': np.nan, 'scale': np.nan, 'goodness_of_fit': 0.0, 'comparable': False}

    # Special handling for reference padding method with <3 onsets
    if padding_method == 'reference' and (n1 < 3 or n2 < 3):
        r1_proc, theta1_proc, r2_proc, theta2_proc = _pad_with_reference(
            r1.copy(), theta1.copy(), r2.copy(), theta2.copy()
        )
    else:
        # For non-reference methods, handle 1v1 case specially
        if n1 == 1 and n2 == 1:
            # Can only compute scale, rotation is arbitrary
            scale = r2[0] / r1[0] if r1[0] != 0 else np.nan
            rotation = theta2[0] - theta1[0]  # One possible rotation
            # Normalize rotation to [-π, π]
            rotation = ((rotation + np.pi) % (2 * np.pi)) - np.pi
            return {'rotation': rotation, 'scale': scale, 'goodness_of_fit': 1.0, 'comparable': True}

        # Make copies for processing
        r1_proc, theta1_proc = r1.copy(), theta1.copy()
        r2_proc, theta2_proc = r2.copy(), theta2.copy()

    # Make copies for processing
    r1_proc, theta1_proc = r1.copy(), theta1.copy()
    r2_proc, theta2_proc = r2.copy(), theta2.copy()

    # Pad shorter rhythm to match longer one
    if n1 != n2:
        if padding_method == 'silent':
            r1_proc, theta1_proc, r2_proc, theta2_proc = _pad_with_silent(
                r1_proc, theta1_proc, r2_proc, theta2_proc
            )
        elif padding_method == 'repeat':
            r1_proc, theta1_proc, r2_proc, theta2_proc = _pad_with_repeat(
                r1_proc, theta1_proc, r2_proc, theta2_proc
            )
        elif padding_method == 'reference':
            r1_proc, theta1_proc, r2_proc, theta2_proc = _pad_with_reference(
                r1_proc, theta1_proc, r2_proc, theta2_proc
            )
        else:
            raise ValueError("padding_method must be 'silent', 'repeat', or 'reference'")

    # Convert to Cartesian coordinates
    x1 = r1_proc * np.cos(theta1_proc)
    y1 = r1_proc * np.sin(theta1_proc)
    x2 = r2_proc * np.cos(theta2_proc)
    y2 = r2_proc * np.sin(theta2_proc)

    # Create shape matrices
    shape1 = np.column_stack([x1, y1])
    shape2 = np.column_stack([x2, y2])

    # Center the shapes (remove translation)
    centroid1 = np.mean(shape1, axis=0)
    centroid2 = np.mean(shape2, axis=0)
    shape1_centered = shape1 - centroid1
    shape2_centered = shape2 - centroid2

    # Compute scales
    scale1 = np.sqrt(np.sum(shape1_centered ** 2))
    scale2 = np.sqrt(np.sum(shape2_centered ** 2))

    if scale1 == 0 or scale2 == 0:
        return {'rotation': np.nan, 'scale': np.nan, 'goodness_of_fit': np.nan, 'comparable': False}

    # Normalize shapes
    shape1_normalized = shape1_centered / scale1
    shape2_normalized = shape2_centered / scale2

    # Find optimal rotation using Procrustes
    rotation_matrix, _ = orthogonal_procrustes(shape1_normalized, shape2_normalized)

    # Extract rotation angle
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Compute scale factor
    scale_factor = scale2 / scale1

    # Apply transformation to shape1
    shape1_transformed = scale_factor * (shape1_normalized @ rotation_matrix) + centroid2

    # Compute goodness of fit (1 - normalized RMSE)
    mse = np.mean((shape1_transformed - shape2) ** 2)
    max_possible_mse = np.mean((shape2 - centroid2) ** 2) + np.mean((centroid1 - centroid2) ** 2)
    if max_possible_mse > 0:
        goodness_of_fit = max(0, 1 - mse / max_possible_mse)
    else:
        goodness_of_fit = 1.0

    results = {
        'rotation': rotation_angle,
        'scale': scale_factor,
        'goodness_of_fit': goodness_of_fit,
        'comparable': True
    }

    if return_details:
        results.update({
            'aligned_rhythm1': list(zip(r1_proc, theta1_proc)),
            'aligned_rhythm2': list(zip(r2_proc, theta2_proc)),
            'transformation_matrix': rotation_matrix,
            'centroid1': centroid1,
            'centroid2': centroid2
        })

    return results


def _pad_with_reference(r1, theta1, r2, theta2):
    """Pad rhythms with <3 onsets using reference points at (r=1, theta=0)"""
    n1, n2 = len(r1), len(r2)

    # Ensure both rhythms have at least 3 points by adding reference points
    if n1 < 3:
        padding_needed = 3 - n1
        r1_padded = np.concatenate([r1, np.ones(padding_needed)])
        theta1_padded = np.concatenate([theta1, np.zeros(padding_needed)])
    else:
        r1_padded, theta1_padded = r1, theta1

    if n2 < 3:
        padding_needed = 3 - n2
        r2_padded = np.concatenate([r2, np.ones(padding_needed)])
        theta2_padded = np.concatenate([theta2, np.zeros(padding_needed)])
    else:
        r2_padded, theta2_padded = r2, theta2

    # Now handle length differences using silent padding if needed
    n1_new, n2_new = len(r1_padded), len(r2_padded)
    if n1_new != n2_new:
        return _pad_with_silent(r1_padded, theta1_padded, r2_padded, theta2_padded)

    return r1_padded, theta1_padded, r2_padded, theta2_padded


def _pad_with_silent(r1, theta1, r2, theta2):
    """Pad shorter rhythm with silent onsets at center"""
    n1, n2 = len(r1), len(r2)

    if n1 < n2:
        # Calculate center of rhythm1
        if np.sum(r1) > 0:
            center1 = np.sum(r1 * theta1) / np.sum(r1)
        else:
            center1 = np.mean(theta1)

        # Pad rhythm1
        padding_needed = n2 - n1
        r1_padded = np.concatenate([r1, np.zeros(padding_needed)])
        theta1_padded = np.concatenate([theta1, np.full(padding_needed, center1)])
        return r1_padded, theta1_padded, r2, theta2

    elif n2 < n1:
        # Calculate center of rhythm2
        if np.sum(r2) > 0:
            center2 = np.sum(r2 * theta2) / np.sum(r2)
        else:
            center2 = np.mean(theta2)

        # Pad rhythm2
        padding_needed = n1 - n2
        r2_padded = np.concatenate([r2, np.zeros(padding_needed)])
        theta2_padded = np.concatenate([theta2, np.full(padding_needed, center2)])
        return r1, theta1, r2_padded, theta2_padded

    return r1, theta1, r2, theta2


def _pad_with_repeat(r1, theta1, r2, theta2):
    """Pad shorter rhythm by repeating points cyclically"""
    n1, n2 = len(r1), len(r2)

    if n1 < n2:
        # Repeat rhythm1 cyclically
        repetitions = (n2 + n1 - 1) // n1  # Ceiling division
        r1_repeated = np.tile(r1, repetitions)[:n2]
        theta1_repeated = np.tile(theta1, repetitions)[:n2]
        return r1_repeated, theta1_repeated, r2, theta2

    elif n2 < n1:
        # Repeat rhythm2 cyclically
        repetitions = (n1 + n2 - 1) // n2  # Ceiling division
        r2_repeated = np.tile(r2, repetitions)[:n1]
        theta2_repeated = np.tile(theta2, repetitions)[:n1]
        return r1, theta1, r2_repeated, theta2_repeated

    return r1, theta1, r2, theta2


if __name__ == '__main__':
    import numpy as np
    import yaml
    from matplotlib import pyplot as plt
    import logging

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import os

    os.environ.pop("MPLDEBUG", None)
    import tqdm
    import torch
    from torch.utils.data import DataLoader
    from model import TripleStreamsVAE
    from data import get_triplestream_dataset
    from helpers.control_feature_utils import run_inference_and_extract_features, run_inference, \
        extract_control_features

    config = yaml.safe_load(open('helpers/configs/TripleStreams_0.5.yaml', 'r'))

    is_testing = True

    dataset = get_triplestream_dataset(
        config=config,
        subset_tag="validation",
        use_cached=True,
        downsampled_size=2000 if is_testing else None,
        print_logs=False  # <---  Set to True to print dataset loading logs
    )

    # input_groove = dataset.input_grooves[:2, :, :]

    def convert_to_polar_rhythm(single_voice_groove):
        """
        Convert input groove from Cartesian to polar coordinates.
        Each row is a note with (velocity, timing_in_radians).
        """
        assert len(single_voice_groove.shape) == 2

        timings = (torch.linspace(0, 31, 32, dtype=torch.float32, device=single_voice_groove.device) * single_voice_groove[:, 0] + single_voice_groove[:, 2]) / 32 * 2.0 * np.pi
        vels = single_voice_groove[:, 1] * single_voice_groove[:, 0]

        return [(vel, timing) for vel, timing in zip(vels.cpu().numpy(), timings.cpu().numpy())]

    polar_i = convert_to_polar_rhythm(dataset.input_grooves[1, :, :])
    zeros = [(1, 0)]
    zeros.extend([(0, 0)] * 31)
    polar_o = convert_to_polar_rhythm(dataset.flat_output_streams[1, :, :])

    rhythm_procrustes_analysis(polar_i, polar_o)
    rhythm_procrustes_analysis(zeros, polar_o)

# # Example usage and testing
# if __name__ == "__main__":
#     # Example rhythms: (velocity, timing_in_radians)
#     # 2-bar loop spans 0 to 4π radians
#
#     # Dense rhythm (kick on 1, snare on 3, hi-hats on off-beats)
#     rhythm_dense = [
#         (1.0, 0),  # strong kick on beat 1
#         (0.3, np.pi / 2),  # hi-hat
#         (0.8, np.pi),  # snare on beat 3
#         (0.3, 3 * np.pi / 2),  # hi-hat
#         (0.6, 2 * np.pi),  # kick on beat 5
#         (0.3, 5 * np.pi / 2),  # hi-hat
#         (0.8, 3 * np.pi),  # snare on beat 7
#         (0.3, 7 * np.pi / 2)  # hi-hat
#     ]
#
#     # Sparse rhythm (just kick and snare)
#     rhythm_sparse = [
#         (1.0, 0),  # kick on beat 1
#         (0.8, np.pi),  # snare on beat 3
#         (0.6, 2 * np.pi)  # kick on beat 5
#     ]
#
#     print("Comparing dense vs sparse rhythms:")
#     print("\nUsing silent padding:")
#     result1 = rhythm_procrustes_analysis(rhythm_sparse, rhythm_dense, padding_method='silent')
#     print(f"Rotation (timing shift): {result1['rotation']:.3f} radians ({result1['rotation'] * 180 / np.pi:.1f}°)")
#     print(f"Scale factor: {result1['scale']:.3f}")
#     print(f"Goodness of fit: {result1['goodness_of_fit']:.3f}")
#
#     print("\nUsing repeat padding:")
#     result2 = rhythm_procrustes_analysis(rhythm_sparse, rhythm_dense, padding_method='repeat')
#     print(f"Rotation (timing shift): {result2['rotation']:.3f} radians ({result2['rotation'] * 180 / np.pi:.1f}°)")
#     print(f"Scale factor: {result2['scale']:.3f}")
#     print(f"Goodness of fit: {result2['goodness_of_fit']:.3f}")
#
#     print("\nUsing reference padding:")
#     result3 = rhythm_procrustes_analysis(rhythm_sparse, rhythm_dense, padding_method='reference')
#     print(f"Rotation (timing shift): {result3['rotation']:.3f} radians ({result3['rotation'] * 180 / np.pi:.1f}°)")
#     print(f"Scale factor: {result3['scale']:.3f}")
#     print(f"Goodness of fit: {result3['goodness_of_fit']:.3f}")
#
#     # Test edge cases
#     print("\nTesting edge cases:")
#
#     # Empty rhythms
#     empty_result = rhythm_procrustes_analysis([], rhythm_sparse)
#     print(f"Empty vs sparse: comparable = {empty_result['comparable']}")
#
#     # Single onset rhythms with reference padding
#     single1 = [(1.0, 0)]
#     single2 = [(0.8, np.pi / 4)]
#     single_result = rhythm_procrustes_analysis(single1, single2)
#     single_ref_result = rhythm_procrustes_analysis(single1, single2, padding_method='reference')
#     print(
#         f"Single vs single (standard): rotation = {single_result['rotation']:.3f}, scale = {single_result['scale']:.3f}")
#     print(
#         f"Single vs single (reference): rotation = {single_ref_result['rotation']:.3f}, scale = {single_ref_result['scale']:.3f}")
#
#     # Two onset rhythms
#     two1 = [(1.0, 0), (0.5, np.pi)]
#     two2 = [(0.8, np.pi / 4), (0.6, 5 * np.pi / 4)]
#     two_ref_result = rhythm_procrustes_analysis(two1, two2, padding_method='reference')
#     print(f"Two vs two (reference): rotation = {two_ref_result['rotation']:.3f}, scale = {two_ref_result['scale']:.3f}")
#     print(f"Two vs two (reference): goodness of fit = {two_ref_result['goodness_of_fit']:.3f}")