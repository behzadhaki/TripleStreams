import numpy as np
from collections import Counter

# ================== Center of Mass in Polar Coordinates ==================

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

# ================== SYNCOPATION METRICS ==================

def get_syncopation_hvo(sequences):
    """
    Calculate syncopation using Longuet-Higgins & Lee 1984 metric.

    Args:
        sequences: numpy array of shape [batch, 32] with hits or velocities

    Returns:
        numpy array of shape [batch] with normalized syncopation scores
    """

    # Longuet-Higgins and Lee 1984 metric profile for 32 steps (2 bars of 4/4)
    metrical_profile = np.array([5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                                 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1])
    max_syncopation = 30.0

    # Active notes (non-zero values)
    is_active = sequences > 0

    # Create shifted versions for next and next+1 positions
    next_active = np.roll(is_active, -1, axis=1)  # (i+1) % 32
    next2_active = np.roll(is_active, -2, axis=1)  # (i+2) % 32

    # Get metrical profiles for current, next, and next+1 positions
    current_metric = metrical_profile[np.newaxis, :]  # Shape: [1, 32]
    next_metric = np.roll(metrical_profile, -1)  # Shape: [32]
    next2_metric = np.roll(metrical_profile, -2)  # Shape: [32]

    # Condition 1: next position is silent and has higher metric weight
    cond1 = (is_active & ~next_active & (next_metric > current_metric))

    # Condition 2: next+1 position is silent and has higher metric weight
    cond2 = (is_active & ~next2_active & (next2_metric > current_metric))

    # Calculate syncopation contributions weighted by sequence values
    sync1 = cond1 * (next_metric - current_metric) * sequences
    sync2 = cond2 * (next2_metric - current_metric) * sequences

    # Sum across time steps for each batch and normalize
    syncopation = np.sum(sync1 + sync2, axis=1)
    return np.round(syncopation / max_syncopation, 5)


# ================== WEAK/STRONG RATIO METRICS ==================

def get_weak_to_strong_ratio(sequences):
    """
    Calculate normalized weak-to-strong ratio for binary hit sequences.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with normalized weak-to-strong ratios (0-1)
    """

    # Define strong and weak positions
    strong_positions = np.array([0, 4, 8, 12, 16, 20, 24, 28])
    weak_positions = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                               17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31])

    # Extract hits at strong and weak positions using advanced indexing
    strong_hits = sequences[:, strong_positions]  # Shape: [batch, 8]
    weak_hits = sequences[:, weak_positions]  # Shape: [batch, 24]

    # Sum hits for each batch
    strong_hit_count = np.sum(strong_hits, axis=1)  # Shape: [batch]
    weak_hit_count = np.sum(weak_hits, axis=1)  # Shape: [batch]

    # Avoid division by zero by setting zero counts to 1
    strong_hit_count = np.where(strong_hit_count == 0, 1, strong_hit_count)

    # Calculate ratio
    ratio = weak_hit_count / strong_hit_count

    # Normalize by maximum possible ratio
    # Max ratio occurs when all weak positions are hit (24) and only 1 strong position is hit
    max_ratio = len(weak_positions) / 1  # 24/1 = 24
    normalized_ratio = ratio / max_ratio

    return np.round(normalized_ratio, 5)


def get_weak_to_all_ratio(sequences):
    """
    Calculate normalized weak-to-all ratio for binary hit sequences.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with normalized weak-to-all ratios (0-1)
    """

    # Define weak positions
    weak_positions = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                               17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31])

    # Extract hits at weak positions
    weak_hits = sequences[:, weak_positions]  # Shape: [batch, 24]

    # Count weak hits and total hits for each batch
    weak_hit_count = np.sum(weak_hits, axis=1)  # Shape: [batch]
    total_hit_count = np.sum(sequences, axis=1)  # Shape: [batch]

    # Avoid division by zero by setting zero total counts to 1
    total_hit_count = np.where(total_hit_count == 0, 1, total_hit_count)

    # Calculate ratio
    ratio = weak_hit_count / total_hit_count

    # Already normalized since max ratio is 1 (all hits are weak)
    # But we can be explicit about it
    max_ratio = 1.0  # When all hits are on weak positions
    normalized_ratio = ratio / max_ratio

    return np.round(normalized_ratio, 5)


# ================== BALANCE, EVENNESS, ENTROPY METRICS ==================

def get_balance(sequences):
    """
    Calculate balance using DFT zeroth coefficient magnitude.
    Balance measures how evenly distributed the onsets are around the circle.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with balance values (0-1, higher = more balanced)
    """

    batch_size, n_steps = sequences.shape

    # Create complex representation on unit circle
    # Each onset position becomes a point: exp(2πi * position / n_steps)
    angles = 2 * np.pi * np.arange(n_steps) / n_steps
    complex_points = np.exp(1j * angles)  # Shape: [32]

    # For each sequence, sum the complex points where onsets occur
    # This is essentially the zeroth DFT coefficient (DC component)
    complex_sums = np.sum(sequences * complex_points[np.newaxis, :], axis=1)

    # Balance is the magnitude of this sum, normalized
    balance_raw = np.abs(complex_sums)

    # Normalize: max balance occurs when all onsets are at same position
    # Min balance occurs when onsets are maximally spread out
    max_balance = np.sum(sequences, axis=1)  # All onsets at same position

    # Avoid division by zero
    max_balance = np.where(max_balance == 0, 1, max_balance)

    # Invert so that 1 = maximally balanced (evenly distributed)
    # 0 = minimally balanced (all at same position)
    balance_normalized = 1 - (balance_raw / max_balance)

    return np.round(balance_normalized, 5)


def get_evenness(sequences):
    """
    Calculate evenness using DFT first coefficient magnitude.
    Evenness measures how regular the spacing between onsets is.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with evenness values (0-1, higher = more even)
    """

    batch_size, n_steps = sequences.shape

    # Create complex representation for first harmonic
    # First coefficient looks at exp(2πi * 1 * position / n_steps)
    angles = 2 * np.pi * np.arange(n_steps) / n_steps
    complex_points = np.exp(1j * angles)  # Shape: [32]

    # Calculate first DFT coefficient
    first_coeff = np.sum(sequences * complex_points[np.newaxis, :], axis=1)

    # Evenness is related to magnitude of first coefficient
    evenness_raw = np.abs(first_coeff)

    # Normalize by number of onsets
    n_onsets = np.sum(sequences, axis=1)
    n_onsets = np.where(n_onsets == 0, 1, n_onsets)  # Avoid division by zero

    evenness_normalized = evenness_raw / n_onsets

    # Scale to 0-1 range (empirically, max seems to be around n_steps/2 for even spacing)
    evenness_normalized = evenness_normalized / (n_steps / 4)  # Approximate normalization
    evenness_normalized = np.clip(evenness_normalized, 0, 1)

    return np.round(evenness_normalized, 5)


def get_interonset_interval_entropy(sequences):
    """
    Calculate interonset interval entropy.
    Measures unpredictability of the distribution of durations between onsets.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with IOI entropy values (0-max_entropy)
    """

    batch_size, n_steps = sequences.shape
    entropies = []

    for batch_idx in range(batch_size):
        sequence = sequences[batch_idx]

        # Find onset positions
        onset_positions = np.where(sequence > 0)[0]

        if len(onset_positions) < 2:
            # Need at least 2 onsets to calculate intervals
            entropies.append(0.0)
            continue

        # Calculate interonset intervals (with wraparound for cyclic rhythm)
        intervals = []
        for i in range(len(onset_positions)):
            current_pos = onset_positions[i]
            next_pos = onset_positions[(i + 1) % len(onset_positions)]

            # Calculate interval with wraparound
            if next_pos > current_pos:
                interval = next_pos - current_pos
            else:
                interval = (n_steps - current_pos) + next_pos

            intervals.append(interval)

        # Calculate entropy of interval distribution
        if len(intervals) == 0:
            entropies.append(0.0)
        else:
            # Count interval frequencies
            interval_counts = Counter(intervals)
            total_intervals = len(intervals)

            # Calculate entropy: H = -sum(p * log2(p))
            entropy = 0.0
            for count in interval_counts.values():
                p = count / total_intervals
                if p > 0:
                    entropy -= p * np.log2(p)

            entropies.append(entropy)

    return np.round(np.array(entropies), 5)


def get_normalized_interonset_interval_entropy(sequences):
    """
    Calculate normalized interonset interval entropy (0-1 range).

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        numpy array of shape [batch] with normalized IOI entropy values (0-1)
    """

    entropies = get_interonset_interval_entropy(sequences)

    # Theoretical maximum entropy occurs when all intervals are equally likely
    # For n onsets, we can have at most n different intervals
    # Max entropy = log2(n) where n is number of unique intervals possible

    # In practice, for cyclic rhythms of length 32, max entropy is around log2(32/2) ≈ 4
    # But this depends on the number of onsets, so we use a more conservative estimate
    max_entropy = np.log2(32)  # Theoretical maximum

    # Normalize to 0-1
    normalized_entropies = entropies / max_entropy
    normalized_entropies = np.clip(normalized_entropies, 0, 1)

    return np.round(normalized_entropies, 5)


# ================== PHASE-SENSITIVE METRICS ==================

def get_phase_metrics(sequences):
    """
    Calculate phase-sensitive metrics for rhythmic sequences.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        dict with phase-related metrics
    """

    batch_size, n_steps = sequences.shape

    # Create complex representation on unit circle
    angles = 2 * np.pi * np.arange(n_steps) / n_steps
    complex_points = np.exp(1j * angles)  # Shape: [32]

    # Calculate first few DFT coefficients (including phase)
    coeffs = {}
    for k in range(4):  # 0th through 3rd harmonics
        harmonic_points = np.exp(1j * k * angles)
        coeffs[k] = np.sum(sequences * harmonic_points[np.newaxis, :], axis=1)

    results = {}

    # 1. Phase angles of key harmonics
    results['phase_0'] = np.round(np.angle(coeffs[0]), 5)  # Phase of balance
    results['phase_1'] = np.round(np.angle(coeffs[1]), 5)  # Phase of evenness
    results['phase_2'] = np.round(np.angle(coeffs[2]), 5)  # 2nd harmonic phase

    # 2. Circular mean (center of mass) - captures overall phase tendency
    circular_mean_complex = coeffs[1] / (np.abs(coeffs[1]) + 1e-10)
    results['circular_mean_phase'] = np.round(np.angle(circular_mean_complex), 5)

    # 3. Phase dispersion - how spread out are the onsets around the circle
    # This uses the circular variance formula: 1 - |mean resultant vector|
    n_onsets = np.sum(sequences, axis=1)
    n_onsets = np.where(n_onsets == 0, 1, n_onsets)  # Avoid division by zero

    mean_resultant_length = np.abs(coeffs[1]) / n_onsets
    phase_dispersion = 1 - mean_resultant_length
    results['phase_dispersion'] = np.round(phase_dispersion, 5)

    # 4. Phase coherence across harmonics
    # Measure how aligned the phases of different harmonics are
    phase_coherence = np.abs(coeffs[1] * np.conj(coeffs[2]) * coeffs[0]) / (
            np.abs(coeffs[0]) * np.abs(coeffs[1]) * np.abs(coeffs[2]) + 1e-10)
    results['phase_coherence'] = np.round(phase_coherence, 5)

    return results


def get_onset_timing_relative_to_downbeat(sequences):
    """
    Calculate how events are distributed relative to the downbeat (position 0).
    This captures rhythmic "feel" - backbeat vs. on-beat emphasis.
    """

    batch_size, n_steps = sequences.shape
    results = {}

    # Define beat positions (assuming 4/4 time with 32nd note resolution)
    strong_beats = [0, 8, 16, 24]  # 1, 2, 3, 4
    weak_beats = [4, 12, 20, 28]  # &, &, &, &
    offbeats = [2, 6, 10, 14, 18, 22, 26, 30]  # e, a, e, a, etc.

    # Calculate emphasis on different beat types
    strong_emphasis = np.sum(sequences[:, strong_beats], axis=1) / len(strong_beats)
    weak_emphasis = np.sum(sequences[:, weak_beats], axis=1) / len(weak_beats)
    offbeat_emphasis = np.sum(sequences[:, offbeats], axis=1) / len(offbeats)

    results['strong_beat_emphasis'] = np.round(strong_emphasis, 5)
    results['weak_beat_emphasis'] = np.round(weak_emphasis, 5)
    results['offbeat_emphasis'] = np.round(offbeat_emphasis, 5)

    # Backbeat ratio (emphasis on beats 2 and 4 vs 1 and 3)
    beat_1_3 = np.sum(sequences[:, [0, 16]], axis=1)
    beat_2_4 = np.sum(sequences[:, [8, 24]], axis=1)

    # Avoid division by zero - set total to 1 when it's 0
    total_strong = beat_1_3 + beat_2_4
    total_strong = np.where(total_strong == 0, 1, total_strong)
    backbeat_ratio = beat_2_4 / total_strong
    results['backbeat_ratio'] = np.round(backbeat_ratio, 5)

    return results


# ================== COMPREHENSIVE FEATURE EXTRACTION ==================

def get_all_rhythm_features(sequences):
    """
    Calculate all rhythm features for sequences.

    Args:
        sequences: numpy array of shape [batch, 32] with binary hits

    Returns:
        dict with all rhythm features
    """

    features = {}

    # Basic metrics
    features['syncopation'] = get_syncopation_hvo(sequences)
    features['weak_to_strong'] = get_weak_to_strong_ratio(sequences)
    features['weak_to_all'] = get_weak_to_all_ratio(sequences)

    # Balance, evenness, entropy
    features['balance'] = get_balance(sequences)
    features['evenness'] = get_evenness(sequences)
    features['entropy'] = get_normalized_interonset_interval_entropy(sequences)

    # Phase-sensitive features
    phase_metrics = get_phase_metrics(sequences)
    timing_metrics = get_onset_timing_relative_to_downbeat(sequences)

    # Combine phase and timing features
    features.update(phase_metrics)
    features.update(timing_metrics)

    return features


# ================== COMPREHENSIVE TEST CASES ==================

def create_test_patterns():
    """
    Create a comprehensive set of test patterns covering different rhythmic characteristics.
    """

    patterns = {}

    # Basic patterns
    patterns['four_on_floor'] = np.zeros(32)
    patterns['four_on_floor'][[0, 8, 16, 24]] = 1  # Basic 4/4 kick pattern

    patterns['backbeat'] = np.zeros(32)
    patterns['backbeat'][[8, 24]] = 1  # Classic snare on 2 and 4

    patterns['offbeat'] = np.zeros(32)
    patterns['offbeat'][[4, 12, 20, 28]] = 1  # Off-beat emphasis

    # Syncopated patterns
    patterns['syncopated_1'] = np.zeros(32)
    patterns['syncopated_1'][[0, 6, 12, 18, 24, 30]] = 1  # Light syncopation

    patterns['heavy_syncopation'] = np.zeros(32)
    patterns['heavy_syncopation'][[1, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29]] = 1

    # Even spacing patterns
    patterns['even_4'] = np.zeros(32)
    patterns['even_4'][[0, 8, 16, 24]] = 1  # Every 8 steps

    patterns['even_8'] = np.zeros(32)
    patterns['even_8'][[0, 4, 8, 12, 16, 20, 24, 28]] = 1  # Every 4 steps

    patterns['even_16'] = np.zeros(32)
    patterns['even_16'][::2] = 1  # Every 2 steps

    # Uneven/clustered patterns
    patterns['clustered_start'] = np.zeros(32)
    patterns['clustered_start'][[0, 1, 2, 3, 4]] = 1  # All at beginning

    patterns['clustered_middle'] = np.zeros(32)
    patterns['clustered_middle'][[14, 15, 16, 17, 18]] = 1  # Clustered in middle

    # Complex polyrhythmic patterns
    patterns['polyrhythm_3_4'] = np.zeros(32)
    # 3 against 4: 3-note pattern over 4 beats
    patterns['polyrhythm_3_4'][[0, 10, 21]] = 1  # Approximate 3-against-4
    patterns['polyrhythm_3_4'][[0, 8, 16, 24]] = 1  # Add the 4-pattern

    # Latin/Afro-Cuban patterns
    patterns['clave_son'] = np.zeros(32)
    patterns['clave_son'][[0, 6, 16, 20, 24]] = 1  # Son clave approximation

    patterns['clave_rumba'] = np.zeros(32)
    patterns['clave_rumba'][[0, 6, 14, 20, 24]] = 1  # Rumba clave approximation

    patterns['bossa_nova'] = np.zeros(32)
    patterns['bossa_nova'][[0, 3, 6, 12, 16, 20, 24, 28]] = 1  # Bossa nova feel

    # Electronic/Dance patterns
    patterns['techno_kick'] = np.zeros(32)
    patterns['techno_kick'][[0, 8, 16, 24]] = 1  # Four-on-the-floor

    patterns['breakbeat'] = np.zeros(32)
    patterns['breakbeat'][[0, 6, 8, 14, 16, 22, 24, 30]] = 1  # Amen break style

    patterns['garage_skip'] = np.zeros(32)
    patterns['garage_skip'][[0, 12, 16, 28]] = 1  # UK garage skip pattern

    # Jazz patterns
    patterns['jazz_swing'] = np.zeros(32)
    patterns['jazz_swing'][[0, 5, 8, 13, 16, 21, 24, 29]] = 1  # Swing eighths

    patterns['jazz_latin'] = np.zeros(32)
    patterns['jazz_latin'][[0, 6, 10, 16, 22, 26]] = 1  # Latin jazz pattern

    # Minimalist patterns
    patterns['single_hit'] = np.zeros(32)
    patterns['single_hit'][0] = 1  # Just downbeat

    patterns['two_hits'] = np.zeros(32)
    patterns['two_hits'][[0, 16]] = 1  # Beats 1 and 3

    patterns['sparse_random'] = np.zeros(32)
    patterns['sparse_random'][[2, 11, 19, 27]] = 1  # Sparse, irregular

    # Dense patterns
    patterns['dense_regular'] = np.zeros(32)
    patterns['dense_regular'][::2] = 1  # Every other step

    patterns['dense_irregular'] = np.zeros(32)
    np.random.seed(42)  # For reproducibility
    positions = np.random.choice(32, 16, replace=False)
    patterns['dense_irregular'][positions] = 1

    # Convert to batch format
    pattern_names = list(patterns.keys())
    pattern_array = np.array([patterns[name] for name in pattern_names])

    return pattern_array, pattern_names


def run_comprehensive_analysis():
    """
    Run comprehensive analysis on all test patterns.
    """

    # Create test patterns
    patterns, pattern_names = create_test_patterns()
    print(f"Analyzing {len(pattern_names)} different rhythm patterns...\n")

    # Calculate all features
    features = get_all_rhythm_features(patterns)

    # Display results in a formatted table
    print("=" * 140)
    print(
        f"{'Pattern Name':<20} {'Sync':<6} {'W/S':<6} {'W/A':<6} {'Bal':<6} {'Even':<6} {'Ent':<6} {'Ph1':<7} {'PhDisp':<7} {'BB':<6} {'Strong':<6} {'Weak':<6}")
    print("=" * 140)

    for i, name in enumerate(pattern_names):
        sync = features['syncopation'][i]
        ws = features['weak_to_strong'][i]
        wa = features['weak_to_all'][i]
        bal = features['balance'][i]
        even = features['evenness'][i]
        ent = features['entropy'][i]
        ph1 = features['phase_1'][i]
        ph_disp = features['phase_dispersion'][i]
        bb = features['backbeat_ratio'][i]
        strong = features['strong_beat_emphasis'][i]
        weak = features['weak_beat_emphasis'][i]

        print(
            f"{name:<20} {sync:<6.3f} {ws:<6.3f} {wa:<6.3f} {bal:<6.3f} {even:<6.3f} {ent:<6.3f} {ph1:<7.3f} {ph_disp:<7.3f} {bb:<6.3f} {strong:<6.3f} {weak:<6.3f}")

    print("=" * 140)
    print("\nColumn Legend:")
    print("Sync = Syncopation, W/S = Weak/Strong, W/A = Weak/All, Bal = Balance")
    print("Even = Evenness, Ent = Entropy, Ph1 = Phase_1, PhDisp = Phase Dispersion")
    print("BB = Backbeat Ratio, Strong = Strong Beat Emphasis, Weak = Weak Beat Emphasis")

    return features, patterns, pattern_names


def analyze_specific_patterns(pattern_indices, patterns, pattern_names, features):
    """
    Detailed analysis of specific patterns.
    """

    print(f"\nDetailed Analysis of Selected Patterns:")
    print("=" * 80)

    for idx in pattern_indices:
        name = pattern_names[idx]
        pattern = patterns[idx]

        print(f"\nPattern: {name}")
        print(f"Hits: {np.where(pattern > 0)[0].tolist()}")

        print(f"  Syncopation: {features['syncopation'][idx]:.5f}")
        print(f"  Balance: {features['balance'][idx]:.5f}")
        print(f"  Evenness: {features['evenness'][idx]:.5f}")
        print(f"  Entropy: {features['entropy'][idx]:.5f}")
        print(f"  Phase 1: {features['phase_1'][idx]:.5f}")
        print(f"  Backbeat Ratio: {features['backbeat_ratio'][idx]:.5f}")
        print(f"  Strong Beat Emphasis: {features['strong_beat_emphasis'][idx]:.5f}")
        print(f"  Phase Dispersion: {features['phase_dispersion'][idx]:.5f}")

# ================== Joint Output Stream Descriptors =================

def rhythm_density_sync_score(seq1, seq2, seq3, sync_weight=0.3):
    """
    Single metric for rhythmic activity and synchronization

    Args:
        seq1, seq2, seq3: Binary sequences [batch, 32]
        sync_weight: How much to weight synchrony vs density (0.5-0.7 works well)

    Returns:
        Single value [0,1] - higher = more active and synchronized
    """
    # Density: overall rhythmic activity (0-1)
    total_onsets = seq1.sum(-1) + seq2.sum(-1) + seq3.sum(-1)
    density = total_onsets / 96

    # Synchrony: how often sequences hit together vs. separately
    single_hits = np.bitwise_xor(seq1, seq2, seq3).sum(-1)  # All 3 hit together
    any_hits = (seq1 | seq2 | seq3).sum(-1)  # Any sequence hits
    any_hits[any_hits == 0] = 1  # Avoid division by zero
    synchrony = single_hits / (any_hits)

    # Combined score
    return sync_weight * synchrony + (1 - sync_weight) * density

def intra_stream_exclusiveness(seq1, seq2, seq3):
    # rate of steps with single hits across all hit steps
    single_hits = np.bitwise_xor(seq1, seq2, seq3).sum(-1)  # All 3 hit together
    total_hits = (seq1 | seq2 | seq3).sum(-1)  # Any sequence hits
    total_hits[total_hits == 0] = 1  # Avoid division by zero
    exclusiveness = single_hits / total_hits
    return exclusiveness

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Run comprehensive analysis
    features, patterns, pattern_names = run_comprehensive_analysis()

    # Analyze some interesting specific patterns
    interesting_patterns = [0, 1, 4, 8, 12, 16]  # Indices of patterns to analyze in detail
    analyze_specific_patterns(interesting_patterns, patterns, pattern_names, features)

    print(f"\n\nAnalysis complete! All {len(pattern_names)} patterns processed with comprehensive feature extraction.")