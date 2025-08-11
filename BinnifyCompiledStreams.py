import numpy as np
from typing import Any, Iterable, List, Dict, Tuple, Optional, Literal, Union
from CompileStreamsAndFeatures import load_compiled_dataset_pkl_bz2

def bin_y_within_percentiles(
        x: Optional[Iterable[Any]] = None,
        y: Optional[Iterable[float]] = None,
        data: Optional[Iterable[float]] = None,
        n_bins: int = 20,
        low_pct: float = 10.0,
        high_pct: float = 90.0,
) -> Union[Tuple[List[str], Dict[Any, Dict[str, Tuple[float, float]]]],
Tuple[List[str], Dict[str, Tuple[float, float]]]]:
    """
    Bin y values within percentile bounds for each unique x value (2D case),
    or bin data values within percentile bounds (1D case).

    Args:
        x: Iterable of x values (any type). If None, treats as 1D case.
        y: Iterable of corresponding y values (float). Used in 2D case.
        data: Iterable of values to bin (float). Used in 1D case when x is None.
        n_bins: Number of bins to create between percentile bounds
        low_pct: Lower percentile bound
        high_pct: Upper percentile bound

    Returns:
        For 2D case (x and y provided):
        - Tuple containing:
            - List of binned y values as strings (bin labels) or -1 for NaN values
            - Dictionary mapping x values to bin definitions

        For 1D case (only data provided):
        - Tuple containing:
            - List of binned data values as strings (bin labels) or -1 for NaN values
            - Dictionary with bin definitions (single set of bins)
    """
    # Determine if this is 1D or 2D case
    if x is None:
        if data is None:
            raise ValueError("Either provide (x, y) for 2D binning or (data) for 1D binning")
        return _bin_1d(data, n_bins, low_pct, high_pct)
    else:
        if y is None:
            raise ValueError("If x is provided, y must also be provided")
        return _bin_2d(x, y, n_bins, low_pct, high_pct)


def _bin_1d(
        data: Iterable[float],
        n_bins: int,
        low_pct: float,
        high_pct: float,
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """Handle 1D binning case."""
    data_list = list(data)

    # Filter out NaN values for binning calculations
    valid_data = [d for d in data_list if not (np.isnan(d) if isinstance(d, (int, float, np.number)) else False)]

    if len(valid_data) == 0:
        # No valid data - return all -1s and empty bin definitions
        return [-1] * len(data_list), {}

    # Calculate percentile bounds
    data_array = np.array(valid_data)
    low_bound = np.percentile(data_array, low_pct)
    high_bound = np.percentile(data_array, high_pct)

    # Create bin edges
    bin_edges = np.linspace(low_bound, high_bound, n_bins + 1)

    # Create bin definitions
    bin_definitions = {}
    for i in range(n_bins):
        bin_name = f"bin_{i}"
        bin_definitions[bin_name] = (bin_edges[i], bin_edges[i + 1])

    # Bin each data value
    binned_data = []
    for d in data_list:
        # Check if data is NaN
        if np.isnan(d) if isinstance(d, (int, float, np.number)) else False:
            binned_data.append(-1)
            continue

        # Determine which bin this value falls into
        if d < low_bound:
            bin_idx = 0  # Below range - assign to first bin
        elif d >= high_bound:
            bin_idx = n_bins - 1  # Above range - assign to last bin
        else:
            # Find appropriate bin
            bin_idx = np.digitize(d, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))  # Clamp to valid range

        binned_data.append(f"bin_{bin_idx}")

    return binned_data, bin_definitions


def _bin_2d(
        x: Iterable[Any],
        y: Iterable[float],
        n_bins: int,
        low_pct: float,
        high_pct: float,
) -> Tuple[List[str], Dict[Any, Dict[str, Tuple[float, float]]]]:
    """Handle 2D binning case (original functionality)."""
    # Convert to lists for easier handling
    x_list = list(x)
    y_list = list(y)

    # Group y values by unique x values, excluding NaN values for binning calculations
    x_to_y = {}
    for xi, yi in zip(x_list, y_list):
        if xi not in x_to_y:
            x_to_y[xi] = []
        # Only add non-NaN values for binning calculations
        if not (np.isnan(yi) if isinstance(yi, (int, float, np.number)) else False):
            x_to_y[xi].append(yi)

    # Result containers
    binned_y = []
    bin_definitions = {}

    # First, create bin definitions for each unique x (using only non-NaN values)
    for xi in set(x_list):
        # Get all non-NaN y values for this x
        y_vals = np.array(x_to_y[xi])

        # Skip if no valid y values for this x
        if len(y_vals) == 0:
            bin_definitions[xi] = {}
            continue

        # Calculate percentile bounds
        low_bound = np.percentile(y_vals, low_pct)
        high_bound = np.percentile(y_vals, high_pct)

        # Create bin edges
        bin_edges = np.linspace(low_bound, high_bound, n_bins + 1)

        # Create bin definitions dictionary for this x
        bin_defs = {}
        for i in range(n_bins):
            bin_name = f"bin_{i}"
            bin_defs[bin_name] = (bin_edges[i], bin_edges[i + 1])

        bin_definitions[xi] = bin_defs

    # Now bin each y value based on its corresponding x
    for i, (xi, yi) in enumerate(zip(x_list, y_list)):
        # Check if y is NaN
        if np.isnan(yi) if isinstance(yi, (int, float, np.number)) else False:
            binned_y.append(-1)
            continue

        # Check if this x has any valid y values for binning
        if len(x_to_y[xi]) == 0:
            binned_y.append(-1)
            continue

        # Get bin edges for this x
        y_vals = np.array(x_to_y[xi])
        low_bound = np.percentile(y_vals, low_pct)
        high_bound = np.percentile(y_vals, high_pct)
        bin_edges = np.linspace(low_bound, high_bound, n_bins + 1)

        # Determine which bin this y value falls into
        if yi < low_bound:
            bin_idx = 0  # Below range - assign to first bin
        elif yi >= high_bound:
            bin_idx = n_bins - 1  # Above range - assign to last bin
        else:
            # Find appropriate bin
            bin_idx = np.digitize(yi, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))  # Clamp to valid range

        binned_y.append(f"bin_{bin_idx}")

    return binned_y, bin_definitions


def map_with_edges_dict(
        edges_dict: Union[Dict[Any, Dict[str, Tuple[float, float]]], Dict[str, Tuple[float, float]]],
        new_x: Optional[Iterable[Any]] = None,
        new_y: Optional[Iterable[float]] = None,
        new_data: Optional[Iterable[float]] = None,
        on_unknown_x: Literal["error", "skip"] = "error",
) -> List[Optional[str]]:
    """
    Map new (x, y) to bin labels using an edges_dict (2D case),
    or map new data to bin labels using bin definitions (1D case).

    Args:
        edges_dict:
            For 2D case: {x: {"bin_0": (low, high), ..., "bin_{n-1}": (low, high)}}
            For 1D case: {"bin_0": (low, high), ..., "bin_{n-1}": (low, high)}
        new_x: New x values (for 2D case)
        new_y: New y values (for 2D case)
        new_data: New data values (for 1D case)
        on_unknown_x: How to handle unknown x values ("error" or "skip")

    Returns:
        List of bin labels (strings) or None for invalid/unknown values

    Clipping rule: y < lowest -> bin_0 ; y > highest -> bin_{n-1}.
    """
    # Determine if this is 1D or 2D case based on edges_dict structure
    if not edges_dict:
        return []

    # Check if edges_dict has the 2D structure (nested dict) or 1D structure (flat dict)
    first_key = next(iter(edges_dict))
    is_2d_case = isinstance(edges_dict[first_key], dict)

    if is_2d_case:
        if new_x is None or new_y is None:
            raise ValueError("For 2D edges_dict, both new_x and new_y must be provided")
        return _map_2d(edges_dict, new_x, new_y, on_unknown_x)
    else:
        if new_data is None:
            raise ValueError("For 1D edges_dict, new_data must be provided")
        return _map_1d(edges_dict, new_data)


def _map_1d(
        edges_dict: Dict[str, Tuple[float, float]],
        new_data: Iterable[float],
) -> List[Optional[str]]:
    """Handle 1D mapping case."""
    data_list = list(map(float, new_data))

    if not edges_dict:
        return [None] * len(data_list)

    # Infer n_bins from the dict keys
    k_indices = sorted(int(k.split("_")[1]) for k in edges_dict.keys())
    n_bins = (k_indices[-1] + 1) if k_indices else 1

    # Rebuild edges array of length n_bins+1
    lows = [edges_dict[f"bin_{k}"][0] for k in range(n_bins)]
    highs = [edges_dict[f"bin_{k}"][1] for k in range(n_bins)]
    edges = np.array([lows[0]] + highs, dtype=float)

    out = []
    for d in data_list:
        if np.allclose(edges[0], edges[-1]):  # degenerate -> bin_0
            out.append("bin_0")
            continue

        k = int(np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1))
        out.append(f"bin_{k}")

    return out


def _map_2d(
        edges_dict: Dict[Any, Dict[str, Tuple[float, float]]],
        new_x: Iterable[Any],
        new_y: Iterable[float],
        on_unknown_x: Literal["error", "skip"],
) -> List[Optional[str]]:
    """Handle 2D mapping case (original functionality)."""
    nx = list(new_x)
    ny = list(map(float, new_y))
    if len(nx) != len(ny):
        raise ValueError("new_x and new_y must have the same length.")

    out: List[Optional[str]] = [None] * len(nx)

    for i, (vx, vy) in enumerate(zip(nx, ny)):
        x_bins = edges_dict.get(vx)
        if x_bins is None:
            if on_unknown_x == "error":
                raise KeyError(f"x={vx!r} not present in edges_dict.")
            out[i] = None
            continue

        # Infer n_bins from the dict keys
        # Assumes names "bin_0"... consistently
        k_indices = sorted(int(k.split("_")[1]) for k in x_bins.keys())
        n_bins = (k_indices[-1] + 1) if k_indices else 1

        # Rebuild edges array of length n_bins+1
        lows = [x_bins[f"bin_{k}"][0] for k in range(n_bins)]
        highs = [x_bins[f"bin_{k}"][1] for k in range(n_bins)]
        edges = np.array([lows[0]] + highs, dtype=float)

        if np.allclose(edges[0], edges[-1]):  # degenerate -> bin_0
            out[i] = "bin_0"
            continue

        k = int(np.clip(np.searchsorted(edges, vy, side="right") - 1, 0, n_bins - 1))
        out[i] = f"bin_{k}"

    return out


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


# Example usage:
if __name__ == "__main__":
    n_encoding_control1_tokens = 33                                         # 33 unique values 0., 0.03125, 0.0625, ..., 1.
    encoding_control1_key = "Flat Out Vs. Input | Hits | Hamming"
    n_encoding_control2_tokens = 10
    encoding_control2_key = "Flat Out Vs. Input | Accent | Hamming"
    n_decoding_control1_tokens = 12
    decoding_control1_key = "Stream 1 Vs. Flat Out | Hits | Hamming"
    n_decoding_control2_tokens = 12
    decoding_control2_key = "Stream 2 Vs. Flat Out | Hits | Hamming"
    n_decoding_control3_tokens = 12
    decoding_control3_key = "Stream 3 Vs. Flat Out | Hits | Hamming"
    
    # dataset = load_compiled_dataset_pkl_bz2("data/triple_streams/cached/TestAccentAt0.75/training_splits/train.pkl.bz2")

    control_ = np.round(dataset[encoding_control1_key][::6], 5)
    hit_hamming_tokens = TokenizeControls(control_array=hit_hamming, n_bins=n_encoding_control1_tokens, low=0, high=1)


    hit_hamming[:-10], hit_hamming_tokens[:-10]

    accent_hamming = np.round(dataset[encoding_control2_key][::6], 5)
    accent_hamming_tokens = TokenizeControls(control_array=accent_hamming, n_bins=n_encoding_control2_tokens, low=0, high=0.85)

    stream1_hamming = np.round(dataset[decoding_control1_key][::6], 5)
    stream1_hamming_tokens = TokenizeControls(control_array=stream1_hamming, n_bins=n_decoding_control1_tokens, low=0, high=0.85)
    stream2_hamming = np.round(dataset[decoding_control2_key][::6], 5)
    stream2_hamming_tokens = TokenizeControls(control_array=stream2_hamming, n_bins=n_decoding_control2_tokens, low=0, high=0.85)
    stream3_hamming = np.round(dataset[decoding_control3_key][::6], 5)
    stream3_hamming_tokens = TokenizeControls(control_array=stream3_hamming, n_bins=n_decoding_control3_tokens, low=0, high=0.85)

    np.unique(stream1_hamming, return_counts=True)

