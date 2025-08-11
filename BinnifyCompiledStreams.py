import numpy as np
import pandas as pd
import math
from typing import Iterable, Any, Dict, Tuple, List, Optional, Literal
from CompileStreamsAndFeatures import *


def bin_y_within_percentiles(
        x: Iterable[Any],
        y: Iterable[float],
        n_bins: int = 20,
        low_pct: float = 10.0,
        high_pct: float = 90.0,
) -> Tuple[List[str or int], Dict[Any, Dict[str, Tuple[float, float]]]]:
    """
    Bin y values within percentile bounds for each unique x value.

    Args:
        x: Iterable of x values (any type)
        y: Iterable of corresponding y values (float)
        n_bins: Number of bins to create between percentile bounds
        low_pct: Lower percentile bound
        high_pct: Upper percentile bound

    Returns:
        Tuple containing:
        - List of binned y values as strings (bin labels) or -1 for NaN values
        - Dictionary mapping x values to bin definitions
    """
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


# ========= 2) Reusable mapper for new (x, y) =========
def map_with_edges_dict(
    edges_dict: Dict[Any, Dict[str, Tuple[float, float]]],
    new_x: Iterable[Any],
    new_y: Iterable[float],
    on_unknown_x: Literal["error", "skip"] = "error",
) -> List[Optional[str]]:
    """
    Map new (x, y) to bin labels using an edges_dict of the form:
        {x: {"bin_0": (low, high), ..., "bin_{n-1}": (low, high)}}

    Clipping rule: y < lowest -> bin_0 ; y > highest -> bin_{n-1}.
    """
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


# ========= 3) Heatmap that adapts to any n_bins =========
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, PanTool, WheelZoomTool

output_notebook()  # if you're in Jupyter

def plot_bin_heatmap(
    x_values,
    bin_labels,
    edges_dict: Optional[Dict[Any, Dict[str, Tuple[float, float]]]] = None,
    title="X vs Bin Heatmap",
):
    """
    Heatmap of bin counts per X with:
      - Only nonzero tiles drawn
      - Only the single max-count cell shows text
      - `bin_0` at the bottom
      - Pan/wheel zoom/box zoom/reset/hover
    """
    df = pd.DataFrame({"x": x_values, "bin": bin_labels})

    # Infer n_bins from labels (robust if some bins unused)
    try:
        max_idx = df["bin"].str.extract(r"bin_(\d+)").astype(float).max().iloc[0]
        n_bins = int(max_idx) + 1 if pd.notna(max_idx) else 1
    except Exception:
        n_bins = 1
    bins = [f"bin_{i}" for i in range(n_bins)]

    # Numeric sort for x (0 at left)
    def sort_key(v):
        try:
            f = float(v)
            return (0, f) if math.isfinite(f) else (1, str(v))
        except Exception:
            return (1, str(v))
    xs_unique = sorted(df["x"].unique(), key=sort_key)

    # Count and keep only >0 tiles
    counts = df.groupby(["x", "bin"]).size().reset_index(name="count")
    counts = counts[counts["count"] > 0].copy()

    if counts.empty:
        p = figure(
            x_range=[str(v) for v in xs_unique],
            y_range=bins,  # bin_0 at bottom
            title=title,
            width=980, height=560,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.xaxis.axis_label = "X"
        p.yaxis.axis_label = "Bin"
        show(p)
        return

    # Single max cell gets text
    max_count = counts["count"].max()
    counts["tile_text"] = ""
    first_max_idx = counts.index[counts["count"] == max_count][0]
    counts.loc[first_max_idx, "tile_text"] = str(int(max_count))

    # Edges in hover (if provided)
    lows, highs, texts = [], [], []
    for vx, vbin in zip(counts["x"], counts["bin"]):
        if edges_dict is not None and vx in edges_dict and vbin in edges_dict[vx]:
            lo, hi = edges_dict[vx][vbin]
            lows.append(lo); highs.append(hi)
            texts.append(f"[{lo:.3g}, {hi:.3g}]" if not math.isclose(lo, hi) else f"{lo:.3g}")
        else:
            lows.append(float("nan")); highs.append(float("nan")); texts.append("")
    counts["edge_low"] = lows
    counts["edge_high"] = highs
    counts["edge_text"] = texts

    source = ColumnDataSource({
        "x": counts["x"].astype(str),
        "bin": counts["bin"],
        "count": counts["count"],
        "tile_text": counts["tile_text"],
        "edge_low": counts["edge_low"],
        "edge_high": counts["edge_high"],
        "edge_text": counts["edge_text"],
    })

    x_factors = [str(v) for v in xs_unique]
    y_factors = bins  # <- bin_0 at bottom

    mapper = LinearColorMapper(palette="Viridis256", low=0, high=int(max_count))

    tools = "pan,wheel_zoom,box_zoom,reset,save,hover"
    p = figure(
        x_range=x_factors,
        y_range=y_factors,
        x_axis_location="below",
        width=980,
        height=560,
        title=title,
        tools=tools,
        tooltips=[
            ("X", "@x"),
            ("Bin", "@bin"),
            ("Count", "@count"),
            ("Edges", "@edge_text"),
            ("Low", "@edge_low{0.###}"),
            ("High", "@edge_high{0.###}"),
        ],
        output_backend="webgl",
    )
    p.toolbar.active_drag = p.select_one(PanTool)
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    p.rect(
        x="x", y="bin", width=1, height=1,
        source=source,
        fill_color={"field": "count", "transform": mapper},
        line_color=None,
    )

    p.text(
        x="x", y="bin", text="tile_text",
        source=source,
        text_align="center",
        text_baseline="middle",
        text_alpha=0.95,
    )

    p.add_layout(ColorBar(color_mapper=mapper, location=(0, 0)), "right")
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Bin"

    show(p)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Compile datasets for triple streams.")
    parser.add_argument("--accent_v_thresh", type=float, default=0.75, help="Accent velocity threshold for accent hits.")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for the heatmap.")
    parser.add_argument("--low_percentile", type=float, default=0.0, help="Low percentile for the heatmap.")
    parser.add_argument("--high_percentile", type=float, default=0.9, help="High percentile for the heatmap.")
    args = parser.parse_args()

    # -------------------------
    # Start Binning based on overal dataset
    # -------------------------
    print("STARTING THE BINNING PROCESS")

    root_path = f"data/triple_streams/cached/AccentAt{args.accent_v_thresh}"
    os.makedirs(root_path, exist_ok=True)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"The specified root path does not exist: {root_path} \n Make sure you have already ran >> python CompileStreamsAndFeatures.py --accent_v_thresh {args.accent_v_thresh} ")


    data_all = {}
    for pkl in tqdm.tqdm(os.listdir(root_path)):
        if pkl.endswith(".pkl.bz2"):
            data = load_compiled_dataset_pkl_bz2(os.path.join(root_path, pkl))
            for k, v in data.items():
                if k not in data_all:
                    data_all[k] = []
                data_all[k].extend(v)

    x = data_all["Flat Out Vs. Input | Hits | Hamming"]
    y = data_all["Flat Out Vs. Input | Accent | Hamming"]
    y_normalized = data_all["Flat Out Vs. Input | Accent | Hamming Normalized"]

    _, edges_dict = bin_y_within_percentiles(x, y, n_bins=args.n_bins, low_pct=args.low_percentile,
                                             high_pct=args.high_percentile)
    _, edges_dict_normalized = bin_y_within_percentiles(x, y_normalized, n_bins=args.n_bins, low_pct=args.low_percentile,
                                                        high_pct=args.high_percentile)

    # Save the edges_dict to a file
    new_save_folder = root_path + "_with_" + str(args.n_bins) + "_bins_in_" + str(args.low_percentile) + "_" + str(args.high_percentile) + "_percentile"
    os.makedirs(new_save_folder, exist_ok=True)

    edges_dict_path = os.path.join(new_save_folder,
                                   f"edges_dict_{args.n_bins}_bins_{args.low_percentile}_{args.high_percentile}.pkl.bz2")
    with bz2.BZ2File(edges_dict_path, 'wb') as f:
        pickle.dump(edges_dict, f)

    print(f"Edges dict saved at: {edges_dict_path}")

    edges_dict_normalized_path = os.path.join(new_save_folder,
                                              f"edges_dict_normalized_{args.n_bins}_bins_{args.low_percentile}_{args.high_percentile}.pkl.bz2")
    with bz2.BZ2File(edges_dict_normalized_path, 'wb') as f:
        pickle.dump(edges_dict_normalized, f)

    print(f"Edges normalized dict saved at: {edges_dict_normalized_path}")

    del data_all

    # -------------------------
    # Open all pkls again and binnify "Flat Out Vs. Input | Accent | Hamming", "Flat Out Vs. Input | Accent | Hamming Normalized"
    # using map_with_edges_dict(edges_dict, x, y, on_unknown_x="error")
    print("STARTING THE MAPPING PROCESS")
    os.makedirs(new_save_folder, exist_ok=True)

    for pkl in tqdm.tqdm(os.listdir(root_path)):
        if pkl.endswith(".pkl.bz2"):
            data = load_compiled_dataset_pkl_bz2(os.path.join(root_path, pkl))
            y = data["Flat Out Vs. Input | Accent | Hamming"]
            y_normalized = data["Flat Out Vs. Input | Accent | Hamming Normalized"]
            x = data["Flat Out Vs. Input | Hits | Hamming"]

            # Map the y values to bins using the edges_dict
            mapped_y = map_with_edges_dict(edges_dict, x, y, on_unknown_x="error")
            mapped_y_normalized = map_with_edges_dict(edges_dict_normalized, x, y_normalized, on_unknown_x="error")

            # Update the data dictionary with the mapped values
            data["Flat Out Vs. Input | Accent | Hamming (Binned)"] = mapped_y
            data["Flat Out Vs. Input | Accent | Hamming Normalized (Binned)"] = mapped_y_normalized

            # Save the updated data in a new directory appending with binning settings
            compiled_data_path = os.path.join(new_save_folder, pkl)
            with bz2.BZ2File(compiled_data_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Updated data saved at: {compiled_data_path}")

    print("Binning process completed for all datasets.")
    print("Accessing the binned data:")

    print("All datasets have been compiled and processed successfully.")

