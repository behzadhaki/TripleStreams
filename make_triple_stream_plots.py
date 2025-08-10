#!/usr/bin/env python3
"""
Sparse, bin-aligned heatmap + jittered scatter

- Scatter: larger, low-opacity, jittered x&y, NO hover
- Heatmap: quads ONLY for count>0; ticks + grid at actual bin centers in use
- Hover (heatmap only): feat1, feat2, count
- Only the single highest bin is labeled
- Fixed-bin mode or 0.01 rounding unique-pair mode
- Caches per dataset+mode in root/cache
- tqdm progress; notebook/CLI friendly
"""
import os
import bz2
import pickle
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm.auto import tqdm

from bokeh.io import output_file, save
from bokeh.models import (
    ColumnDataSource, Select, CustomJS,
    LinearColorMapper, ColorBar, BasicTicker, FixedTicker,
    Tabs, Panel, LabelSet, HoverTool
)
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.palettes import Viridis256


# -----------------
# Loader
# -----------------
def try_import_loader():
    candidates = [
        ("tripleStreamsDataloaderUtils", "load_compiled_dataset_pkl_bz2"),
        ("triple_stream_data_utils", "load_compiled_dataset_pkl_bz2"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            loader = getattr(mod, fn_name, None)
            if loader is not None:
                return loader
        except Exception:
            continue

    def _fallback_loader(path: str):
        with bz2.open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    return _fallback_loader


load_compiled_dataset_pkl_bz2 = try_import_loader()


# -----------------
# Utils
# -----------------
def discover_available_datasets(root_dir: str) -> Dict[str, str]:
    files = [f for f in os.listdir(root_dir) if f.endswith(".pkl.bz2")]
    return {f.split(".pkl.bz2")[0]: os.path.join(root_dir, f) for f in files}


def ensure_cache_dir(root_dir: str) -> str:
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_key(cache_dir: str, dataset_tag: str, feat1: str, feat2: str,
              n_bins_feat1: Optional[int], n_bins_feat2: Optional[int]) -> str:
    safe = lambda s: s.replace(os.sep, "_")
    mode = "round2" if (n_bins_feat1 is None or n_bins_feat2 is None) \
           else f"bins{int(n_bins_feat1)}x{int(n_bins_feat2)}"
    return os.path.join(cache_dir, f"hist2d__{safe(dataset_tag)}__{safe(feat1)}__{safe(feat2)}__{mode}.npz")


def is_cache_valid(cache_path: str, data_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(data_path)
    except Exception:
        return False


def build_round_edges_from_uniques(values: np.ndarray, step: float = 0.01) -> np.ndarray:
    """Edges from ONLY unique 0.01-rounded centers in the data (no empty rows/cols)."""
    centers = np.unique(np.round(values, 2))
    if centers.size == 0:
        return np.array([0.0, step])
    start = centers[0] - step/2.0
    edges = start + np.arange(centers.size + 1) * step
    return np.round(edges, 6)


def compute_hist2d_custom(
        feat1_vals: np.ndarray,
        feat2_vals: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_bins_feat1: Optional[int],
        n_bins_feat2: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns H (ny, nx), xedges (nx+1), yedges (ny+1).
    - Fixed bins: histogram2d with provided bins/range
    - Rounding mode: 0.01 rounding, edges from unique centers only
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    if n_bins_feat1 is not None and n_bins_feat2 is not None:
        H, xedges, yedges = np.histogram2d(
            feat1_vals, feat2_vals,
            bins=(n_bins_feat1, n_bins_feat2),
            range=[[xmin, xmax], [ymin, ymax]]
        )
        return H, xedges, yedges

    step = 0.01
    x_round = np.round(feat1_vals, 2)
    y_round = np.round(feat2_vals, 2)

    xedges = build_round_edges_from_uniques(x_round, step=step)
    yedges = build_round_edges_from_uniques(y_round, step=step)

    H, _, _ = np.histogram2d(x_round, y_round, bins=(xedges, yedges))
    return H, xedges, yedges


def downsample_pairs(x: np.ndarray, y: np.ndarray, max_points: int):
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def max_label_position(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    idx_flat = int(np.argmax(H))
    ny, nx = H.shape
    row = idx_flat // nx
    col = idx_flat % nx
    xcenters = 0.5*(xedges[:-1] + xedges[1:])
    ycenters = 0.5*(yedges[:-1] + yedges[1:])
    return xcenters[col], ycenters[row], str(int(H[row, col]))


def make_quad_source_sparse(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    """
    Return:
      - ColumnDataSource for quads (ONLY count>0), with slight epsilon growth to avoid hairline gaps
      - xcenters_kept, ycenters_kept: tick positions for columns/rows that actually exist
    """
    xleft = xedges[:-1]; xright = xedges[1:]
    ybottom = yedges[:-1]; ytop = yedges[1:]

    col_sum = H.sum(axis=0)  # nx
    row_sum = H.sum(axis=1)  # ny
    keep_cols = np.where(col_sum > 0)[0]
    keep_rows = np.where(row_sum > 0)[0]

    # tick positions ONLY where data exists
    xcenters = 0.5*(xedges[:-1] + xedges[1:])
    ycenters = 0.5*(yedges[:-1] + yedges[1:])
    x_ticks = xcenters[keep_cols].tolist()
    y_ticks = ycenters[keep_rows].tolist()

    # small epsilon to ensure flush visuals (expand a hair)
    dxs = np.diff(xedges); dys = np.diff(yedges)
    epsx = float(max(1e-12, 1e-3 * np.min(dxs))) if dxs.size else 0.0
    epsy = float(max(1e-12, 1e-3 * np.min(dys))) if dys.size else 0.0

    xs_left, xs_right, ys_bottom, ys_top = [], [], [], []
    xcenter, ycenter, counts = [], [], []

    for i in keep_rows:
        for j in keep_cols:
            c = float(H[i, j])
            if c <= 0:
                continue
            xl = xleft[j] - epsx/2; xr = xright[j] + epsx/2
            yb = ybottom[i] - epsy/2; yt = ytop[i] + epsy/2
            xs_left.append(xl); xs_right.append(xr)
            ys_bottom.append(yb); ys_top.append(yt)
            xcenter.append(0.5*(xleft[j]+xright[j])); ycenter.append(0.5*(ybottom[i]+ytop[i]))
            counts.append(c)

    src = ColumnDataSource(dict(
        left=xs_left, right=xs_right, bottom=ys_bottom, top=ys_top,
        x=xcenter, y=ycenter, count=counts
    ))
    return src, x_ticks, y_ticks


# -----------------
# Core builder (returns layout: selector over tabs)
# -----------------
def build_layout(root: str,
                 dataset_tags: List[str],
                 feature1: str,
                 feature2: str,
                 n_bins_feat1: Optional[int] = None,
                 n_bins_feat2: Optional[int] = None,
                 scatter_max_points: int = 100_000):
    available = discover_available_datasets(root)
    cache_dir = ensure_cache_dir(root)

    valid_tags = [t for t in dataset_tags if t in available]
    missing = [t for t in dataset_tags if t not in available]
    if missing:
        print(f"[warn] Missing dataset_tags (ignored): {missing}")
    if not valid_tags:
        raise RuntimeError("None of the provided dataset_tags exist under the root path.")

    # Load datasets
    dataset_feature_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    print("Loading datasets...")
    for tag in tqdm(valid_tags, desc="Datasets"):
        path = available[tag]
        d_dict = load_compiled_dataset_pkl_bz2(path)

        if feature1 not in d_dict or feature2 not in d_dict:
            raise KeyError(f"Features '{feature1}' and/or '{feature2}' not found in dataset '{tag}'.")

        x = np.asarray(d_dict[feature1], dtype=float)
        y = np.asarray(d_dict[feature2], dtype=float)
        x = np.concatenate([np.ravel(v) for v in x]) if x.dtype == object else np.ravel(x)
        y = np.concatenate([np.ravel(v) for v in y]) if y.dtype == object else np.ravel(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        dataset_feature_arrays[tag] = {feature1: x, feature2: y}

    # Global ranges (robust, for consistent axes)
    all_x = np.concatenate([dataset_feature_arrays[tag][feature1] for tag in valid_tags])
    all_y = np.concatenate([dataset_feature_arrays[tag][feature2] for tag in valid_tags])

    x_min, x_max = np.percentile(all_x, [1, 99])
    y_min, y_max = np.percentile(all_y, [1, 99])

    pad_x = 0.02 * (x_max - x_min) if x_max > x_min else 1.0
    pad_y = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
    x_range = (float(x_min - pad_x), float(x_max + pad_x))
    y_range = (float(y_min - pad_y), float(y_max + pad_y))

    # Build heatmaps (cache), sparse quads, and max labels + ticks
    heatmap_meta: Dict[str, Dict[str, Any]] = {}
    quad_sources: Dict[str, ColumnDataSource] = {}
    labels_meta: Dict[str, Dict[str, Any]] = {}
    xticks_meta: Dict[str, List[float]] = {}
    yticks_meta: Dict[str, List[float]] = {}

    print("Preparing heatmap caches...")
    for tag in tqdm(valid_tags, desc="Heatmaps"):
        data_path = available[tag]
        cache_path = cache_key(cache_dir, tag, feature1, feature2, n_bins_feat1, n_bins_feat2)
        x = dataset_feature_arrays[tag][feature1]
        y = dataset_feature_arrays[tag][feature2]

        if is_cache_valid(cache_path, data_path):
            cached = np.load(cache_path, allow_pickle=True)
            H = cached["H"]; xedges = cached["xedges"]; yedges = cached["yedges"]
        else:
            H, xedges, yedges = compute_hist2d_custom(x, y, x_range, y_range, n_bins_feat1, n_bins_feat2)
            np.savez_compressed(cache_path, H=H, xedges=xedges, yedges=yedges)

        heatmap_meta[tag] = {"H": H.astype(float), "xedges": xedges, "yedges": yedges}

        src, x_ticks, y_ticks = make_quad_source_sparse(H, xedges, yedges)
        quad_sources[tag] = src
        xticks_meta[tag] = x_ticks
        yticks_meta[tag] = y_ticks

        lx, ly, ltext = max_label_position(H, xedges, yedges)
        labels_meta[tag] = {"x": [lx], "y": [ly], "text": [ltext]}

    # Scatter (no hover) with jitter (percent of axis span)
    rng = np.random.default_rng(12345)
    jitter_x = (x_range[1] - x_range[0]) * 0.005
    jitter_y = (y_range[1] - y_range[0]) * 0.005

    scatter_sources: Dict[str, ColumnDataSource] = {}
    print("Preparing scatter data...")
    for tag in tqdm(valid_tags, desc="Scatter"):
        x = dataset_feature_arrays[tag][feature1]
        y = dataset_feature_arrays[tag][feature2]
        xs, ys = downsample_pairs(x, y, scatter_max_points)
        xs_vis = xs + rng.uniform(-jitter_x, jitter_x, size=xs.shape)
        ys_vis = ys + rng.uniform(-jitter_y, jitter_y, size=ys.shape)
        scatter_sources[tag] = ColumnDataSource(dict(x=xs_vis, y=ys_vis))

    # --- Figures ---
    TOOLS_NO_HOVER = "pan,wheel_zoom,box_zoom,reset,save"

    # Scatter
    scatter_fig = figure(
        width=750, height=640, x_range=x_range, y_range=y_range,
        tools=TOOLS_NO_HOVER, title=f"Scatter: {feature1} vs {feature2}",
        output_backend="webgl",
    )
    scatter_fig.circle("x", "y", source=scatter_sources[valid_tags[0]],
                       size=5, fill_alpha=0.06, line_alpha=0)
    scatter_fig.xaxis.axis_label = feature1
    scatter_fig.yaxis.axis_label = feature2
    scatter_fig.match_aspect = True
    scatter_fig.grid.grid_line_alpha = 0.4

    # Heatmap
    heatmap_fig = figure(
        width=750, height=640, x_range=x_range, y_range=y_range,
        tools=TOOLS_NO_HOVER, title=f"Heatmap: {feature1} vs {feature2}",
        output_backend="webgl",
    )
    heatmap_fig.grid.grid_line_alpha = 0.4

    # Color mapper
    color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)

    quad_src_initial = quad_sources[valid_tags[0]]
    quad_renderer = heatmap_fig.quad(
        left="left", right="right", bottom="bottom", top="top",
        source=quad_src_initial,
        fill_color={'field': 'count', 'transform': color_mapper},
        fill_alpha=1.0,
        line_alpha=0.0,
    )

    # Hover (HEATMAP ONLY)
    hover = HoverTool(
        tooltips=[
            ("feat1 =", "@x{0.00}"),
            ("feat2 =", "@y{0.00}"),
            ("count =", "@count{0,0}"),
        ],
        mode="mouse",
        renderers=[quad_renderer]
    )
    heatmap_fig.add_tools(hover)

    # One label only (max)
    label_source = ColumnDataSource(data=labels_meta[valid_tags[0]])
    labels = LabelSet(x='x', y='y', text='text', source=label_source,
                      text_baseline="middle", text_align="center",
                      text_font_size="10pt", text_color="white",
                      background_fill_color="black", background_fill_alpha=0.25)
    heatmap_fig.add_layout(labels)

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=8, location=(0,0))
    heatmap_fig.add_layout(color_bar, 'right')
    heatmap_fig.xaxis.axis_label = feature1
    heatmap_fig.yaxis.axis_label = feature2
    heatmap_fig.match_aspect = True

    # Tickers at actual bin centers for the INITIAL dataset
    x_ticker = FixedTicker(ticks=xticks_meta[valid_tags[0]])
    y_ticker = FixedTicker(ticks=yticks_meta[valid_tags[0]])
    heatmap_fig.xaxis[0].ticker = x_ticker
    heatmap_fig.yaxis[0].ticker = y_ticker
    # Use same tickers on scatter so grid aligns across tabs
    scatter_fig.xaxis[0].ticker = x_ticker
    scatter_fig.yaxis[0].ticker = y_ticker

    # Initialize color high
    initial_H = heatmap_meta[valid_tags[0]]["H"]
    color_mapper.high = float(np.max(initial_H)) if np.max(initial_H) > 0 else 1.0

    # Selector
    selector = Select(title="Dataset", value=valid_tags[0], options=valid_tags)

    # Payloads for JS
    payload_scatter = {tag: dict(
        x=scatter_sources[tag].data["x"].tolist(),
        y=scatter_sources[tag].data["y"].tolist()
    ) for tag in valid_tags}

    # pack quad data
    def pack_quad(src: ColumnDataSource):
        d = src.data
        out = {}
        for k in ("left","right","bottom","top","x","y","count"):
            out[k] = [float(v) for v in d[k]]
        return out
    payload_quads = {tag: pack_quad(quad_sources[tag]) for tag in valid_tags}

    payload_label = {tag: labels_meta[tag] for tag in valid_tags}
    payload_maxH  = {tag: float(np.max(heatmap_meta[tag]["H"])) for tag in valid_tags}

    # ticks payloads
    payload_xticks = {tag: xticks_meta[tag] for tag in valid_tags}
    payload_yticks = {tag: yticks_meta[tag] for tag in valid_tags}

    callback = CustomJS(args=dict(
        selector=selector,
        scatter_src=scatter_sources[valid_tags[0]],
        quad_src=quad_src_initial,
        color_mapper=color_mapper,
        label_src=label_source,
        x_ticker=x_ticker,
        y_ticker=y_ticker,
        scatter_payload=payload_scatter,
        quad_payload=payload_quads,
        label_payload=payload_label,
        maxH_payload=payload_maxH,
        xticks_payload=payload_xticks,
        yticks_payload=payload_yticks,
    ), code="""
const tag = selector.value;

// Scatter
const sp = scatter_payload[tag];
scatter_src.data = {x: sp.x, y: sp.y};
scatter_src.change.emit();

// Heatmap quads
const qp = quad_payload[tag];
quad_src.data = qp;
quad_src.change.emit();

// Label
const lp = label_payload[tag];
label_src.data = lp;
label_src.change.emit();

// Color scale
const mh = maxH_payload[tag];
color_mapper.high = (mh > 0) ? mh : 1.0;

// Ticks (actual bin centers with data)
x_ticker.ticks = xticks_payload[tag];
y_ticker.ticks = yticks_payload[tag];
x_ticker.change.emit();
y_ticker.change.emit();
""")

    selector.js_on_change("value", callback)

    # Tabs (selector above)
    scatter_tab = Panel(child=scatter_fig, title="Scatter")
    heatmap_tab = Panel(child=heatmap_fig, title="Heatmap")
    tabs = Tabs(tabs=[scatter_tab, heatmap_tab])

    return column(selector, tabs)


def build_plots(root: str,
                dataset_tags: List[str],
                feature1: str,
                feature2: str,
                n_bins_feat1: Optional[int] = None,
                n_bins_feat2: Optional[int] = None,
                scatter_max_points: int = 100_000,
                output_html_path: str = "triple_stream_plots.html") -> str:
    layout = build_layout(root, dataset_tags, feature1, feature2,
                          n_bins_feat1=n_bins_feat1, n_bins_feat2=n_bins_feat2,
                          scatter_max_points=scatter_max_points)
    output_file(output_html_path, title=f"Feature Plots: {feature1} vs {feature2}")
    save(layout)
    return os.path.abspath(output_html_path)


# -----------------
# CLI
# -----------------
def main():
    parser = argparse.ArgumentParser(description="Build cached, tabbed Bokeh plots (Scatter & Heatmap).")
    parser.add_argument("--root", required=True, help="Root folder containing .pkl.bz2 files")
    parser.add_argument("--tags", nargs="+", required=True, help="Dataset tags (filenames without .pkl.bz2)")
    parser.add_argument("--feature1", required=True, help="Feature name for X axis")
    parser.add_argument("--feature2", required=True, help="Feature name for Y axis")
    parser.add_argument("--n-bins-feat1", type=int, default=None,
                        help="Bins along feature1 (X). If omitted/None → 0.01 rounding mode.")
    parser.add_argument("--n-bins-feat2", type=int, default=None,
                        help="Bins along feature2 (Y). If omitted/None → 0.01 rounding mode.")
    parser.add_argument("--scatter-max-points", type=int, default=100000, help="Scatter downsample size")
    parser.add_argument("--output", default="triple_stream_plots.html", help="Output HTML filename")

    args = parser.parse_args()

    path = build_plots(
        root=args.root,
        dataset_tags=args.tags,
        feature1=args.feature1,
        feature2=args.feature2,
        n_bins_feat1=args.n_bins_feat1,
        n_bins_feat2=args.n_bins_feat2,
        scatter_max_points=args.scatter_max_points,
        output_html_path=args.output
    )
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
