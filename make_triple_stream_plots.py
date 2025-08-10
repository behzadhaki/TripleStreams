#!/usr/bin/env python3
"""
Sparse, bin-aligned heatmap + jittered scatter with shared coloring

- Scatter: larger, low-opacity, jittered x&y, colored by density, WITH hover
- Heatmap: quads ONLY for count>0; ticks + grid at actual bin centers in use
- Hover (both): feat1, feat2, count/density
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

def _to_pylist(x):
    try:
        return x.tolist()
    except AttributeError:
        return list(x)

from tqdm.auto import tqdm

from bokeh.io import output_file, save
from bokeh.models import (
    ColumnDataSource, Select, CustomJS,
    LinearColorMapper, ColorBar, BasicTicker, FixedTicker,
    Tabs, Panel, LabelSet, HoverTool, Range1d
)

from bokeh.models import FuncTickFormatter

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
    cache_dir = os.path.join(root_dir, "analysis/cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_key(cache_dir: str, dataset_tag: str, feat1: str, feat2: str,
              n_bins_feat1: Optional[int], n_bins_feat2: Optional[int]) -> str:
    safe = lambda s: s.replace(os.sep, "_")
    mode = "round2" if (n_bins_feat1 is None or n_bins_feat2 is None) \
        else f"bins{int(n_bins_feat1)}x{int(n_bins_feat2)}"
    return os.path.join(cache_dir, f"hist2d__{safe(dataset_tag)}__{safe(feat1)}__{safe(feat2)}__{mode}.npz")


def is_cache_valid(cache_path: str, data_path: str, force_recalc: bool = False) -> bool:
    if force_recalc:
        return False
    if not os.path.exists(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(data_path)
    except Exception:
        return False


def compute_hist2d_custom(
        feat1_vals: np.ndarray,
        feat2_vals: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_bins_feat1: Optional[int],
        n_bins_feat2: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = x_range
    ymin, ymax = y_range

    if n_bins_feat1 is not None and n_bins_feat2 is not None:
        H, xedges, yedges = np.histogram2d(
            feat1_vals, feat2_vals,
            bins=(n_bins_feat1, n_bins_feat2),
            range=[[xmin, xmax], [ymin, ymax]]
        )
    else:
        step = 1 / 32
        xedges = np.arange(0, xmax + step, step)
        yedges = np.arange(0, ymax + step, step)
        if xedges[-1] < xmax:
            xedges = np.append(xedges, xedges[-1] + step)
        if yedges[-1] < ymax:
            yedges = np.append(yedges, yedges[-1] + step)

        H, xedges, yedges = np.histogram2d(
            feat1_vals, feat2_vals,
            bins=(xedges, yedges)
        )

    # IMPORTANT: numpy returns H with shape (nx, ny). We want H[y, x].
    H = H.T
    return H, xedges, yedges


def compute_point_densities(x: np.ndarray, y: np.ndarray, H: np.ndarray,
                            xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    """Compute density value for each point based on which bin it falls into."""
    densities = np.zeros(len(x))

    for i, (xi, yi) in enumerate(zip(x, y)):
        # Find which bin this point belongs to
        x_bin = np.searchsorted(xedges, xi, side='right') - 1
        y_bin = np.searchsorted(yedges, yi, side='right') - 1

        # Clamp to valid range (0 to n_bins-1)
        # H.shape is (ny, nx) where ny = len(yedges)-1, nx = len(xedges)-1
        x_bin = np.clip(x_bin, 0, H.shape[1] - 1)
        y_bin = np.clip(y_bin, 0, H.shape[0] - 1)

        densities[i] = H[y_bin, x_bin]

    return densities


def downsample_pairs_with_density(x: np.ndarray, y: np.ndarray, densities: np.ndarray, max_points: int):
    n = len(x)
    if n <= max_points:
        return x, y, densities
    idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
    return x[idx], y[idx], densities[idx]


def max_label_position(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    idx_flat = int(np.argmax(H))
    ny, nx = H.shape
    row = idx_flat // nx
    col = idx_flat % nx
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    return xcenters[col], ycenters[row], str(int(H[row, col]))


def make_quad_source_sparse(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray):
    ny, nx = H.shape

    # Use all edges as provided
    xleft = xedges[:-1]
    xright = xedges[1:]
    ybottom = yedges[:-1]
    ytop = yedges[1:]

    # Find non-empty bins
    xs_left, xs_right, ys_bottom, ys_top = [], [], [], []
    xcenter, ycenter, counts = [], [], []

    for i in range(ny):
        for j in range(nx):
            c = float(H[i, j])
            if c > 0:  # Only add non-zero bins
                xs_left.append(xleft[j])
                xs_right.append(xright[j])
                ys_bottom.append(ybottom[i])
                ys_top.append(ytop[i])
                xcenter.append(0.5 * (xleft[j] + xright[j]))
                ycenter.append(0.5 * (ybottom[i] + ytop[i]))
                counts.append(c)

    # Calculate tick positions at bin centers
    x_ticks = [0.5 * (xedges[i] + xedges[i+1]) for i in range(len(xedges)-1)]
    y_ticks = [0.5 * (yedges[i] + yedges[i+1]) for i in range(len(yedges)-1)]

    src = ColumnDataSource(dict(
        left=xs_left, right=xs_right, bottom=ys_bottom, top=ys_top,
        x=xcenter, y=ycenter, count=counts
    ))

    return src, x_ticks, y_ticks


def filter_ticks_every_n(ticks: List[float], n: int = 3) -> List[float]:
    """Return every nth tick from the list."""
    if len(ticks) <= 1:
        return ticks
    return [ticks[i] for i in range(0, len(ticks), n)]


# -----------------
# Core builder (returns layout: selector over tabs)
# -----------------
def build_layout(root: str,
                 dataset_tags: List[str],
                 feature1: str,
                 feature2: str,
                 n_bins_feat1: Optional[int] = None,
                 n_bins_feat2: Optional[int] = None,
                 scatter_max_points: int = 100_000,
                 force_recalc: bool = False):
    available = discover_available_datasets(root)
    cache_dir = ensure_cache_dir(root)

    valid_tags = [t for t in dataset_tags if t in available]
    missing = [t for t in dataset_tags if t not in available]
    if missing:
        print(f"[warn] Missing dataset_tags (ignored): {missing}")
    if not valid_tags:
        raise RuntimeError("None of the provided dataset_tags exist under the root path.")

    # Load datasets
    dataset_feature_arrays = {}
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

    # Global ranges (force [0,1] when no bins specified)
    if n_bins_feat1 is None or n_bins_feat2 is None:
        x_range = (0.0, 1.0)
        y_range = (0.0, 1.0)
    else:
        all_x = np.concatenate([dataset_feature_arrays[tag][feature1] for tag in valid_tags])
        all_y = np.concatenate([dataset_feature_arrays[tag][feature2] for tag in valid_tags])
        x_min, x_max = np.percentile(all_x, [1, 99])
        y_min, y_max = np.percentile(all_y, [1, 99])
        pad_x = 0.02 * (x_max - x_min) if x_max > x_min else 1.0
        pad_y = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
        x_range = (float(x_min - pad_x), float(x_max + pad_x))
        y_range = (float(y_min - pad_y), float(y_max + pad_y))

    # Create shared range objects
    x_range_obj = Range1d(x_range[0], x_range[1])
    y_range_obj = Range1d(y_range[0], y_range[1])

    # Build heatmaps
    heatmap_meta = {}
    quad_sources = {}
    labels_meta = {}
    xticks_meta = {}
    yticks_meta = {}
    xticks_labels_meta = {}
    yticks_labels_meta = {}

    print("Preparing heatmap caches...")
    for tag in tqdm(valid_tags, desc="Heatmaps"):
        data_path = available[tag]
        cache_path = cache_key(cache_dir, tag, feature1, feature2, n_bins_feat1, n_bins_feat2)
        x = dataset_feature_arrays[tag][feature1]
        y = dataset_feature_arrays[tag][feature2]

        if is_cache_valid(cache_path, data_path, force_recalc):
            cached = np.load(cache_path, allow_pickle=True)
            H = cached["H"]
            xedges = cached["xedges"]
            yedges = cached["yedges"]
        else:
            H, xedges, yedges = compute_hist2d_custom(x, y, x_range, y_range, n_bins_feat1, n_bins_feat2)
            np.savez_compressed(cache_path, H=H, xedges=xedges, yedges=yedges)

        heatmap_meta[tag] = {"H": H.astype(float), "xedges": xedges, "yedges": yedges}
        src, x_ticks, y_ticks = make_quad_source_sparse(H, xedges, yedges)
        quad_sources[tag] = src
        xticks_meta[tag] = x_ticks
        yticks_meta[tag] = y_ticks

        # Create labels for every 4th tick
        x_labels = {tick: f"{tick:.2f}" if i % 4 == 0 else ""
                    for i, tick in enumerate(x_ticks)}
        y_labels = {tick: f"{tick:.2f}" if i % 4 == 0 else ""
                    for i, tick in enumerate(y_ticks)}
        xticks_labels_meta[tag] = x_labels
        yticks_labels_meta[tag] = y_labels

        lx, ly, ltext = max_label_position(H, xedges, yedges)
        labels_meta[tag] = {"x": [lx], "y": [ly], "text": [ltext]}

    # Scatter with density coloring
    rng = np.random.default_rng(12345)
    jitter_x = (x_range[1] - x_range[0]) * 0.005
    jitter_y = (y_range[1] - y_range[0]) * 0.005

    scatter_sources = {}
    print("Preparing scatter data...")
    for tag in tqdm(valid_tags, desc="Scatter"):
        x = dataset_feature_arrays[tag][feature1]
        y = dataset_feature_arrays[tag][feature2]
        H = heatmap_meta[tag]["H"]
        xedges = heatmap_meta[tag]["xedges"]
        yedges = heatmap_meta[tag]["yedges"]
        densities = compute_point_densities(x, y, H, xedges, yedges)
        xs, ys, ds = downsample_pairs_with_density(x, y, densities, scatter_max_points)
        xs_vis = xs + rng.uniform(-jitter_x, jitter_x, size=xs.shape)
        ys_vis = ys + rng.uniform(-jitter_y, jitter_y, size=ys.shape)
        scatter_sources[tag] = ColumnDataSource(dict(
            x=xs_vis, y=ys_vis, density=ds
        ))

    # Create figures with shared ranges
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    initial_H = heatmap_meta[valid_tags[0]]["H"]
    color_mapper = LinearColorMapper(
        palette=Viridis256,
        low=0,
        high=float(np.max(initial_H)) if np.max(initial_H) > 0 else 1.0
    )

    # Scatter plot
    scatter_fig = figure(
        width=750, height=640,
        x_range=x_range_obj, y_range=y_range_obj,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        title=f"Scatter: {feature1} vs {feature2}",
        output_backend="webgl"
    )
    scatter_renderer = scatter_fig.circle(
        "x", "y", source=scatter_sources[valid_tags[0]],
        size=5, fill_alpha=0.3, line_alpha=0,
        fill_color={'field': 'density', 'transform': color_mapper}
    )

    # Heatmap plot
    heatmap_fig = figure(
        width=750, height=640,
        x_range=x_range_obj, y_range=y_range_obj,
        tools=TOOLS,
        title=f"Heatmap: {feature1} vs {feature2}",
        output_backend="webgl"
    )
    quad_renderer = heatmap_fig.quad(
        left="left", right="right", bottom="bottom", top="top",
        source=quad_sources[valid_tags[0]],
        fill_color={'field': 'count', 'transform': color_mapper},
        fill_alpha=1.0, line_alpha=0.0
    )

    # Add hover tools
    heatmap_hover = HoverTool(
        tooltips=[
            (f"{feature1}", "@x{0.00}"),
            (f"{feature2}", "@y{0.00}"),
            ("Count", "@count{0,0}")
        ],
        renderers=[quad_renderer]
    )
    heatmap_fig.add_tools(heatmap_hover)

    # Add max count label
    label_source = ColumnDataSource(data=labels_meta[valid_tags[0]])
    labels = LabelSet(
        x='x', y='y', text='text', source=label_source,
        text_baseline="middle", text_align="center",
        text_font_size="10pt", text_color="white",
        background_fill_color="black", background_fill_alpha=0.25
    )
    heatmap_fig.add_layout(labels)

    # Add color bars
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=8,
        location=(0, 0)
    )
    heatmap_fig.add_layout(color_bar, 'right')
    scatter_fig.add_layout(color_bar, 'right')

    # Configure axes
    for fig in [scatter_fig, heatmap_fig]:
        fig.xaxis.axis_label = feature1
        fig.yaxis.axis_label = feature2
        fig.match_aspect = True
        fig.grid.grid_line_alpha = 0.4

        # Use the same tickers and formatters
        fig.xaxis.ticker = FixedTicker(ticks=xticks_meta[valid_tags[0]])
        fig.yaxis.ticker = FixedTicker(ticks=yticks_meta[valid_tags[0]])

        x_formatter = FuncTickFormatter(code=f"""
            var labels = {xticks_labels_meta[valid_tags[0]]};
            return labels[tick] || "";
        """)
        y_formatter = FuncTickFormatter(code=f"""
            var labels = {yticks_labels_meta[valid_tags[0]]};
            return labels[tick] || "";
        """)
        fig.xaxis.formatter = x_formatter
        fig.yaxis.formatter = y_formatter

    # Create selector and callback
    selector = Select(title="Dataset", value=valid_tags[0], options=valid_tags)

    # Prepare JavaScript callback payloads
    payload_scatter = {tag: dict(
        x=_to_pylist(scatter_sources[tag].data["x"]),
        y=_to_pylist(scatter_sources[tag].data["y"]),
        density=_to_pylist(scatter_sources[tag].data["density"])
    ) for tag in valid_tags}

    payload_quads = {tag: dict(
        left=_to_pylist(quad_sources[tag].data["left"]),
        right=_to_pylist(quad_sources[tag].data["right"]),
        bottom=_to_pylist(quad_sources[tag].data["bottom"]),
        top=_to_pylist(quad_sources[tag].data["top"]),
        x=_to_pylist(quad_sources[tag].data["x"]),
        y=_to_pylist(quad_sources[tag].data["y"]),
        count=_to_pylist(quad_sources[tag].data["count"])
    ) for tag in valid_tags}

    callback = CustomJS(args=dict(
        selector=selector,
        scatter_src=scatter_sources[valid_tags[0]],
        quad_src=quad_sources[valid_tags[0]],
        color_mapper=color_mapper,
        label_src=label_source,
        scatter_fig=scatter_fig,
        heatmap_fig=heatmap_fig,
        scatter_payload=payload_scatter,
        quad_payload=payload_quads,
        label_payload=labels_meta,
        maxH_payload={tag: float(np.max(heatmap_meta[tag]["H"])) for tag in valid_tags},
        xticks_payload=xticks_meta,
        yticks_payload=yticks_meta,
        xlabels_payload=xticks_labels_meta,
        ylabels_payload=yticks_labels_meta
    ), code="""
    const tag = selector.value;

    // Update data sources
    scatter_src.data = scatter_payload[tag];
    scatter_src.change.emit();

    quad_src.data = quad_payload[tag];
    quad_src.change.emit();

    label_src.data = label_payload[tag];
    label_src.change.emit();

    // Update color scale
    color_mapper.high = maxH_payload[tag] > 0 ? maxH_payload[tag] : 1.0;

    // Sync view ranges
    const x_start = scatter_fig.x_range.start;
    const x_end = scatter_fig.x_range.end;
    const y_start = scatter_fig.y_range.start;
    const y_end = scatter_fig.y_range.end;

    heatmap_fig.x_range.start = x_start;
    heatmap_fig.x_range.end = x_end;
    heatmap_fig.y_range.start = y_start;
    heatmap_fig.y_range.end = y_end;

    // Update tickers (must be after range updates)
    scatter_fig.xaxis.ticker.ticks = xticks_payload[tag];
    scatter_fig.yaxis.ticker.ticks = yticks_payload[tag];
    heatmap_fig.xaxis.ticker.ticks = xticks_payload[tag];
    heatmap_fig.yaxis.ticker.ticks = yticks_payload[tag];

    // Update formatters
    scatter_fig.xaxis.formatter.code = 'var labels=' + JSON.stringify(xlabels_payload[tag]) + ';return labels[tick]||"";';
    scatter_fig.yaxis.formatter.code = 'var labels=' + JSON.stringify(ylabels_payload[tag]) + ';return labels[tick]||"";';
    heatmap_fig.xaxis.formatter.code = scatter_fig.xaxis.formatter.code;
    heatmap_fig.yaxis.formatter.code = scatter_fig.yaxis.formatter.code;
    """)

    selector.js_on_change("value", callback)

    # Create tabs
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
                output_html_path: str = "triple_stream_plots.html",
                force_recalc: bool = False) -> str:
    layout = build_layout(root, dataset_tags, feature1, feature2,
                          n_bins_feat1=n_bins_feat1, n_bins_feat2=n_bins_feat2,
                          scatter_max_points=scatter_max_points,
                          force_recalc=force_recalc)
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
    parser.add_argument("--force-recalc", action="store_true",
                        help="Force recalculation of histograms, ignoring cache")

    args = parser.parse_args()

    path = build_plots(
        root=args.root,
        dataset_tags=args.tags,
        feature1=args.feature1,
        feature2=args.feature2,
        n_bins_feat1=args.n_bins_feat1,
        n_bins_feat2=args.n_bins_feat2,
        scatter_max_points=args.scatter_max_points,
        output_html_path=args.output,
        force_recalc=args.force_recalc
    )
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()