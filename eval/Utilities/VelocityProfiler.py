import numpy as np
from hvo_sequence.hvo_seq import HVO_Sequence

import os, bz2, pickle
from pandas import DataFrame

from bokeh.palettes import Category10
from bokeh.plotting import figure, show, save
from bokeh.io import output_file

from bokeh.embed import file_html
from bokeh.resources import CDN
import wandb
from bokeh.models import Panel, Tabs

from bokeh.layouts import column


def extract_flat_velocity_profile_time_step_resolution(hvo_seq_list: list[HVO_Sequence],
                                                       time_steps: int = 32,
                                                       ignore_non_hits: bool = True):
    """
    The function extracts the velocity profile of the hvo sequence list

    The profile is extracted as follows:
    - hvo is flattened to a single v and o sequence
    - velocities and timings are extracted per time step
    - the Q1, Median and Q3 are extracted for each time step (for both velocities and timings)

    :param hvo_seq_list: list of HVO_Sequence
    :param time_steps: int, default (32)
    :param ignore_non_hits: bool, default (True)
    :return: (q1, median, q3) for velocities and timings. Each element is a tuple of (timestep+offset, velocity)
            That is, Q1 = [(real_time q1 at time 0, velocity q1 at time 0), (real_time Q1 at time 1, velocity q1 at time 1), ...]
    """

    velocity_data = [[] for _ in range(time_steps)]
    offset_data = [[] for _ in range(time_steps)]

    for hvo_seq in hvo_seq_list:
        hvo = hvo_seq.flatten_voices(reduce_dim=True)[:time_steps, :]
        hits = hvo[:, 0]
        velocities = hvo[:, 1]
        offsets = hvo[:, 2]
        for i in range(hvo.shape[0]):
            if ignore_non_hits and hits[i] == 0:
                continue
            else:
                velocity_data[i].append(velocities[i])
                offset_data[i].append(offsets[i])

    q1 = []
    median = []
    q3 = []

    for ts, (vs, os) in enumerate(zip(velocity_data, offset_data)):
        vs_ = np.array(vs)
        os_ = np.array(os)
        if len(vs_) == 0:
            q1.append((ts, 0))
            median.append((ts, 0))
            q3.append((ts, 0))
        else:
            q1.append((np.percentile(os_, 25) + ts, np.percentile(vs_, 25)))
            median.append((np.percentile(os_, 50) + ts, np.percentile(vs_, 50)))
            q3.append((np.percentile(os_, 75) + ts, np.percentile(vs_, 75)))

    return q1, median, q3


def plot_flat_profile_time_step_resolution(hvo_seq_list: list[HVO_Sequence],
                                           time_steps: int = 32,
                                           return_as_wandb_html: bool = False, ):
    q1, median, q3 = extract_flat_velocity_profile_time_step_resolution(hvo_seq_list=hvo_seq_list,
                                                                        time_steps=time_steps, ignore_non_hits=True)

    # output_file("misc/temp_flat_velocity_profile.html")

    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    p = figure(tools=TOOLS, plot_width=600, plot_height=300, title="Flat Velocity Profile")

    # draw a grid where there are vertical lines and the ones on 0, 4, 8, ... are thicker
    for i in range(0, time_steps, 4):
        p.line([i, i], [0, 127], line_width=1, color="black")

    for i in range(0, time_steps):
        p.line([i, i], [0, 127], line_width=0.5, color="black")

    # put major ticks on 0, 4, 8, ...
    p.xaxis.ticker = [i for i in range(0, time_steps, 4)]
    p.xaxis.major_label_overrides = {i: str(i) for i in range(0, time_steps, 4)}

    p.scatter([x[0] for x in q1], [x[1] for x in q1], color="blue")
    p.line([x[0] for x in q1], [x[1] for x in q1], line_width=2, color="blue", legend_label="Q1")
    p.scatter([x[0] for x in median], [x[1] for x in median], color="green")
    p.line([x[0] for x in median], [x[1] for x in median], line_width=2, color="green", legend_label="Median")
    p.scatter([x[0] for x in q3], [x[1] for x in q3], color="red")
    p.line([x[0] for x in q3], [x[1] for x in q3], line_width=2, color="red", legend_label="Q3")

    p.legend.click_policy = "hide"

    # y axis 0 to 1
    p.y_range.start = 0
    p.y_range.end = 1

    save(p)

    if return_as_wandb_html:
        return wandb.Html(file_html(p, CDN, "temp_flat_velocity_profile.html"))
    else:
        return p


def extract_flat_velocity_profile(hvo_seq_list: list[HVO_Sequence],
                                  n_bins_per_step: int = 1,
                                  time_steps: int = 32,
                                  ignore_non_hits: bool = True,
                                  use_IQR: bool = True):
    """
    The function extracts the velocity profile of the hvo sequence list

    The profile is extracted as follows:
    - hvo is flattened to a single v and o sequence
    - velocities and timings are extracted per time step
    - the Q1, Median and Q3 are extracted for each time step (for both velocities and timings)

    :param hvo_seq_list: list of HVO_Sequence
    :param time_steps: int, default (32)
    :param ignore_non_hits: bool, default (True)
    :return: (time_stamps, q1, median, q3) for velocities and timings. Each element is a tuple of (time, velocity)
            That is, Q1 = [(real_time q1 at time 0, velocity q1 at time 0), (real_time Q1 at time 1, velocity q1 at time 1), ...]
            The shape of the output is (time_steps*n_bins_per_step)
    """
    time_velocity_data = []

    for hvo_seq in hvo_seq_list:
        hvo = hvo_seq.flatten_voices(reduce_dim=True)
        hits = hvo[:, 0]
        velocities = hvo[:, 1]
        offsets = hvo[:, 2]
        for i in range(hvo.shape[0]):
            if ignore_non_hits and hits[i] == 0:
                continue
            else:
                time_velocity_data.append((offsets[i]+i, velocities[i]))

    time_stamps = np.linspace(0, time_steps, time_steps * n_bins_per_step)

    times = [[] for _ in range(time_steps * n_bins_per_step)]
    vels = [[] for _ in range(time_steps * n_bins_per_step)]

    step_res = 1 / n_bins_per_step

    for t, v in time_velocity_data:
        idx = int(t // step_res)
        times[idx].append(t)
        vels[idx].append(v)

    q1 = []
    median = []
    q3 = []
    available_time_stamps = []

    for ts, (ts_, vs) in enumerate(zip(times, vels)):
        vs_ = np.array(vs)
        if len(vs_) != 0:
            if use_IQR:
                q1.append((time_stamps[ts], np.percentile(vs_, 25)))
                median.append((time_stamps[ts], np.percentile(vs_, 50)))
                q3.append((time_stamps[ts], np.percentile(vs_, 75)))
                available_time_stamps.append(time_stamps[ts])
            else:
                m_ = np.mean(vs_)
                std = np.std(vs_)
                q1.append((time_stamps[ts], m_ - std))
                median.append((time_stamps[ts], m_))
                q3.append((time_stamps[ts], m_ + std))
                available_time_stamps.append(time_stamps[ts])

    return available_time_stamps, q1, median, q3, time_velocity_data


def plot_flat_profile(hvo_seq_list: list[HVO_Sequence],
                      n_bins_per_step: int = 1,
                      time_steps: int = 32,
                      return_as_wandb_html: bool = False,
                      filter_genres: list[str] = None,
                      use_IQR: bool = True,
                      show_scatter_data: bool = True
                      ):
    hvo_seq_list = hvo_seq_list if filter_genres is None else [hvo_seq for hvo_seq in hvo_seq_list if hvo_seq.metadata["style_primary"] in filter_genres]

    time_stamps, q1, median, q3, time_velocity_data = extract_flat_velocity_profile(
        hvo_seq_list=hvo_seq_list,
        n_bins_per_step=n_bins_per_step,
        time_steps=time_steps, ignore_non_hits=True, use_IQR=use_IQR)

    # output_file("misc/temp_flat_velocity_profile.html")

    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    title = "/".join(filter_genres) if filter_genres is not None else "All Genres"
    p_vel = figure(tools=TOOLS, plot_width=1000, plot_height=500, title=title)

    # draw a grid where there are vertical lines and the ones on 0, 4, 8, ... are thicker
    for i in range(0, time_steps, 16):
        p_vel.line([i, i], [-100, 127], line_width=2, color="black")

    for i in range(0, time_steps, 4):
        p_vel.line([i, i], [-100, 127], line_width=1, color="black", line_alpha=0.5)

    for i in range(0, time_steps):
        #dashed
        p_vel.line([i, i], [-100, 127], line_width=0.5, color="black", line_dash="dashed", line_alpha=0.5)

    # put major ticks on 0, 4, 8, ...
    p_vel.xaxis.ticker = [i for i in range(0, time_steps, 4)]
    p_vel.xaxis.major_label_overrides = {i: str(i) for i in range(0, time_steps, 4)}


    if show_scatter_data:
        scatter_point_x_in_IQR = []
        scatter_point_y_in_IQR = []

        scatter_point_x_out_IQR = []
        scatter_point_y_out_IQR = []
        time_stamps = np.array(time_stamps)
        # plot small transparent circles for time_velocity_data (if within q1 and q3)
        for t, v in time_velocity_data:
            # get index of closest time stamp
            t_ = np.argmin(np.abs(time_stamps - t))

            if q1[t_][1] <= v <= q3[t_][1]:
                scatter_point_x_in_IQR.append(t)
                scatter_point_y_in_IQR.append(v)
            else:
                scatter_point_x_out_IQR.append(t)
                scatter_point_y_out_IQR.append(v)


        p_vel.scatter(scatter_point_x_in_IQR, scatter_point_y_in_IQR, color="turquoise", alpha=0.1, legend_label="Data Within Bounds")
        scatter_out_IQR = p_vel.scatter(scatter_point_x_out_IQR, scatter_point_y_out_IQR, color="red", alpha=0.1, legend_label="Data Outside Bounds")
        scatter_out_IQR.visible = False


    p_vel.line([x[0] for x in q1], [x[1] for x in q1], line_width=1, color="blue", legend_label="Q1" if use_IQR else "Mean-Std")
    p_vel.line([x[0] for x in median], [x[1] for x in median], line_width=1, color="green", legend_label="Median" if use_IQR else "Mean")
    p_vel.line([x[0] for x in q3], [x[1] for x in q3], line_width=1, color="red", legend_label="Q3" if use_IQR else "Mean+Std")

    p_vel.legend.click_policy = "hide"

    # show legend in a row at the center bottom
    p_vel.legend.location = "bottom_center"
    p_vel.legend.orientation = "horizontal"

    # y axis 0 to 1
    p_vel.y_range.start = -0.2
    p_vel.y_range.end = 1.2
    
    # ---------------------------------------------------------------------------------------------------------------
    # hit statistics
    # ---------------------------------------------------------------------------------------------------------------
    hit_counts_flat = np.zeros(time_steps)
    hit_counts_full = np.zeros((time_steps, 9))

    for hvo_seq in hvo_seq_list:
        hvo_flat = hvo_seq.flatten_voices(reduce_dim=True)[:time_steps, :]
        hits_flat = hvo_flat[:, 0]
        hits_full = hvo_seq.hits

        for i in range(hvo_flat.shape[0]):
            hit_counts_flat[i] += hits_flat[i]
            hit_counts_full[i] += hits_full[i, :]

    voice_labels = ["K", "S", "CH", "OH", "LT", "MT", "HT", "C", "R"]

    totals_full = np.sum(hit_counts_full, axis=1)

    p_hit_counts = figure(tools=TOOLS, plot_width=1000, plot_height=400, title="Hit Counts")
    p_hit_normalized = figure(tools=TOOLS, plot_width=1000, plot_height=400, title="Hit Counts Normalized")

    # draw a grid where there are vertical lines and the ones on 0, 4, 8, ... are thicker
    for i in range(0, time_steps, 16):
        p_hit_counts.line([i, i], [-127, 127], line_width=2, color="black")
        p_hit_normalized.line([i, i], [-127, 127], line_width=2, color="black")

    for i in range(0, time_steps, 4):
        p_hit_counts.line([i, i], [127, 127], line_width=1, color="black", line_alpha=0.5)
        p_hit_normalized.line([i, i], [-127, 127], line_width=1, color="black", line_alpha=0.5)

    for i in range(0, time_steps):
        #dashed
        p_hit_counts.line([i, i], [-1000, 1000], line_width=0.5, color="black", line_dash="dashed", line_alpha=0.5)
        p_hit_normalized.line([i, i], [-127, 127], line_width=0.5, color="black", line_dash="dashed", line_alpha=0.5)

    # put major ticks on 0, 4, 8, ...
    p_hit_counts.xaxis.ticker = [i for i in range(0, time_steps, 4)]
    p_hit_normalized.xaxis.ticker = [i for i in range(0, time_steps, 4)]
    p_hit_counts.xaxis.major_label_overrides = {i: str(i) for i in range(0, time_steps, 4)}
    p_hit_normalized.xaxis.major_label_overrides = {i: str(i) for i in range(0, time_steps, 4)}

    # plot histograms of flat hits (totals and normalized)
    # show a large square
    p_hit_counts.scatter([i for i in range(time_steps)], hit_counts_flat, color="black", size=10, legend_label="Flat")

    # plot histograms of full hits (totals and normalized)
    times_ = np.zeros((time_steps, 9))
    for t in range(time_steps):
        for i in range(9):
            times_[t, i] = t + (- 0.3 + 0.1/2 * i)

    for i in range(9):
        p1_ = p_hit_counts.scatter(times_[:, i], hit_counts_full[:, i], color=Category10[9][i], size=10, legend_label=f"{voice_labels[i]}")
        p2_ = p_hit_normalized.scatter(times_[:, i], hit_counts_full[:, i] / totals_full, color=Category10[9][i], size=10, legend_label=f"{voice_labels[i]}")
        p1_.visible = False
        p2_.visible = False
    # hide policy
    p_hit_counts.legend.click_policy = "hide"
    p_hit_normalized.legend.click_policy = "hide"

    p_hit_counts.y_range.start = 0

    p_hit_normalized.y_range.start = -0.2
    p_hit_normalized.y_range.end = 1.2

    # show legend in a row at the center bottom
    p_hit_counts.legend.location = "right"
    p_hit_counts.legend.orientation = "vertical"
    p_hit_normalized.legend.location = "right"
    p_hit_normalized.legend.orientation = "vertical"

    # set legend font size to 8
    p_hit_counts.legend.label_text_font_size = "8pt"
    p_hit_normalized.legend.label_text_font_size = "8pt"

    # Create bokeh a vertical layout of p_vel, p_hit_counts, p_hit_normalized
    p = column(p_vel, p_hit_counts, p_hit_normalized)

    if return_as_wandb_html:
        return wandb.Html(file_html(p, CDN, "temp_flat_velocity_profile.html"))
    else:
        return p


def plot_per_flat_genre_velocity_profile(hvo_seq_list: list[HVO_Sequence],
                                         genres: list[str] = None,
                                         n_bins_per_step: int = 1,
                                         time_steps: int = 32,
                                         return_as_wandb_html: bool = False,
                                         use_IQR: bool = True,
                                         show_scatter_data: bool = True):
    tabs = []

    tabs.append(("All", plot_flat_profile(hvo_seq_list=hvo_seq_list,
                                          n_bins_per_step=n_bins_per_step,
                                          time_steps=time_steps,
                                          return_as_wandb_html=False,
                                          filter_genres=None,
                                          use_IQR=use_IQR,
                                          show_scatter_data=show_scatter_data)))

    if genres is None:
        genres = sorted(list(set([hvo_seq.metadata["style_primary"] for hvo_seq in hvo_seq_list])))
        print(genres)
    

    for genre in genres:
        hvo_seq_list_genre = [hvo_seq for hvo_seq in hvo_seq_list if hvo_seq.metadata["style_primary"] == genre]

        tabs.append((genre, plot_flat_profile(hvo_seq_list=hvo_seq_list_genre,
                              n_bins_per_step=n_bins_per_step,
                              time_steps=time_steps,
                              return_as_wandb_html=False,
                              filter_genres=[genre],
                              use_IQR=use_IQR,
                              show_scatter_data=show_scatter_data)))

    tabs = [Panel(child=p, title=genre) for genre, p in tabs]

    tabs = Tabs(tabs=tabs)

    if return_as_wandb_html:
        return wandb.Html(file_html(tabs, CDN, "temp_per_flat_genre_velocity_profile.html"))
    else:
        return tabs


if __name__ == "__main__":
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset
    from data import Groove2Drum2BarDataset

    # load dataset as torch.utils.data.Dataset
    training_dataset = Groove2Drum2BarDataset(
        dataset_setting_json_path="data/dataset_json_settings/Balanced_5000_per_genre_performed_4_4.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        down_sampled_ratio=None,
        collapse_tapped_sequence=False,
        sort_by_metadata_key="loop_id",
        num_voice_density_bins=3,
        num_tempo_bins=6,
        augment_dataset=False,
    )

    p = plot_per_flat_genre_velocity_profile(
        hvo_seq_list=training_dataset.hvo_sequences,
        genres=None,
        n_bins_per_step=1,
        time_steps=32,
        return_as_wandb_html=False,
        use_IQR=True,
        show_scatter_data=False)

    save(p, filename="misc/temp_per_flat_genre_velocity_profile.html")

    # time, q1, median, q3, time_velocity_data = extract_flat_velocity_profile(training_dataset.hvo_sequences, n_bins_per_step=4, time_steps=32, ignore_non_hits=True)
    # p = plot_flat_profile(training_dataset.hvo_sequences, n_bins_per_step=1, time_steps=32, return_as_wandb_html=True,
    #                       filter_genres=["Funk"])
    # q1, median, q3 = extract_flat_velocity_profile_time_step_resolution(training_dataset.hvo_sequences, time_steps=64, ignore_non_hits=True)
    #
    # p = plot_flat_profile_time_step_resolution(training_dataset.hvo_sequences, time_steps=32, return_as_wandb_html=True)
    # p2 = plot_flat_profile_time_step_resolution(training_dataset.hvo_sequences, time_steps=32, return_as_wandb_html=False)

