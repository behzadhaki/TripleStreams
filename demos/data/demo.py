import torch

from data import load_original_gmd_dataset_pickle

# Load 2bar gmd dataset as a dictionary
gmd_dict = load_original_gmd_dataset_pickle(
    gmd_pickle_path="data/gmd/resources/storedDicts/groove_2bar-midionly.bz2pickle")

gmd_dict.keys()          # dict_keys(['train', 'test', 'validation'])
gmd_dict['train'].keys()        #dict_keys(['drummer', 'session', 'loop_id', 'master_id', 'style_primary', 'style_secondary', 'bpm', 'beat_type', 'time_signature', 'full_midi_filename', 'full_audio_filename', 'midi', 'note_sequence'])


# =================================================================================================
# Extract HVO_Sequences from the dictionaries

from data import extract_hvo_sequences_dict, get_drum_mapping_using_label

hvo_dict = extract_hvo_sequences_dict (
    gmd_dict=gmd_dict,
    beat_division_factor=[4],
    drum_mapping=get_drum_mapping_using_label("ROLAND_REDUCED_MAPPING"))

# =================================================================================================

# Load GMD Dataset in `HVO_Sequence` format using a single command
from data import load_bz2_hvo_sequences
train_set = load_bz2_hvo_sequences(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    force_regenerate=False)

# =================================================================================================

# Load Mega Dataset in `HVO_Sequence` format using a single command
from data import load_bz2_hvo_sequences
train_set = load_bz2_hvo_sequences(
    dataset_setting_json_path="data/dataset_json_settings/Balanced_1000_performed_no_GMD.json",
    subset_tag="test",
    force_regenerate=False)
collections = [hvo_seq.metadata["drummer"] for hvo_seq in train_set]
# get counts of each collection
from collections import Counter
collection_counts = Counter(collections)
collection_counts

# =================================================================================================
# Load dataset as torch.utils.data.Dataset
from data import MonotonicGrooveDataset

# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False,
    load_as_tensor=True,
    sort_by_metadata_key="loop_id",
    genre_loss_balancing_beta=0.5
)

training_dataset.__getitem__(0)


# =================================================================================================
# Load Mega dataset as torch.utils.data.Dataset
from data import Groove2Drum2BarDataset

# load dataset as torch.utils.data.Dataset
down_sampled_train_set = Groove2Drum2BarDataset(
    dataset_setting_json_path="data/dataset_json_settings/Balanced_100_per_genre_performed_4_4.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    down_sampled_ratio=None,
    collapse_tapped_sequence=False,
    sort_by_metadata_key="loop_id",
    num_voice_density_bins=3,
    num_tempo_bins=6,
    augment_dataset=True,
)

a = down_sampled_train_set[0]
kick_counts = down_sampled_train_set.kick_counts
kick_density_bins = down_sampled_train_set.kick_density_bins
tempo_bins = down_sampled_train_set.tempo_bins

# count unique values in a list
from collections import Counter
Counter(kick_counts), Counter(kick_density_bins.numpy()), Counter(tempo_bins.numpy())



# =================================================================================================
# Load Mega dataset as torch.utils.data.Dataset
from data import Groove2Drum2BarDataset

# load dataset as torch.utils.data.Dataset
training_dataset = Groove2Drum2BarDataset(
    dataset_setting_json_path="data/dataset_json_settings/Balanced_5000_per_genre_performed_4_4.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    down_sampled_ratio=None,
    collapse_tapped_sequence=True,
    sort_by_metadata_key="loop_id",
    num_voice_density_bins=3,
    num_tempo_bins=6,
    num_global_density_bins=7,
    augment_dataset=False,
    force_regenerate=True
)


def get_bins_for_global_density(total_hits_per_sample: list, num_bins=8):
    """
    Calculates the lower and upper bounds for the global density bins

    :param total_hits_per_sample:
    :param num_bins:
    :return: lower_bounds, upper_bounds
    """

    assert num_bins > 0, "num_bins should be greater than 0"

    total_hits_per_sample = sorted(total_hits_per_sample)

    samples_per_bin = len(total_hits_per_sample) // num_bins

    grouped_bins = [total_hits_per_sample[i * samples_per_bin: (i + 1) * samples_per_bin] for i in range(num_bins)]

    lower_bounds = [group[0] for group in grouped_bins]
    upper_bounds = [group[-1] for group in grouped_bins]
    upper_bounds[-1] = total_hits_per_sample[-1] + 10  # to ensure that the last bin is inclusive on the upper bound

    return lower_bounds, upper_bounds

def map_voice_densities_to_categorical(voice_counts, lower_bounds, upper_bounds):
    """
    Maps the voice counts to a categorical value based on the lower and upper bounds provided
    :param voice_counts:
    :param lower_bounds:
    :param upper_bounds:
    :return:
    """

    categories = []
    adjusted_upper_bounds = upper_bounds.copy()
    adjusted_upper_bounds[-1] = adjusted_upper_bounds[-1] + 1  # to ensure that the last bin is inclusive on the upper bound

    for count in voice_counts:
        if count == 0:
            categories.append(0)
        else:
            for idx, (low, high) in enumerate(zip(lower_bounds, adjusted_upper_bounds)):
                if low <= count < high:
                    categories.append(idx + 1)
                    break

    return categories

out_hits = training_dataset.output_grooves[:, :, :9]

import torch
hvo_hits = [hvo_.hits for hvo_ in training_dataset.hvo_sequences]
hvo_hits = torch.tensor(hvo_hits)

total_hits_per_sample = hvo_hits.flatten(start_dim=1).sum(dim=1)
hvo_sums = [hvo_.hits.sum() for hvo_ in training_dataset.hvo_sequences]
hvo_sums = torch.tensor(hvo_sums)
torch.all((hvo_sums == total_hits_per_sample))
hvo_sums
total_hits_per_sample

mapped = map_voice_densities_to_categorical(hvo_sums, *get_bins_for_global_density(hvo_sums.numpy(), num_bins=7))

mapped = torch.tensor(mapped)
mapped == training_dataset.global_density_bins

mapped[0], training_dataset.global_density_bins[0]

from collections import Counter
Counter(mapped.numpy())


training_dataset.global_density_highBound_full_trnSet

out_hits = training_dataset.output_grooves[:, :, :9]
out_vels = training_dataset.output_grooves[:, :, 9:18]
get_bins_for_global_density(out_hits.flatten(start_dim=1).sum(dim=1).numpy(), num_bins=7)
in_hits = training_dataset.input_grooves[:, :, :1]
training_dataset.input_grooves.shape
# sum each example in the batch
out_hits_sum = out_hits.flatten(start_dim=1).sum(dim=1)
out_vels_sum = out_vels.flatten(start_dim=1).sum(dim=1)
in_hits_sum = in_hits.flatten(start_dim=1).sum(dim=1)
print(sorted(out_hits_sum.numpy()))
density_bins = training_dataset.global_density_bins.numpy()
density_bins.shape
from collections import Counter
Counter(density_bins)
# assign 10 colors

in_hits_sum.shape
in_hits_sum
ratio = (out_hits_sum / in_hits_sum).numpy()
average_vels = (out_vels_sum / in_hits_sum).numpy()
ratio.shape
# plot hits sum (x) vs average velocity (y), use color associated density bin
import matplotlib.pyplot as plt
colors = plt.cm.get_cmap('tab10', 10)
plt.scatter(out_hits_sum.numpy(), average_vels, c=density_bins, cmap=colors, alpha=0.5, marker='o', s=1)
plt.show()

Counter(density_bins)

# plot a histogram of the ratio values (using a 0.1 bin size)
import matplotlib.pyplot as plt
plt.hist(out_hits_sum.numpy(), bins=20)
plt.show()


print("len(test_dataset.kick_density_bins) = ", len(training_dataset.kick_density_bins))
print("len(test_dataset.snare_density_bins) = ", len(training_dataset.snare_density_bins))
training_dataset.genre_mapping
print(training_dataset.global_density_lowBound_full_trnSet, "\n", training_dataset.global_density_highBound_full_trnSet)

from collections import Counter
Counter(training_dataset.global_density_bins.numpy())
print("Kick Mutes: ", Counter(training_dataset.kick_is_muted.numpy()))
print("Snare Mutes: ", Counter(training_dataset.snare_is_muted.numpy()))
print("Hat Mutes: ", Counter(training_dataset.hat_is_muted.numpy()))
print("Tom Mutes: ", Counter(training_dataset.tom_is_muted.numpy()))
print("Cymbal Mutes: ", Counter(training_dataset.cymbal_is_muted.numpy()))

hist = training_dataset.get_genre_histogram()
import matplotlib.pyplot as plt
plt.title("Genre Histogram")
plt.bar(hist.keys(), hist.values())
# rotate x labels 90 degrees
plt.xticks(rotation=90)
plt.show()

hist = training_dataset.get_tempos_per_genre()
import matplotlib.pyplot as plt
plt.title("Tempos per Genre")
plt.rc('font', size=8)
# limit y to 40 - 200
plt.ylim(40, 200)
plt.xticks(rotation=90)
plt.boxplot(hist.values(), labels=hist.keys())
# small font
plt.show()

hist = training_dataset.get_complexities_per_genre()
import matplotlib.pyplot as plt
plt.title("Complexities per Genre")
plt.rc('font', size=8)
plt.xticks(rotation=90)
plt.boxplot(hist.values(), labels=hist.keys())
# small font
plt.show()

# training_dataset.visualize_global_hit_count_ratios_heatmap()
#training_dataset.visualize_genre_distributions(show_inverted_weights=True)
# =================================================================================================

# use the above dataset in the training pipeline, you need to use torch.utils.data.DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

epochs = 10
for epoch in range(epochs):
    # in each epoch we iterate over the entire dataset
    for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
        print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
              f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")