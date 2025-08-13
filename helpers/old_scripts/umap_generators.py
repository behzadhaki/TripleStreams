from model import BaseVAE, GenreClassifier, MuteVAE, MuteGenreLatentVAE, MuteLatentGenreInputVAE, load_model
import os
import torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
from bokeh.io import save

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_classifier = load_model("trained_models/genre_classifier.pth", GenreClassifier).to(device)
model_BaseVAE_0_2 = load_model("trained_models/base_vae_beta_0_2.pth", BaseVAE).to(device)
model_BaseVAE_0_5 = load_model("trained_models/base_vae_beta_0_5.pth", BaseVAE).to(device)
model_BaseVAE_1_0 = load_model("trained_models/base_vae_beta_1_0.pth", BaseVAE).to(device)
base_models = [model_BaseVAE_0_2, model_BaseVAE_0_5, model_BaseVAE_1_0]
model_MuteVAE_0_2 = load_model("trained_models/mute_vae_beta_0_2.pth", MuteVAE).to(device)
model_MuteVAE_0_5 = load_model("trained_models/mute_vae_beta_0_5.pth", MuteVAE).to(device)
model_MuteVAE_1_0 = load_model("trained_models/mute_vae_beta_1_0.pth", MuteVAE).to(device)
mute_models = [model_MuteVAE_0_2, model_MuteVAE_0_5, model_MuteVAE_1_0]
model_MuteGenreLatentVAE_0_2 = load_model("trained_models/mute_genre_latent_vae_beta_0_2.pth", MuteGenreLatentVAE).to(device)
model_MuteGenreLatentVAE_0_5 = load_model("trained_models/mute_genre_latent_vae_beta_0_5.pth", MuteGenreLatentVAE).to(device)
model_MuteGenreLatentVAE_1_0 = load_model("trained_models/mute_genre_latent_vae_beta_1_0.pth", MuteGenreLatentVAE).to(device)
mute_genre_models = [model_MuteGenreLatentVAE_0_2, model_MuteGenreLatentVAE_0_5, model_MuteGenreLatentVAE_1_0]
model_MuteLatentGenreInputVAE_0_2 = load_model("trained_models/mute_latent_genre_input_vae_beta_0_2.pth", MuteLatentGenreInputVAE).to(device)
model_MuteLatentGenreInputVAE_0_5 = load_model("trained_models/mute_latent_genre_input_vae_beta_0_5.pth", MuteLatentGenreInputVAE).to(device)
model_MuteLatentGenreInputVAE_1_0 = load_model("trained_models/mute_latent_genre_input_vae_beta_1_0.pth", MuteLatentGenreInputVAE).to(device)
mute_genre_input_models = [model_MuteLatentGenreInputVAE_0_2, model_MuteLatentGenreInputVAE_0_5, model_MuteLatentGenreInputVAE_1_0]

model_MuteLatentGenreInputVAE_0_2

from data.src.dataLoaders import Groove2Drum2BarDataset

# dump as bz2pickle file
import bz2
import pickle

# open Groove2Drum2BarDataset.bz2pickle
with bz2.BZ2File('./Groove2Drum2BarDataset.bz2pickle', 'rb') as f:
    dataset = pickle.load(f)

from helpers.eval_utils import generate_umap

betas = [0.2, 0.5, 1.0]

f_names = []
p_s = []
for i, model in enumerate(base_models):
    beta = betas[i]
    fname = f'beta = {beta}'

    p_ = generate_umap(
        model=model,
        device='cuda',
        test_dataset=dataset,
        subset_name=fname
    )

    f_names.append(fname)
    p_s.append(p_)


for i, model in enumerate(mute_models):
    beta = betas[i]
    fname = f'beta = {beta}'

    p_ = generate_umap(
        model=model,
        device='cuda',
        test_dataset=dataset,
        subset_name=fname
    )

    f_names.append(fname)
    p_s.append(p_)

for i, model in enumerate(mute_genre_models):
    beta = betas[i]
    fname = f'beta = {beta}'

    p_ = generate_umap(
        model=model,
        device='cuda',
        test_dataset=dataset,
        subset_name=fname
    )

    f_names.append(fname)
    p_s.append(p_)


for i, model in enumerate(mute_genre_input_models):
    beta = betas[i]
    fname = f'beta = {beta}'

    p_ = generate_umap(
        model=model,
        device='cuda',
        test_dataset=dataset,
        subset_name=fname)

    f_names.append(fname)
    p_s.append(p_)


# show in a 4x3 grid
from bokeh.layouts import gridplot

grid = gridplot(p_s, ncols=3, plot_width=600, plot_height=400)
save(grid, filename="umap_plots.html")

# from bokeh.models import Panel, Tabs
# from copy import deepcopy
#
# # put base models in one tab
# base_tab = Tabs(tabs=[Panel(child=p_i, title=f_name) for p_i, f_name in zip(p_s[:3], f_names[:3])])
#
# # put mute models in one tab
# mute_tab = Tabs(tabs=[Panel(child=p_i, title=f_name) for p_i, f_name in zip(p_s[3:6], f_names[3:6])])
#
# # put mute genre models in one tab
# mute_genre_tab = Tabs(tabs=[Panel(child=p_i, title=f_name) for p_i, f_name in zip(p_s[6:9], f_names[6:9])])
#
# # put mute genre input models in one tab
# mute_genre_input_tab = Tabs(tabs=[Panel(child=p_i, title=f_name) for p_i, f_name in zip(p_s[9:], f_names[9:])])
#
# # put all tabs in one tab
# all_tabs = Tabs(tabs=[Panel(child=base_tab, title='Base'),
#                       Panel(child=mute_tab, title='Mute'),
#                       Panel(child=mute_genre_tab, title='MuteGenre1'),
#                       Panel(child=mute_genre_input_tab, title='Mutegenre2')])
#
#
# save(all_tabs, 'umapTabs.html')
#

