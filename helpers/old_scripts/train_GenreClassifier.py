import os

import tqdm

import wandb

import torch
from model import GenreClassifier
from helpers import train_utils, eval_utils
from data.src.dataLoaders import Groove2Drum2BarDataset
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse

logger = getLogger("train_**.py")
logger.setLevel(DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

parser = argparse.ArgumentParser()

# ----------------------- Set True When Testing ----------------
parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", type=bool,
                    default=False)

# ----------------------- WANDB Settings -----------------------
parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuratio. If available, the rest of the arguments will be ignored", default=None)
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name",
                    default="GenreClassifier")

# ----------------------- Model Parameters -----------------------
# d_model_dec_ratio denotes the ratio of the dec relative to enc size
parser.add_argument("--d_model", type=int, help="Dimension of the encoder model", default=32)
parser.add_argument("--nhead", type=int, help="Number of attention heads for the encoder", default=2)
parser.add_argument("--dim_feedforward", type=float,
                    help="ratio of the dimension of enc feed-frwrd layer relative to enc dmodel", default=128)
parser.add_argument("--num_layers", type=int, help="Number of encoder layers", default=3)
parser.add_argument("--n_dec_lyrs_ratio", type=float, help="Number of decoder layers as a ratio of "
                                               "num_layers as a ratio of d_ff_enc", default=1)

# ----------------------- Training Parameters -----------------------
parser.add_argument("--dropout", type=float, help="Dropout", default=0.4)
parser.add_argument("--velocity_dropout", type=float, help="velocity_dropout", default=0.4)
parser.add_argument("--offset_dropout", type=float, help="offset_dropout", default=0.4)
parser.add_argument("--force_data_on_cuda", type=bool, help="places all training data on cude", default=True)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam' loss", default="sgd",
                    choices=['sgd', 'adam'])
parser.add_argument("--n_genres", type=int, help="Number of genres", default=10)

# ----------------------- Data Parameters -----------------------
parser.add_argument("--dataset_json_dir", type=str,
                    help="Path to the folder hosting the dataset json file",
                    default="data/dataset_json_settings")
parser.add_argument("--dataset_json_fname", type=str,
                    help="fs",
                    default="Balanced_5000_performed_2000_programmed.json")
parser.add_argument("--evaluate_on_subset", type=str,
                    help="Using test or evaluation subset for evaluating the model", default="test",
                    choices=['test', 'evaluation'] )

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/GenreClassifier")
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=5)

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

# Disable wandb logging in testing mode
if args.is_testing:
    os.environ["WANDB_MODE"] = "disabled"

loaded_via_config = False
if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
        if "wandb_project" not in hparams.keys():
            hparams["wandb_project"] = args.wandb_project
        loaded_via_config = True
        print(hparams)
else:
    hparams = dict(
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=int(args.num_layers),
        nhead=args.nhead,
        dropout=args.dropout,
        velocity_dropout=args.velocity_dropout,
        offset_dropout=args.offset_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_genres=args.n_genres,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        dataset_json_dir=args.dataset_json_dir,
        dataset_json_fname=args.dataset_json_fname,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,                         # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],          # name of the project
        entity="behzadhaki",                          # saves in the mmil_vae_cntd team account
        settings=wandb.Settings(code_dir="train_GenreClassifier.py")    # for code saving
    )

    if loaded_via_config:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_GenreClassifier.py")
        wandb.run.log_artifact(model_code)

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    print(config)
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    should_place_all_data_on_cuda = args.force_data_on_cuda and torch.cuda.is_available()
    training_dataset = Groove2Drum2BarDataset(
        dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=True,
        use_cached=True,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        augment_dataset=True,
        move_all_to_gpu=should_place_all_data_on_cuda
    )
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = Groove2Drum2BarDataset(
        dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=True,
        use_cached=True,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        augment_dataset=True,
        move_all_to_gpu=should_place_all_data_on_cuda
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    model_cpu = GenreClassifier(config)

    model_on_device = model_cpu.to(config.device)
    wandb.watch(model_on_device, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0

    # Create genre mapping
    print(training_dataset.get_genre_mapping())
    genre_mapping = {int(i.index(1)): genre for genre, i in training_dataset.get_genre_mapping().items()}
    print(genre_mapping)

    # Batch Data IO Extractor
    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        model_on_device.train()

        logger.info("***************************Training...")

        total_loss = []
        for batch_idx, batch_data in tqdm.tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()

            # Forward Pass
            # ------------------------------------------------------------------------------------------
            inputs_ = batch_data[5].to(config.device) if batch_data[5].device.type != config.device else batch_data[5]
            genre_tags = batch_data[4].to(config.device) if batch_data[4].device.type != config.device else batch_data[4]

            output = model_on_device(inputs_)
            loss = criterion(output, genre_tags)
            loss.mean().backward()
            optimizer.step()

            total_loss.append(loss.mean().item())

        wandb.log({"total_loss": sum(total_loss)/len(total_loss)}, commit=False)

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # Evaluate per genre
        # ------------------------------------------------------------------------------------------
        per_genre_loss = {i: [] for i in range(config.n_genres)}
        # ---------------------------------------------------------------------------------------------------

        model_on_device.eval()       # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(train_dataloader):
                inputs_ = batch_data[5].to(config.device) if batch_data[5].device.type != config.device else batch_data[5]
                genre_tags = batch_data[4].to(config.device) if batch_data[4].device.type != config.device else batch_data[4]

                preds = model_on_device.forward(inputs_)
                loss = criterion(preds, genre_tags)

                for i, pred in enumerate(preds):
                    per_genre_loss[genre_tags[i].item()].append(loss[i].item())

        for genre, loss_list in per_genre_loss.items():
            wandb.log({f"genre_{genre_mapping[genre]}_loss_train": sum(loss_list) / len(loss_list)}, commit=False)

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # Run the testing loop
        # ------------------------------------------------------------------------------------------
        per_genre_loss = {i: [] for i in range(config.n_genres)}
        model_on_device.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_dataloader):
                inputs_ = batch_data[5].to(config.device) if batch_data[5].device.type != config.device else batch_data[5]
                genre_tags = batch_data[4].to(config.device) if batch_data[4].device.type != config.device else batch_data[4]

                preds = model_on_device.forward(inputs_)
                loss = criterion(preds, genre_tags)

                for i, pred in enumerate(preds):
                    per_genre_loss[genre_tags[i].item()].append(loss[i].item())

        for genre, loss_list in per_genre_loss.items():
            wandb.log({f"genre_{genre_mapping[genre]}_loss_test": sum(loss_list) / len(loss_list)}, commit=False)

        # total loss
        total_loss_test = []
        for losses in per_genre_loss.values():
            total_loss_test.extend(losses)
        wandb.log({"total_loss_test": sum(total_loss_test) / len(total_loss_test)}, commit=False)

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        wandb.log({"epoch": epoch}, commit=True)

        # Save the model
        # ------------------------------------------------------------------------------------------
        if epoch % 5 == 0:
            model_artifact = wandb.Artifact(f'model_epoch_{epoch}', type='model')
            model_path = f"{args.save_model_dir}/{args.wandb_project}/{run_name}_{run_id}/{epoch}.pth"
            model_on_device.save(model_path)
            model_artifact.add_file(model_path)
            wandb_run.log_artifact(model_artifact)
            logger.info(f"Model saved to {model_path}")

wandb.finish()

