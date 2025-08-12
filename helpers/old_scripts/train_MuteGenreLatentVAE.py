import os

import wandb

import torch
from model import MuteGenreLatentVAE
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
                    default="MuteGenreLatentVAE")

# ----------------------- Model Parameters -----------------------
# d_model_dec_ratio denotes the ratio of the dec relative to enc size
parser.add_argument("--d_model_enc", type=int, help="Dimension of the encoder model", default=32)
parser.add_argument("--d_model_dec_ratio", type=int,help="Dimension of the decoder model as a ratio of d_model_enc", default=1)
parser.add_argument("--embedding_size_src", type=int, help="Dimension of the source embedding", default=3)
parser.add_argument("--embedding_size_tgt",  type=int, help="Dimension of the target embedding", default=27)
parser.add_argument("--nhead_enc", type=int, help="Number of attention heads for the encoder", default=2)
parser.add_argument("--nhead_dec", type=int, help="Number of attention heads for the decoder", default=2)
# d_ff_enc_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_enc_to_dmodel", type=float,
                    help="ratio of the dimension of enc feed-frwrd layer relative to enc dmodel", default=1)
# d_ff_dec_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_dec_to_dmodel", type=float,
                    help="ratio of the dimension of dec feed-frwrd layer relative to decoder dmodel", default=1)
# n_dec_lyrs_ratio denotes the ratio of the dec relative to n_enc_lyrs
parser.add_argument("--n_enc_lyrs", type=int, help="Number of encoder layers", default=3)
parser.add_argument("--n_dec_lyrs_ratio", type=float, help="Number of decoder layers as a ratio of "
                                               "n_enc_lyrs as a ratio of d_ff_enc", default=1)
parser.add_argument("--max_len_enc", type=int, help="Maximum length of the encoder", default=32)
parser.add_argument("--max_len_dec", type=int, help="Maximum length of the decoder", default=32)
parser.add_argument("--latent_dim", type=int, help="Overall Dimension of the latent space", default=16)
parser.add_argument("--n_genres", type=int, help="Number of genres", default=9)

# ----------------------- Loss Parameters -----------------------
parser.add_argument("--beta_annealing_per_cycle_rising_ratio", type=float, help="rising ratio in each cycle to anneal beta", default=1)
parser.add_argument("--beta_annealing_per_cycle_period", type=int, help="Number of epochs for each cycle of Beta annealing", default=100)
parser.add_argument("--beta_annealing_start_first_rise_at_epoch", type=int, help="Warm up epochs (KL = 0) before starting the first cycle ", default=0)

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
parser.add_argument("--beta_annealing_activated", help="Use cyclical annealing on KL beta term", type=int,
                    default=1)
parser.add_argument("--beta_level", type=float, help="Max level of beta term on KL", default=0.2)

parser.add_argument("--scale_h_loss", type=float, help="Scale for hit loss", default=1)
parser.add_argument("--scale_v_loss", type=float, help="Scale for velocity loss", default=1)
parser.add_argument("--scale_o_loss", type=float, help="Scale for offset loss", default=1)

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

# ----------------------- Evaluation Params -----------------------
parser.add_argument("--calculate_hit_scores_on_train", type=bool,
                    help="Evaluates the quality of the hit models on training set",
                    default=True)
parser.add_argument("--calculate_hit_scores_on_test", type=bool,
                    help="Evaluates the quality of the hit models on test/evaluation set",
                    default=True)
parser.add_argument("--piano_roll_samples", type=bool, help="Generate audio samples", default=True)
parser.add_argument("--piano_roll_frequency", type=int, help="Frequency of piano roll generatio", default=10)
parser.add_argument("--hit_score_frequency", type=int, help="Frequency of hit score generatio", default=5)

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/VAE")
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
    d_model_dec = int(float(args.d_model_enc) * float(args.d_model_dec_ratio))
    dim_feedforward_enc = int(float(args.d_ff_enc_to_dmodel)*float(args.d_model_enc))
    dim_feedforward_dec = int(float(args.d_ff_dec_to_dmodel) * d_model_dec)
    num_decoder_layers = int(float(args.n_enc_lyrs) * float(args.n_dec_lyrs_ratio))
    hparams = dict(
        d_model_enc=args.d_model_enc,
        d_model_dec=d_model_dec,
        dim_feedforward_enc=dim_feedforward_enc,
        dim_feedforward_dec=dim_feedforward_dec,
        num_encoder_layers=int(args.n_enc_lyrs),
        num_decoder_layers=num_decoder_layers,
        embedding_size_src=args.embedding_size_src,
        embedding_size_tgt=args.embedding_size_tgt,
        nhead_enc=args.nhead_enc,
        nhead_dec=args.nhead_dec,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        max_len_enc=args.max_len_enc,
        max_len_dec=args.max_len_dec,
        n_genres=args.n_genres,
        velocity_dropout=args.velocity_dropout,
        offset_dropout=args.offset_dropout,
        beta_annealing_per_cycle_rising_ratio=float(args.beta_annealing_per_cycle_rising_ratio),
        beta_annealing_per_cycle_period=args.beta_annealing_per_cycle_period,
        beta_annealing_start_first_rise_at_epoch=args.beta_annealing_start_first_rise_at_epoch,
        beta_annealing_activated=True if args.beta_annealing_activated == 1 else False,
        beta_level=float(args.beta_level),
        scale_h_loss=args.scale_h_loss,
        scale_v_loss=args.scale_v_loss,
        scale_o_loss=args.scale_o_loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
        settings=wandb.Settings(code_dir="train_MuteGenreLatentVAE.py")    # for code saving
    )

    if loaded_via_config:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_MuteGenreLatentVAE.py")
        wandb.run.log_artifact(model_code)

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    print(config)
    run_name = wandb_run.name
    run_id = wandb_run.id
    collapse_tapped_sequence = (args.embedding_size_src == 3)
    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    should_place_all_data_on_cuda = args.force_data_on_cuda and torch.cuda.is_available()
    training_dataset = Groove2Drum2BarDataset(
        dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
        subset_tag="train",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        use_cached=True,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        augment_dataset=True,
        move_all_to_gpu=should_place_all_data_on_cuda
    )
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = Groove2Drum2BarDataset(
        dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
        subset_tag="test",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        use_cached=True,
        down_sampled_ratio=0.1 if args.is_testing is True else None,
        augment_dataset=True,
        move_all_to_gpu=should_place_all_data_on_cuda
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    model_cpu = MuteGenreLatentVAE(config)

    model_on_device = model_cpu.to(config.device)
    wandb.watch(model_on_device, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------

    hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    velocity_loss_fn = torch.nn.MSELoss(reduction='none')
    offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0

    beta_np_cyc = train_utils.generate_beta_curve(
        n_epochs=config.epochs,
        period_epochs=config.beta_annealing_per_cycle_period,
        rise_ratio=config.beta_annealing_per_cycle_rising_ratio,
        start_first_rise_at_epoch=config.beta_annealing_start_first_rise_at_epoch)

    # Batch Data IO Extractor
    def batch_data_extractor(data_, device=config.device):
        # Extract the data from the batch
        inputs_ = data_[0].to(device) if data_[0].device.type != device else data_[0]
        genre_tags = data_[4].to(device) if data_[4].device.type != device else data_[4]
        outputs_ = data_[5].to(device) if data_[5].device.type != device else data_[5]
        kick_is_muted = data_[8].to(device) if data_[8].device.type != device else data_[8]
        snare_is_muted = data_[9].to(device) if data_[9].device.type != device else data_[9]
        hat_is_muted = data_[10].to(device) if data_[10].device.type != device else data_[10]
        tom_is_muted = data_[11].to(device) if data_[11].device.type != device else data_[11]
        cymbal_is_muted = data_[12].to(device) if data_[12].device.type != device else data_[12]
        return inputs_, genre_tags, outputs_, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted

    def predict_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        flat_hvo_groove, genre_tags, outputs_, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted = batch_data_extractor(batch_data, device)
        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=flat_hvo_groove,
                genre_tags=genre_tags,
                kick_is_muted=kick_is_muted,
                snare_is_muted=snare_is_muted,
                hat_is_muted=hat_is_muted,
                tom_is_muted=tom_is_muted,
                cymbal_is_muted=cymbal_is_muted)
        return hvo, latent_z

    def forward_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        flat_hvo_groove, genre_tags, target_outputs, kick_is_muted, snare_is_muted, hat_is_muted, tom_is_muted, cymbal_is_muted = batch_data_extractor(batch_data, device)
        h_logits, v_logits, o_logits, mu, log_var, latent_z = model_.forward(
            flat_hvo_groove=flat_hvo_groove, 
            genre_tags=genre_tags,
            kick_is_muted=kick_is_muted,
            snare_is_muted=snare_is_muted,
            hat_is_muted=hat_is_muted,
            tom_is_muted=tom_is_muted,
            cymbal_is_muted=cymbal_is_muted)
        return h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs
        
    previous_loaded_dataset_for_umap_train = None
    previous_loaded_dataset_for_umap_test = None
    previous_evaluator_for_piano_rolls = None
    previous_evaluator_for_hit_scores_train = None
    previous_evaluator_for_hit_scores_test = None

    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        model_on_device.train()

        logger.info("***************************Training...")

        kl_beta = beta_np_cyc[epoch] * config.beta_level if config.beta_annealing_activated else config.beta_level
        train_log_metrics, step_ = train_utils.train_loop(
            train_dataloader=train_dataloader,
            forward_method=forward_using_batch_data,
            optimizer=optimizer,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            starting_step=step_,
            kl_beta=kl_beta,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss
        )

        wandb.log(train_log_metrics, commit=False)
        wandb.log({"kl_beta": kl_beta}, commit=False)

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        model_on_device.eval()       # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)

        logger.info("***************************Testing...")

        test_log_metrics = train_utils.test_loop(
            test_dataloader=test_dataloader,
            forward_method=forward_using_batch_data,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            kl_beta=kl_beta,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss
        )

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        wandb.log(test_log_metrics, commit=False)
        logger.info(f"Epoch {epoch} Finished with total train loss of {train_log_metrics['Loss_Criteria/loss_total_rec_w_kl_train']} "
                    f"and test loss of {test_log_metrics['Loss_Criteria/loss_total_rec_w_kl_test']}")

        # Generate PianoRolls and UMAP Plots  and KL/OA PLots if Needed
        # ---------------------------------------------------------------------------------------------------
        if args.piano_roll_samples:
            if epoch % args.piano_roll_frequency == 0:
                logger.info("________Generating PianoRolls...")
                media, previous_evaluator_for_piano_rolls = eval_utils.get_pianoroll_for_wandb(
                    model=model_on_device,
                    predict_using_batch_data=predict_using_batch_data,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='test',
                    down_sampled_ratio=0.02,
                    cached_folder="cached/GrooveEvaluator/templates",
                    divide_by_genre=True,
                    previous_evaluator=previous_evaluator_for_piano_rolls,
                    need_piano_roll=True,
                    need_kl_plot=False,
                    need_audio=False
                )
                wandb.log(media, commit=False)

                # umap
                logger.info("________Generating UMAP...")
                media, previous_loaded_dataset_for_umap_test = eval_utils.generate_umap_for_wandb(
                    predict_using_batch_data=predict_using_batch_data,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='test',
                    previous_loaded_dataset=previous_loaded_dataset_for_umap_test,
                    down_sampled_ratio=0.5,
                )
                if media is not None:
                    wandb.log(media, commit=False)

                # media, previous_loaded_dataset_for_umap_train = eval_utils.generate_umap_for_wandb(
                #     predict_using_batch_data=predict_using_batch_data,
                #     dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                #     subset_name='train',
                #     previous_loaded_dataset=previous_loaded_dataset_for_umap_train,
                #     down_sampled_ratio=0.3,
                # )
                # if media is not None:
                #     wandb.log(media, commit=False)

        # Get Hit Scores for the entire train and the entire test set
        # ---------------------------------------------------------------------------------------------------
        if args.calculate_hit_scores_on_train:
            if epoch % args.hit_score_frequency == 0:
                logger.info("________Calculating Hit Scores on Train Set...")
                train_set_hit_scores, previous_evaluator_for_hit_scores_train = eval_utils.get_hit_scores(
                    predict_using_batch_data=predict_using_batch_data,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='train',
                    down_sampled_ratio=None,
                    cached_folder="cached/GrooveEvaluator/templates",
                    previous_evaluator=previous_evaluator_for_hit_scores_train,
                    divide_by_genre=False
                )
                wandb.log(train_set_hit_scores, commit=False)

        if args.calculate_hit_scores_on_test:
            logger.info("________Calculating Hit Scores on Test Set...")
            test_set_hit_scores, previous_evaluator_for_hit_scores_test = eval_utils.get_hit_scores(
                predict_using_batch_data=predict_using_batch_data,
                dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                subset_name=args.evaluate_on_subset,
                down_sampled_ratio=None,
                cached_folder="cached/GrooveEvaluator/templates",
                previous_evaluator=previous_evaluator_for_hit_scores_test,
                divide_by_genre=False,

            )
            wandb.log(test_set_hit_scores, commit=False)

        # Commit the metrics to wandb
        # ---------------------------------------------------------------------------------------------------
        wandb.log({"epoch": epoch}, step=epoch)

        # Save the model if needed
        # ---------------------------------------------------------------------------------------------------
        if args.save_model:
            if epoch % args.save_model_frequency == 0 and epoch > 0:
                if epoch < 10:
                    ep_ = f"00{epoch}"
                elif epoch < 100:
                    ep_ = f"0{epoch}"
                else:
                    ep_ = epoch
                model_artifact = wandb.Artifact(f'model_epoch_{ep_}', type='model')
                model_path = f"{args.save_model_dir}/{args.wandb_project}/{run_name}_{run_id}/{ep_}.pth"
                model_on_device.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

    wandb.finish()

