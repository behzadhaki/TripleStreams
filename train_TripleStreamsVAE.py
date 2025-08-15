import os
import wandb
import torch
from model import TripleStreamsVAE
from helpers import train_utils
from helpers import eval_utils_TripleStreams as eval_utils
from data.src.dataLoaders import get_triplestream_dataset
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse

logger = getLogger("")
logger.setLevel(DEBUG)

parser = argparse.ArgumentParser()

# ----------------------- Set True When Testing ----------------
parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", type=bool,
                    default=False)

# ----------------------- WANDB Settings -----------------------
parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
parser.add_argument("--config",
                    help="Yaml file for configuration. If available, the rest of the arguments will be ignored",
                    default=None)
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name", default="TripleStreamsVAE")

# ----------------------- Checkpoint Resume Parameters -----------------------
parser.add_argument("--resume_from_checkpoint", type=bool, help="Resume training from checkpoint", default=False)
parser.add_argument("--checkpoint_wandb_run_id", type=str, help="WandB run ID to resume from", default=None)
parser.add_argument("--checkpoint_step", type=int, help="Step number to resume from (optional)", default=None)
parser.add_argument("--checkpoint_artifact_name", type=str, help="Name of model artifact (e.g., model_step_10000)", default=None)
parser.add_argument("--reset_optimizer", type=bool, help="Reset optimizer state when resuming", default=False)
parser.add_argument("--reset_beta_scheduler", type=bool, help="Reset beta scheduler when resuming", default=False)

# ----------------------- Model Parameters -----------------------
parser.add_argument("--d_model_enc", type=int, help="Dimension of the encoder model", default=64)
parser.add_argument("--d_model_dec", type=int, help="Dimension of the decoder model", default=128)
parser.add_argument("--embedding_size_src", type=int, help="Dimension of the source embedding", default=3)
parser.add_argument("--embedding_size_tgt", type=int, help="Dimension of the target embedding", default=9)
parser.add_argument("--nhead_enc", type=int, help="Number of attention heads for the encoder", default=8)
parser.add_argument("--nhead_dec", type=int, help="Number of attention heads for the decoder", default=8)
parser.add_argument("--dim_feedforward_enc", type=int, help="Dimension of encoder feedforward layer", default=128)
parser.add_argument("--dim_feedforward_dec", type=int, help="Dimension of decoder feedforward layer", default=512)
parser.add_argument("--num_encoder_layers", type=int, help="Number of encoder layers", default=7)
parser.add_argument("--num_decoder_layers", type=int, help="Number of decoder layers", default=12)
parser.add_argument("--max_len", type=int, help="Maximum sequence length", default=32)
parser.add_argument("--latent_dim", type=int, help="Overall Dimension of the latent space", default=128)

# ---------------------- Control Parameters -----------------------
parser.add_argument("--prepend_control_tokens", type=bool, help="Prepend controls rather than summing", default=True)
parser.add_argument("--encoding_control1_key", type=str, help="control 1 applied to encoder",
                    default="Flat Out Vs. Input | Hits | Hamming")
parser.add_argument("--encoding_control2_key", type=str, help="control 2 applied to encoder",
                    default="Flat Out Vs. Input | Accent | Hamming")
parser.add_argument("--decoding_control1_key", type=str, help="control 1 applied to decoder",
                    default="Stream 1 Vs. Flat Out | Hits | Hamming")
parser.add_argument("--decoding_control2_key", type=str, help="control 2 applied to decoder",
                    default="Stream 2 Vs. Flat Out | Hits | Hamming")
parser.add_argument("--decoding_control3_key", type=str, help="control 3 applied to decoder",
                    default="Stream 3 Vs. Flat Out | Hits | Hamming")
parser.add_argument("--n_encoding_control1_tokens", type=int, help="Number of tokens", default=33)
parser.add_argument("--n_encoding_control2_tokens", type=int, help="Number of tokens", default=10)
parser.add_argument("--n_decoding_control1_tokens", type=int, help="Number of tokens", default=10)
parser.add_argument("--n_decoding_control2_tokens", type=int, help="Number of tokens", default=10)
parser.add_argument("--n_decoding_control3_tokens", type=int, help="Number of tokens", default=10)

# ----------------------- Step-Based Beta Annealing Parameters -----------------------
parser.add_argument("--beta_annealing_period_steps", type=int, default=25000,
                    help="Number of steps for each cycle of Beta annealing")
parser.add_argument("--beta_annealing_start_first_rise_at_step", type=int, default=5000,
                    help="Warm up steps before starting the first cycle")
parser.add_argument("--beta_annealing_per_cycle_rising_ratio", type=float,
                    help="rising ratio in each cycle to anneal beta", default=0.5)
parser.add_argument("--beta_annealing_activated", help="Use cyclical annealing on KL beta term", type=bool,
                    default=True)
parser.add_argument("--beta_level", type=float, help="Max level of beta term on KL", default=0.2)

# ----------------------- Step-Based Logging Parameters -----------------------
parser.add_argument("--step_log_frequency", type=int, default=50, help="Log metrics to wandb every N steps")
parser.add_argument("--step_hit_score_frequency", type=int, default=2000, help="Calculate hit scores every N steps")
parser.add_argument("--step_piano_roll_frequency", type=int, default=1500, help="Generate piano rolls every N steps")

# ----------------------- Training Parameters -----------------------
parser.add_argument("--dropout", type=float, help="Dropout", default=0.1)
parser.add_argument("--velocity_dropout", type=float, help="velocity_dropout", default=0)
parser.add_argument("--offset_dropout", type=float, help="offset_dropout", default=0)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1000)
parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
parser.add_argument("--lr", type=float, help="Learning rate", default=0.0006369948608989782)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam'", default="adam",
                    choices=['sgd', 'adam'])

parser.add_argument("--scale_h_loss", type=float, help="Scale for hit loss", default=1)
parser.add_argument("--scale_v_loss", type=float, help="Scale for velocity loss", default=1)
parser.add_argument("--scale_o_loss", type=float, help="Scale for offset loss", default=1)

# ----------------------- GPU Settings --------------------------
parser.add_argument("--device", type=str, help="Device to use for training - either 'cuda', 'cpu' or 'mps'",
                    required=True)
parser.add_argument("--move_all_to_cuda", type=bool, help="places all training data on cuda", default=True)

# ----------------------- Data Parameters -----------------------
parser.add_argument("--dataset_root_path", type=str, help="Root path for dataset files",
                    default="data/triple_streams/model_ready/AccentAt0.75/")
parser.add_argument("--dataset_files", nargs='+', help="List of dataset files",
                    default=["01_candombe_four_voices.pkl.bz2"])
parser.add_argument("--evaluate_on_subset", type=str, help="Using test or evaluation subset for evaluating the model",
                    default="test", choices=['test', 'evaluation'])

# ----------------------- Evaluation Params -----------------------
parser.add_argument("--calculate_hit_scores_on_train", type=bool,
                    help="Evaluates the quality of the hit models on training set", default=True)
parser.add_argument("--calculate_hit_scores_on_test", type=bool,
                    help="Evaluates the quality of the hit models on test/evaluation set", default=True)
parser.add_argument("--piano_roll_samples", type=bool, help="Generate piano roll samples", default=True)

# ----------------------- Model Saving Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/TripleStreamsVAE")
parser.add_argument("--save_model_frequency_steps", type=int, help="Save model every n steps", default=10000)

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

# Load configuration
loaded_via_config = False
if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
        if "wandb_project" not in hparams.keys():
            hparams["wandb_project"] = args.wandb_project
        if "device" in hparams.keys():
            logger.warning(f"\n\nRemove device from config file. Using CLI argument instead: {args.device}\n\n")
        hparams["device"] = args.device  # Always use CLI device argument
        loaded_via_config = True
else:
    hparams = dict(
        # Model architecture
        d_model_enc=args.d_model_enc,
        d_model_dec=args.d_model_dec,
        dim_feedforward_enc=args.dim_feedforward_enc,
        dim_feedforward_dec=args.dim_feedforward_dec,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        embedding_size_src=args.embedding_size_src,
        embedding_size_tgt=args.embedding_size_tgt,
        nhead_enc=args.nhead_enc,
        nhead_dec=args.nhead_dec,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        max_len_enc=args.max_len,
        max_len_dec=args.max_len,

        # Control tokens
        prepend_control_tokens=args.prepend_control_tokens,
        encoding_control1_key=args.encoding_control1_key,
        encoding_control2_key=args.encoding_control2_key,
        decoding_control1_key=args.decoding_control1_key,
        decoding_control2_key=args.decoding_control2_key,
        decoding_control3_key=args.decoding_control3_key,
        n_encoding_control1_tokens=args.n_encoding_control1_tokens,
        n_encoding_control2_tokens=args.n_encoding_control2_tokens,
        n_decoding_control1_tokens=args.n_decoding_control1_tokens,
        n_decoding_control2_tokens=args.n_decoding_control2_tokens,
        n_decoding_control3_tokens=args.n_decoding_control3_tokens,
        velocity_dropout=args.velocity_dropout,
        offset_dropout=args.offset_dropout,

        # Step-based beta annealing
        beta_annealing_period_steps=args.beta_annealing_period_steps,
        beta_annealing_start_first_rise_at_step=args.beta_annealing_start_first_rise_at_step,
        beta_annealing_per_cycle_rising_ratio=args.beta_annealing_per_cycle_rising_ratio,
        beta_annealing_activated=args.beta_annealing_activated,
        beta_level=args.beta_level,

        # Step-based logging
        step_log_frequency=args.step_log_frequency,
        step_hit_score_frequency=args.step_hit_score_frequency,
        step_piano_roll_frequency=args.step_piano_roll_frequency,

        # Loss scaling
        scale_h_loss=args.scale_h_loss,
        scale_v_loss=args.scale_v_loss,
        scale_o_loss=args.scale_o_loss,

        # Training params
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        device=args.device,

        # Data parameters
        dataset_root_path=args.dataset_root_path,
        dataset_files=args.dataset_files,

        # Model saving
        save_model_frequency_steps=args.save_model_frequency_steps,

        # Checkpoint resume parameters
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_wandb_run_id=args.checkpoint_wandb_run_id,
        checkpoint_step=args.checkpoint_step,
        checkpoint_artifact_name=args.checkpoint_artifact_name,
        reset_optimizer=args.reset_optimizer,
        reset_beta_scheduler=args.reset_beta_scheduler
    )

# Device availability checks
if args.device == 'mps' and not torch.has_mps:
    logger.warning("\n\n MPS is not available. Falling back to CPU.")
    hparams["device"] = 'cpu'

if args.device == 'cuda' and not torch.cuda.is_available():
    logger.warning("\n\n CUDA is not available. Falling back to CPU.")
    hparams["device"] = 'cpu'

is_testing = hparams.get("is_testing", False) or args.is_testing

# Print configuration
print("\n\n|" + "=" * 80 + "|")
print(f"\n\tHyperparameters for the run:")
print("\n|" + "=" * 80 + "|\n\n")
for key, value in hparams.items():
    print(f"\t{key}: {value}")
print("\n\n|" + "=" * 80 + "|")
if loaded_via_config:
    print(f"Loaded via config file: {args.config}")
print("|" + "=" * 80 + "|\n\n\n")

if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"

if __name__ == "__main__":

    # Initialize wandb
    wandb_run = wandb.init(
        config=hparams,
        project=hparams["wandb_project"],
        entity="behzadhaki",
        settings=wandb.Settings(code_dir="train_TripleStreamsVAE.py")
    )

    if loaded_via_config:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_TripleStreamsVAE.py")
        wandb.run.log_artifact(model_code)

    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets
    training_dataset = get_triplestream_dataset(
        config=config,
        subset_tag="train",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
        move_all_to_cuda=args.move_all_to_cuda
    )

    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = get_triplestream_dataset(
        config=config,
        subset_tag="test",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    model_cpu = TripleStreamsVAE(config)
    model_on_device = model_cpu.to(config.device)
    wandb.watch(model_on_device, log="all", log_freq=1)

    # Instantiate loss functions and optimizer
    hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    velocity_loss_fn = torch.nn.MSELoss(reduction='none')
    offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Setup step-based beta annealing
    if config.beta_annealing_activated:
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * config.epochs

        beta_scheduler = train_utils.BetaAnnealingScheduler(
            total_steps=total_steps,
            period_steps=config.beta_annealing_period_steps,
            rise_ratio=config.beta_annealing_per_cycle_rising_ratio,
            start_first_rise_at_step=config.beta_annealing_start_first_rise_at_step,
            beta_level=config.beta_level
        )
        logger.info(f"Using step-based beta annealing with {total_steps} total steps")
    else:
        beta_scheduler = None
        logger.info("Beta annealing disabled")

    # Setup resumable training (Option 3: Simple approach)
    starting_step, model_on_device, optimizer, beta_scheduler = train_utils.setup_resumable_training(
        config, model_on_device, optimizer, beta_scheduler
    )

    # Batch Data IO Extractor
    def batch_data_extractor(data_, device=config.device):
        input_grooves = data_[0].to(device) if data_[0].device.type != device else data_[0]
        output_streams = data_[1].to(device) if data_[1].device.type != device else data_[1]
        encoding_control1_tokens = data_[2].to(device) if data_[2].device.type != device else data_[2]
        encoding_control2_tokens = data_[3].to(device) if data_[3].device.type != device else data_[3]
        decoding_control1_tokens = data_[4].to(device) if data_[4].device.type != device else data_[4]
        decoding_control2_tokens = data_[5].to(device) if data_[5].device.type != device else data_[5]
        decoding_control3_tokens = data_[6].to(device) if data_[6].device.type != device else data_[6]
        metadata = data_[7]
        indices = data_[8]

        return (input_grooves, output_streams, encoding_control1_tokens, encoding_control2_tokens,
                decoding_control1_tokens, decoding_control2_tokens, decoding_control3_tokens, metadata, indices)


    def predict_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        (input_grooves, output_streams, encoding_control1_tokens, encoding_control2_tokens,
         decoding_control1_tokens, decoding_control2_tokens, decoding_control3_tokens, metadata,
         indices) = batch_data_extractor(batch_data, device)

        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=input_grooves,
                encoding_control1_token=encoding_control1_tokens,
                encoding_control2_token=encoding_control2_tokens,
                decoding_control1_token=decoding_control1_tokens,
                decoding_control2_token=decoding_control2_tokens,
                decoding_control3_token=decoding_control3_tokens)

        return hvo, latent_z


    def forward_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        (input_grooves, target_output_streams, encoding_control1_tokens, encoding_control2_tokens,
         decoding_control1_tokens, decoding_control2_tokens, decoding_control3_tokens, metadata,
         indices) = batch_data_extractor(batch_data, device)

        h_logits, v_logits, o_logits, mu, log_var, latent_z = model_.forward(
            flat_hvo_groove=input_grooves,
            encoding_control1_token=encoding_control1_tokens,
            encoding_control2_token=encoding_control2_tokens,
            decoding_control1_token=decoding_control1_tokens,
            decoding_control2_token=decoding_control2_tokens,
            decoding_control3_token=decoding_control3_tokens)

        return h_logits, v_logits, o_logits, mu, log_var, latent_z, target_output_streams


    # Setup evaluation callbacks for step-based evaluation
    def quick_hit_scores_train(step):
        """Quick hit score evaluation for training set"""
        logger.info(f"Step {step}: Calculating Hit Scores on Train Set")
        train_set_hit_scores, _ = eval_utils.get_hit_scores(
            config=config,
            subset_tag='train',
            use_cached=True,
            downsampled_size=1000,
            predict_using_batch_data_method=predict_using_batch_data,
            divide_by_collection=False,
            previous_evaluator=None
        )
        return {f"Quick_Eval/{k}": v for k, v in train_set_hit_scores.items()}


    def quick_hit_scores_test(step):
        """Quick hit score evaluation for test set"""
        logger.info(f"Step {step}: Calculating Hit Scores on Test Set")
        test_set_hit_scores, _ = eval_utils.get_hit_scores(
            config=config,
            subset_tag='test',
            use_cached=True,
            downsampled_size=1000,
            predict_using_batch_data_method=predict_using_batch_data,
            divide_by_collection=False,
            previous_evaluator=None
        )
        return {f"Quick_Eval/{k}": v for k, v in test_set_hit_scores.items()}


    def piano_roll_generation(step):
        """Generate piano rolls for visualization"""
        logger.info(f"Step {step}: Generating Piano Rolls")
        media, _ = eval_utils.get_pianoroll_for_wandb(
            config=config,
            subset_tag='test',
            use_cached=True,
            downsampled_size=50,
            predict_using_batch_data_method=predict_using_batch_data,
            tag_key="collection",
            cached_folder="cached/GrooveEvaluator/templates/PRolls",
            divide_by_collection=True,
            previous_evaluator=None,
            need_piano_roll=True,
            need_kl_plot=False,
            need_audio=False
        )
        return media


    def save_model_checkpoint(step):
        """Save model checkpoint with enhanced state"""
        return train_utils.save_model_checkpoint_enhanced(
            model_on_device, optimizer, beta_scheduler, step,
            args.save_model_dir, config.wandb_project, run_name, run_id
        )


    # Define evaluation callbacks
    eval_callbacks = {}

    if args.calculate_hit_scores_on_train:
        eval_callbacks['hit_scores_train'] = {
            'function': quick_hit_scores_train,
            'frequency': config.step_hit_score_frequency
        }

    if args.calculate_hit_scores_on_test:
        eval_callbacks['hit_scores_test'] = {
            'function': quick_hit_scores_test,
            'frequency': config.step_hit_score_frequency
        }

    if args.piano_roll_samples:
        eval_callbacks['piano_rolls'] = {
            'function': piano_roll_generation,
            'frequency': config.step_piano_roll_frequency
        }

    if args.save_model:
        eval_callbacks['save_model'] = {
            'function': save_model_checkpoint,
            'frequency': config.save_model_frequency_steps
        }

    # Training loop
    step_ = starting_step

    for epoch in range(config.epochs):
        print("\n\n|" + "=" * 50 + "|")
        print(f"\t\tEpoch {epoch} of {config.epochs}, steps so far {step_}")
        if starting_step > 0:
            print(f"\t\tResumed from step {starting_step}")
        print("|" + "=" * 50 + "|")

        # Training phase
        model_on_device.train()
        logger.info("\n***************************Training...")

        train_log_metrics, step_ = train_utils.train_loop_step_based(
            train_dataloader=train_dataloader,
            forward_method=forward_using_batch_data,
            optimizer=optimizer,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            starting_step=step_,
            beta_scheduler=beta_scheduler,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss,
            log_frequency=config.step_log_frequency,
            eval_callbacks=eval_callbacks
        )

        wandb.log(train_log_metrics, commit=False)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # Testing phase
        model_on_device.eval()
        logger.info("\n***************************Testing...")

        test_log_metrics, _ = train_utils.test_loop_step_based(
            test_dataloader=test_dataloader,
            forward_method=forward_using_batch_data,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            starting_step=step_,
            beta_scheduler=beta_scheduler,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss,
            log_frequency=config.step_log_frequency
        )

        wandb.log(test_log_metrics, commit=False)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

        logger.info(
            f"Epoch {epoch} Finished with total train loss of {train_log_metrics.get('Train_Epoch_Metrics/loss_total_rec_w_kl', 'N/A')} "
            f"and test loss of {test_log_metrics.get('Test_Epoch_Metrics/loss_total_rec_w_kl', 'N/A')}")

        wandb.log({"epoch": epoch}, step=step_)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

    wandb.finish()