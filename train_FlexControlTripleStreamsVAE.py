import os
import wandb
import torch
from model import FlexControlTripleStreamsVAE
from helpers import train_utils
from helpers import eval_utils
from data.src.dataLoaders import get_flexcontrol_triplestream_dataset
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
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name", default="FlexControlTripleStreamsVAE")

# ----------------------- Checkpoint Resume Parameters -----------------------
parser.add_argument("--resume_from_checkpoint", type=bool, help="Resume training from checkpoint", default=False)
parser.add_argument("--checkpoint_wandb_run_id", type=str, help="WandB run ID to resume from", default=None)
parser.add_argument("--checkpoint_step", type=int, help="Step number to resume from (optional)", default=None)
parser.add_argument("--checkpoint_artifact_name", type=str, help="Name of model artifact (e.g., model_step_10000)",
                    default=None)
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

# ---------------------- Flexible Control Parameters -----------------------
parser.add_argument("--n_encoding_control_tokens", nargs='+', type=int,
                    help="Number of tokens for each encoding control",
                    default=[33, 10])
parser.add_argument("--encoding_control_modes", nargs='+', type=str,
                    help="Mode for each encoding control ('prepend' or 'add')",
                    default=['prepend', 'add'])
parser.add_argument("--encoding_control_keys", nargs='+', type=str, help="Keys for encoding controls",
                    default=["Flat Out Vs. Input | Hits | Hamming", "Flat Out Vs. Input | Accent | Hamming"])
parser.add_argument("--n_decoding_control_tokens", nargs='+', type=int,
                    help="Number of tokens for each decoding control",
                    default=[10, 10, 10])
parser.add_argument("--decoding_control_modes", nargs='+', type=str,
                    help="Mode for each decoding control ('prepend' or 'add')",
                    default=['prepend', 'prepend', 'prepend'])
parser.add_argument("--decoding_control_keys", nargs='+', type=str, help="Keys for decoding controls",
                    default=["Stream 1 Vs. Flat Out | Hits | Hamming", "Stream 2 Vs. Flat Out | Hits | Hamming",
                             "Stream 3 Vs. Flat Out | Hits | Hamming"])

# ----------------------- Epoch-Percentage-Based Beta Annealing Parameters -----------------------
parser.add_argument("--beta_annealing_period_epoch_pct", type=float, default=100.0,
                    help="Beta annealing period as percentage of epoch (100.0 = 1 full epoch per cycle)")
parser.add_argument("--beta_annealing_start_first_rise_at_epoch_pct", type=float, default=20.0,
                    help="Start first beta rise at this epoch percentage (20.0 = after 20% of first epoch)")
parser.add_argument("--beta_annealing_per_cycle_rising_ratio", type=float,
                    help="rising ratio in each cycle to anneal beta", default=0.5)
parser.add_argument("--beta_annealing_gap_ratio", type=float, default=0.0,
                    help="Gap ratio at end of each cycle (0.0-1.0)")
parser.add_argument("--beta_annealing_activated", help="Use cyclical annealing on KL beta term", type=bool,
                    default=True)
parser.add_argument("--beta_level", type=float, help="Max level of beta term on KL", default=0.2)

# ----------------------- Epoch-Percentage-Based Logging Parameters -----------------------
parser.add_argument("--step_log_frequency_epoch_pct", type=float, default=1.0,
                    help="Log metrics to wandb every N%% of epoch (1.0 = every 1%% of epoch)")
parser.add_argument("--step_hit_score_frequency_epoch_pct", type=float, default=50.0,
                    help="Calculate hit scores every N%% of epoch (50.0 = every 50%% of epoch)")
parser.add_argument("--step_piano_roll_frequency_epoch_pct", type=float, default=200.0,
                    help="Generate piano rolls every N%% of epoch (200.0 = every 2 epochs)")
parser.add_argument("--save_model_frequency_epoch_pct", type=float, default=500.0,
                    help="Save model every N%% of epoch (500.0 = every 5 epochs)")

# ----------------------- Training Control -----------------------
parser.add_argument("--start_shuffle_on_epoch", type=float, default=0.0,
                    help="Start shuffling dataloader from this epoch (0.0 = shuffle from start, 5.5 = start at epoch 5.5)")

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
parser.add_argument("--save_model_dir", type=str, help="Path to save the model",
                    default="misc/FlexControlTripleStreamsVAE")

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")


def convert_epoch_percentages_to_steps(config, steps_per_epoch):
    """
    Convert all epoch percentage parameters to step-based parameters

    Args:
        config: Configuration object with epoch percentage parameters
        steps_per_epoch: Number of steps per epoch

    Returns:
        dict: Step-based parameters for backward compatibility
    """

    def epoch_pct_to_steps(epoch_pct, min_steps=1):
        """Convert epoch percentage to steps"""
        return max(min_steps, int(steps_per_epoch * epoch_pct / 100.0))

    # Convert beta annealing parameters
    beta_params = {
        'beta_annealing_period_steps': epoch_pct_to_steps(
            getattr(config, 'beta_annealing_period_epoch_pct', 100.0)
        ),
        'beta_annealing_start_first_rise_at_step': epoch_pct_to_steps(
            getattr(config, 'beta_annealing_start_first_rise_at_epoch_pct', 20.0)
        ),
    }

    # Convert logging frequencies
    logging_params = {
        'step_log_frequency': epoch_pct_to_steps(
            getattr(config, 'step_log_frequency_epoch_pct', 1.0)
        ),
        'step_hit_score_frequency': epoch_pct_to_steps(
            getattr(config, 'step_hit_score_frequency_epoch_pct', 50.0)
        ),
        'step_piano_roll_frequency': epoch_pct_to_steps(
            getattr(config, 'step_piano_roll_frequency_epoch_pct', 200.0)
        ),
        'save_model_frequency_steps': epoch_pct_to_steps(
            getattr(config, 'save_model_frequency_epoch_pct', 500.0)
        ),
    }

    # Convert shuffle epoch to step
    shuffle_params = {
        'start_shuffle_on_step': epoch_pct_to_steps(
            getattr(config, 'start_shuffle_on_epoch', 0.0) * 100.0,  # Convert epoch to epoch_pct
            min_steps=0
        ),
    }

    # Print conversion summary
    print(f"\n{'=' * 80}")
    print(f"EPOCH PERCENTAGE TO STEPS CONVERSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"\nBeta Annealing Parameters:")
    print(
        f"  Period: {getattr(config, 'beta_annealing_period_epoch_pct', 100.0):.1f}% epoch → {beta_params['beta_annealing_period_steps']} steps")
    print(
        f"  Start rise: {getattr(config, 'beta_annealing_start_first_rise_at_epoch_pct', 20.0):.1f}% epoch → {beta_params['beta_annealing_start_first_rise_at_step']} steps")

    print(f"\nLogging Frequencies:")
    print(
        f"  Log: {getattr(config, 'step_log_frequency_epoch_pct', 1.0):.1f}% epoch → every {logging_params['step_log_frequency']} steps")
    print(
        f"  Hit scores: {getattr(config, 'step_hit_score_frequency_epoch_pct', 50.0):.1f}% epoch → every {logging_params['step_hit_score_frequency']} steps")
    print(
        f"  Piano rolls: {getattr(config, 'step_piano_roll_frequency_epoch_pct', 200.0):.1f}% epoch → every {logging_params['step_piano_roll_frequency']} steps")
    print(
        f"  Save model: {getattr(config, 'save_model_frequency_epoch_pct', 500.0):.1f}% epoch → every {logging_params['save_model_frequency_steps']} steps")

    print(f"\nShuffle Control:")
    print(
        f"  Start shuffle: epoch {getattr(config, 'start_shuffle_on_epoch', 0.0):.1f} → step {shuffle_params['start_shuffle_on_step']}")
    print(f"{'=' * 80}\n")

    # Combine all parameters
    converted_params = {**beta_params, **logging_params, **shuffle_params}

    return converted_params


def create_dataloader_with_conditional_shuffle(dataset, batch_size, current_step, start_shuffle_step):
    """Create DataLoader with shuffle control based on current step"""
    should_shuffle = current_step >= start_shuffle_step
    return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle)


def validate_control_configuration(config):
    """Validate that control configuration is consistent"""
    # Check that lists have matching lengths
    assert len(config['n_encoding_control_tokens']) == len(config['encoding_control_modes']), \
        "Number of encoding control tokens must match number of encoding control modes"
    assert len(config['n_encoding_control_tokens']) == len(config['encoding_control_keys']), \
        "Number of encoding control tokens must match number of encoding control keys"

    assert len(config['n_decoding_control_tokens']) == len(config['decoding_control_modes']), \
        "Number of decoding control tokens must match number of decoding control modes"
    assert len(config['n_decoding_control_tokens']) == len(config['decoding_control_keys']), \
        "Number of decoding control tokens must match number of decoding control keys"

    # Check that modes are valid
    valid_modes = ['prepend', 'add']
    for mode in config['encoding_control_modes']:
        assert mode in valid_modes, f"Invalid encoding control mode: {mode}. Must be one of {valid_modes}"
    for mode in config['decoding_control_modes']:
        assert mode in valid_modes, f"Invalid decoding control mode: {mode}. Must be one of {valid_modes}"

    print(f"\n{'=' * 60}")
    print(f"FLEXIBLE CONTROL CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Encoding Controls ({len(config['n_encoding_control_tokens'])} total):")
    for i, (n_tokens, mode, key) in enumerate(zip(config['n_encoding_control_tokens'],
                                                  config['encoding_control_modes'],
                                                  config['encoding_control_keys'])):
        print(f"  {i}: {n_tokens} tokens, {mode} mode, key: {key}")

    print(f"\nDecoding Controls ({len(config['n_decoding_control_tokens'])} total):")
    for i, (n_tokens, mode, key) in enumerate(zip(config['n_decoding_control_tokens'],
                                                  config['decoding_control_modes'],
                                                  config['decoding_control_keys'])):
        print(f"  {i}: {n_tokens} tokens, {mode} mode, key: {key}")
    print(f"{'=' * 60}\n")


def convert_legacy_config_to_flexcontrol(legacy_config):
    """
    Convert a legacy TripleStreamsVAE config to FlexControlTripleStreamsVAE format

    Args:
        legacy_config: Old configuration dict

    Returns:
        dict: Converted configuration
    """

    flexible_config = legacy_config.copy()

    # Extract legacy control configuration
    prepend_mode = legacy_config.get('prepend_control_tokens', False)

    # Convert control token counts
    flexible_config['n_encoding_control_tokens'] = [
        legacy_config.get('n_encoding_control1_tokens', 33),
        legacy_config.get('n_encoding_control2_tokens', 10)
    ]

    flexible_config['n_decoding_control_tokens'] = [
        legacy_config.get('n_decoding_control1_tokens', 10),
        legacy_config.get('n_decoding_control2_tokens', 10),
        legacy_config.get('n_decoding_control3_tokens', 10)
    ]

    # Set control modes based on legacy prepend_control_tokens setting
    if prepend_mode:
        flexible_config['encoding_control_modes'] = ['prepend', 'add']
        flexible_config['decoding_control_modes'] = ['prepend', 'prepend', 'prepend']
    else:
        flexible_config['encoding_control_modes'] = ['add', 'add']
        flexible_config['decoding_control_modes'] = ['add', 'add', 'add']

    # Set control keys
    flexible_config['encoding_control_keys'] = [
        legacy_config.get('encoding_control1_key', "Flat Out Vs. Input | Hits | Hamming"),
        legacy_config.get('encoding_control2_key', "Flat Out Vs. Input | Accent | Hamming")
    ]

    flexible_config['decoding_control_keys'] = [
        legacy_config.get('decoding_control1_key', "Stream 1 Vs. Flat Out | Hits | Hamming"),
        legacy_config.get('decoding_control2_key', "Stream 2 Vs. Flat Out | Hits | Hamming"),
        legacy_config.get('decoding_control3_key', "Stream 3 Vs. Flat Out | Hits | Hamming")
    ]

    # Remove legacy parameters to avoid confusion
    legacy_params_to_remove = [
        'prepend_control_tokens',
        'n_encoding_control1_tokens', 'n_encoding_control2_tokens',
        'n_decoding_control1_tokens', 'n_decoding_control2_tokens', 'n_decoding_control3_tokens',
        'encoding_control1_key', 'encoding_control2_key',
        'decoding_control1_key', 'decoding_control2_key', 'decoding_control3_key'
    ]

    for param in legacy_params_to_remove:
        flexible_config.pop(param, None)

    print(f"🔄 Legacy configuration converted:")
    print(
        f"  prepend_control_tokens: {prepend_mode} → encoding_modes: {flexible_config['encoding_control_modes']}, decoding_modes: {flexible_config['decoding_control_modes']}")
    print(
        f"  Control tokens: enc={flexible_config['n_encoding_control_tokens']}, dec={flexible_config['n_decoding_control_tokens']}")

    return flexible_config


# Load configuration
loaded_via_config = False
if args.config is not None:
    print(f"\n\n!!!Loading configuration from {args.config}!!!\n\n")
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)

        # Check if this is a legacy config and convert if needed
        if "prepend_control_tokens" in hparams or "n_encoding_control1_tokens" in hparams:
            print("🔄 Detected legacy configuration format. Converting to FlexControl format...")
            hparams = convert_legacy_config_to_flexcontrol(hparams)
            print("✅ Successfully converted legacy configuration")

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

        # Flexible control tokens
        n_encoding_control_tokens=args.n_encoding_control_tokens,
        encoding_control_modes=args.encoding_control_modes,
        encoding_control_keys=args.encoding_control_keys,
        n_decoding_control_tokens=args.n_decoding_control_tokens,
        decoding_control_modes=args.decoding_control_modes,
        decoding_control_keys=args.decoding_control_keys,
        velocity_dropout=args.velocity_dropout,
        offset_dropout=args.offset_dropout,

        # Epoch-percentage-based beta annealing
        beta_annealing_period_epoch_pct=args.beta_annealing_period_epoch_pct,
        beta_annealing_start_first_rise_at_epoch_pct=args.beta_annealing_start_first_rise_at_epoch_pct,
        beta_annealing_per_cycle_rising_ratio=args.beta_annealing_per_cycle_rising_ratio,
        beta_annealing_gap_ratio=args.beta_annealing_gap_ratio,
        beta_annealing_activated=args.beta_annealing_activated,
        beta_level=args.beta_level,

        # Epoch-percentage-based logging
        step_log_frequency_epoch_pct=args.step_log_frequency_epoch_pct,
        step_hit_score_frequency_epoch_pct=args.step_hit_score_frequency_epoch_pct,
        step_piano_roll_frequency_epoch_pct=args.step_piano_roll_frequency_epoch_pct,
        save_model_frequency_epoch_pct=args.save_model_frequency_epoch_pct,

        # Shuffle control
        start_shuffle_on_epoch=args.start_shuffle_on_epoch,

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

# Validate control configuration
validate_control_configuration(hparams)

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

assert "wandb_project" in hparams.keys(), "wandb_project not specified"

if __name__ == "__main__":

    # Initialize wandb
    wandb_run = wandb.init(
        config=hparams,
        project=hparams["wandb_project"],
        entity="behzadhaki",
        settings=wandb.Settings(code_dir="train_FlexControlTripleStreamsVAE.py")
    )

    if loaded_via_config:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_FlexControlTripleStreamsVAE.py")
        wandb.run.log_artifact(model_code)

    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Initialize the model
    model_cpu = FlexControlTripleStreamsVAE(config)
    model_on_device = model_cpu.to(config.device)

    # Instantiate loss functions and optimizer
    hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    velocity_loss_fn = torch.nn.MSELoss(reduction='none')
    offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Load Training and Testing Datasets
    training_dataset = get_flexcontrol_triplestream_dataset(
        config=config,
        subset_tag="train",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
        move_all_to_cuda=args.move_all_to_cuda,
        print_logs=True
    )

    test_dataset = get_flexcontrol_triplestream_dataset(
        config=config,
        subset_tag="test",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
        print_logs=True
    )

    print(f"\n\n|{len(training_dataset)} training samples and {len(test_dataset)} testing samples loaded|\n\n")

    # Calculate steps per epoch and convert epoch percentages to steps
    steps_per_epoch = len(DataLoader(training_dataset, batch_size=config.batch_size, shuffle=False))
    converted_params = convert_epoch_percentages_to_steps(config, steps_per_epoch)

    # Setup step-based beta annealing using converted parameters
    if config.beta_annealing_activated:
        total_steps = steps_per_epoch * config.epochs

        print(f"\n\n|Setting Up Beta Annealing Scheduler|\n\n")
        beta_scheduler = train_utils.BetaAnnealingScheduler(
            total_steps=total_steps,
            period_steps=converted_params['beta_annealing_period_steps'],
            rise_ratio=config.beta_annealing_per_cycle_rising_ratio,
            gap_ratio=getattr(config, 'beta_annealing_gap_ratio', 0.0),
            start_first_rise_at_step=converted_params['beta_annealing_start_first_rise_at_step'],
            beta_level=config.beta_level
        )
        logger.info(f"Using step-based beta annealing with {total_steps} total steps")
    else:
        beta_scheduler = None
        logger.info("Beta annealing disabled")

    # Setup resumable training
    print("\n\n|Setting up resumable training if needed|\n\n")
    starting_step, model_on_device, optimizer, beta_scheduler = train_utils.setup_resumable_training_flexcontrol(
        config, model_on_device, optimizer, beta_scheduler, wandb_run
    )


    # Batch Data IO Extractor
    def batch_data_extractor(data_, device=config.device):
        input_grooves = data_[0].to(device) if data_[0].device.type != device else data_[0]
        output_streams = data_[1].to(device) if data_[1].device.type != device else data_[1]
        encoding_control_tokens = data_[2].to(device) if data_[2].device.type != device else data_[2]
        decoding_control_tokens = data_[3].to(device) if data_[3].device.type != device else data_[3]
        metadata = data_[4]
        indices = data_[5]

        return (input_grooves, output_streams, encoding_control_tokens, decoding_control_tokens,
                metadata, indices)


    def predict_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        (input_grooves, output_streams, encoding_control_tokens, decoding_control_tokens,
         metadata, indices) = batch_data_extractor(batch_data, device)

        with torch.no_grad():
            hvo, latent_z = model_.predict(
                flat_hvo_groove=input_grooves,
                encoding_control_tokens=encoding_control_tokens,
                decoding_control_tokens=decoding_control_tokens)

        return hvo, latent_z


    def forward_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        (input_grooves, target_output_streams, encoding_control_tokens, decoding_control_tokens,
         metadata, indices) = batch_data_extractor(batch_data, device)

        h_logits, v_logits, o_logits, mu, log_var, latent_z = model_.forward(
            flat_hvo_groove=input_grooves,
            encoding_control_tokens=encoding_control_tokens,
            decoding_control_tokens=decoding_control_tokens)

        return h_logits, v_logits, o_logits, mu, log_var, latent_z, target_output_streams


    # Setup evaluation callbacks for step-based evaluation
    def quick_hit_scores_train(step):
        """Quick hit score evaluation for training set"""
        train_set_hit_scores, _ = eval_utils.get_hit_scores(
            config=config,
            subset_tag='train',
            use_cached=True,
            downsampled_size=1000,
            predict_using_batch_data_method=predict_using_batch_data,
            divide_by_collection=True,
            previous_evaluator=None
        )
        return {f"Quick_Eval/{k}": v for k, v in train_set_hit_scores.items()}


    def quick_hit_scores_test(step):
        """Quick hit score evaluation for test set"""
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
        media, _ = eval_utils.get_pianoroll_for_wandb(
            config=config,
            subset_tag='test',
            use_cached=True,
            downsampled_size=200,
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


    # Training loop with dynamic shuffle control
    step_ = starting_step

    for epoch in range(config.epochs):
        print("\n\n|" + "=" * 70 + "|")
        print(f"\t\tEpoch {epoch} of {config.epochs}")
        print(f"\t\tSteps so far: {step_}")
        if starting_step > 0 and epoch == 0:
            print(f"\t\tResumed from step {starting_step}")

        # Check shuffle status for this epoch
        shuffle_enabled = step_ >= converted_params['start_shuffle_on_step']
        shuffle_status = "ON" if shuffle_enabled else "OFF"
        print(f"\t\tDataLoader shuffle: {shuffle_status}")
        if not shuffle_enabled and step_ + steps_per_epoch >= converted_params['start_shuffle_on_step']:
            print(
                f"\t\t*** Shuffle will START during this epoch at step {converted_params['start_shuffle_on_step']} ***")
        print("|" + "=" * 70 + "|")

        # Create DataLoaders with conditional shuffle based on current step
        train_dataloader = create_dataloader_with_conditional_shuffle(
            training_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )
        test_dataloader = create_dataloader_with_conditional_shuffle(
            test_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )

        # Define evaluation callbacks with converted step frequencies
        eval_callbacks = {}

        if args.calculate_hit_scores_on_train:
            eval_callbacks['hit_scores_train'] = {
                'function': quick_hit_scores_train,
                'frequency': converted_params['step_hit_score_frequency']
            }

        if args.calculate_hit_scores_on_test:
            eval_callbacks['hit_scores_test'] = {
                'function': quick_hit_scores_test,
                'frequency': converted_params['step_hit_score_frequency']
            }

        if args.piano_roll_samples:
            eval_callbacks['piano_rolls'] = {
                'function': piano_roll_generation,
                'frequency': converted_params['step_piano_roll_frequency']
            }

        if args.save_model:
            eval_callbacks['save_model'] = {
                'function': save_model_checkpoint,
                'frequency': converted_params['save_model_frequency_steps']
            }

        # Training phase
        model_on_device.train()
        logger.info(f"\n***************************Training epoch {epoch}...")

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
            log_frequency=converted_params['step_log_frequency'],
            eval_callbacks=eval_callbacks
        )

        wandb.log(train_log_metrics, commit=False)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # Testing phase
        model_on_device.eval()
        logger.info(f"\n***************************Testing epoch {epoch}...")

        # Create fresh test dataloader for testing phase (may have different shuffle state)
        test_dataloader_eval = create_dataloader_with_conditional_shuffle(
            test_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )

        test_log_metrics, _ = train_utils.test_loop_step_based(
            test_dataloader=test_dataloader_eval,
            forward_method=forward_using_batch_data,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            starting_step=step_,
            beta_scheduler=beta_scheduler,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss,
            log_frequency=converted_params['step_log_frequency']
        )

        wandb.log(test_log_metrics, commit=False)

        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # Log epoch-level information
        epoch_info = {
            "epoch": epoch,
            "shuffle_enabled": step_ >= converted_params['start_shuffle_on_step'],
            "steps_per_epoch": steps_per_epoch,
            "current_epoch_from_steps": step_ / steps_per_epoch,
        }

        # Add beta scheduler info if available
        if beta_scheduler is not None:
            cycle_info = beta_scheduler.get_cycle_info(step_)
            epoch_info.update({
                "beta_cycle_number": cycle_info['cycle_number'],
                "beta_phase": cycle_info['phase'],
                "current_beta": cycle_info['beta']
            })

        wandb.log(epoch_info, step=step_)

        logger.info(
            f"Epoch {epoch} completed - Train loss: "
            f"{train_log_metrics.get('Train_Epoch_Metrics/loss_total_rec_w_kl', 'N/A'):.4f}, "
            f"Test loss: {test_log_metrics.get('Test_Epoch_Metrics/loss_total_rec_w_kl', 'N/A'):.4f}"
        )

        if config.device == 'cuda':
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n\n{'=' * 80}")
    print(f"TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Final step: {step_}")
    print(f"Total epochs completed: {config.epochs}")
    print(f"Final epoch from steps: {step_ / steps_per_epoch:.2f}")
    if beta_scheduler is not None:
        final_cycle_info = beta_scheduler.get_cycle_info(step_)
        print(f"Final beta cycle: {final_cycle_info['cycle_number']}")
        print(f"Final beta phase: {final_cycle_info['phase']}")
        print(f"Final beta value: {final_cycle_info['beta']:.4f}")
    print(f"{'=' * 80}\n")

    wandb.finish()