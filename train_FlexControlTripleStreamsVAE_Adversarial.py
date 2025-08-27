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

# Import adversarial components
from helpers.adversarial_components import create_adversarial_components
from helpers.adversarial_train_utils import (
    train_loop_step_based_adversarial,
    test_loop_step_based_adversarial,
    save_model_checkpoint_enhanced_adversarial,
    setup_resumable_training_adversarial
)

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
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name",
                    default="FlexControlTripleStreamsVAE_Adversarial")

# ----------------------- Adversarial Training Parameters -----------------------
parser.add_argument("--use_adversarial_training", type=bool, help="Enable adversarial training", default=True)
parser.add_argument("--adversarial_weight", type=float, help="Weight for adversarial loss", default=0.1)
parser.add_argument("--adversarial_warmup_steps_epoch_pct", type=float, help="Adversarial warmup in epoch percentage",
                    default=10.0)
parser.add_argument("--discriminator_lr", type=float, help="Learning rate for discriminator", default=0.0001)
parser.add_argument("--discriminator_hidden_dim", type=int, help="Hidden dimension for discriminator", default=256)
parser.add_argument("--d_train_frequency", type=int, help="Train discriminator every N steps", default=1)
parser.add_argument("--feature_matching_weight", type=float, help="Weight for feature matching loss", default=10.0)
parser.add_argument("--diversity_weight", type=float, help="Weight for diversity regularization", default=0.5)

# ----------------------- Checkpoint Resume Parameters -----------------------
parser.add_argument("--resume_from_checkpoint", type=bool, help="Resume training from checkpoint", default=False)
parser.add_argument("--checkpoint_wandb_run_id", type=str, help="WandB run ID to resume from", default=None)
parser.add_argument("--checkpoint_step", type=int, help="Step number to resume from (optional)", default=None)
parser.add_argument("--checkpoint_artifact_name", type=str, help="Name of model artifact (e.g., model_step_10000)",
                    default=None)
parser.add_argument("--reset_optimizer", type=bool, help="Reset optimizer state when resuming", default=False)
parser.add_argument("--reset_discriminator", type=bool, help="Reset discriminator when resuming", default=False)
parser.add_argument("--reset_beta_scheduler", type=bool, help="Reset beta scheduler when resuming", default=False)

# ----------------------- Model Parameters (same as original) -----------------------
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

# ---------------------- Flexible Control Parameters (same as original) -----------------------
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

# ----------------------- Other Parameters (same as original) -----------------------
parser.add_argument("--beta_annealing_period_epoch_pct", type=float, default=100.0)
parser.add_argument("--beta_annealing_start_first_rise_at_epoch_pct", type=float, default=20.0)
parser.add_argument("--beta_annealing_per_cycle_rising_ratio", type=float, default=0.5)
parser.add_argument("--beta_annealing_gap_ratio", type=float, default=0.0)
parser.add_argument("--beta_annealing_activated", type=bool, default=True)
parser.add_argument("--beta_level", type=float, help="Max level of beta term on KL", default=0.2)

parser.add_argument("--step_log_frequency_epoch_pct", type=float, default=1.0)
parser.add_argument("--step_hit_score_frequency_epoch_pct", type=float, default=50.0)
parser.add_argument("--step_piano_roll_frequency_epoch_pct", type=float, default=200.0)
parser.add_argument("--save_model_frequency_epoch_pct", type=float, default=500.0)

parser.add_argument("--start_shuffle_on_epoch", type=float, default=0.0)
parser.add_argument("--dropout", type=float, help="Dropout", default=0.1)
parser.add_argument("--velocity_dropout", type=float, help="velocity_dropout", default=0)
parser.add_argument("--offset_dropout", type=float, help="offset_dropout", default=0)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1000)
parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
parser.add_argument("--lr", type=float, help="Learning rate", default=0.0006369948608989782)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam'", default="adam")

parser.add_argument("--scale_h_loss", type=float, help="Scale for hit loss", default=1)
parser.add_argument("--scale_v_loss", type=float, help="Scale for velocity loss", default=1)
parser.add_argument("--scale_o_loss", type=float, help="Scale for offset loss", default=1)

parser.add_argument("--device", type=str, help="Device to use for training", required=True)
parser.add_argument("--move_all_to_cuda", type=bool, help="places all training data on cuda", default=True)

parser.add_argument("--dataset_root_path", type=str, help="Root path for dataset files",
                    default="data/triple_streams/model_ready/AccentAt0.75/")
parser.add_argument("--dataset_files", nargs='+', help="List of dataset files",
                    default=["01_candombe_four_voices.pkl.bz2"])
parser.add_argument("--evaluate_on_subset", type=str, help="Using test or evaluation subset",
                    default="test", choices=['test', 'evaluation'])
parser.add_argument("--augment_with_no_inputs", type=bool, default=False)

parser.add_argument("--calculate_hit_scores_on_train", type=bool, default=True)
parser.add_argument("--calculate_hit_scores_on_test", type=bool, default=True)
parser.add_argument("--piano_roll_samples", type=bool, help="Generate piano roll samples", default=True)

parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model",
                    default="misc/FlexControlTripleStreamsVAE_Adversarial")

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

# Load configuration (same logic as original)
loaded_via_config = False
if args.config is not None:
    print(f"\n\n!!!Loading configuration from {args.config}!!!\n\n")
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)

        # Convert legacy config if needed (same as original)
        from train_FlexControlTripleStreamsVAE import convert_legacy_config_to_flexcontrol

        if "prepend_control_tokens" in hparams or "n_encoding_control1_tokens" in hparams:
            print("Converting legacy configuration format to FlexControl format...")
            hparams = convert_legacy_config_to_flexcontrol(hparams)
            print("Successfully converted legacy configuration")

        if "wandb_project" not in hparams.keys():
            hparams["wandb_project"] = args.wandb_project
        if "device" in hparams.keys():
            logger.warning(f"\n\nRemove device from config file. Using CLI argument instead: {args.device}\n\n")
        hparams["device"] = args.device  # Always use CLI device argument
        loaded_via_config = True
else:
    # Build hparams from CLI arguments (including adversarial parameters)
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

        # Adversarial training parameters
        use_adversarial_training=args.use_adversarial_training,
        adversarial_weight=args.adversarial_weight,
        adversarial_warmup_steps_epoch_pct=args.adversarial_warmup_steps_epoch_pct,
        discriminator_lr=args.discriminator_lr,
        discriminator_hidden_dim=args.discriminator_hidden_dim,
        d_train_frequency=args.d_train_frequency,
        feature_matching_weight=args.feature_matching_weight,
        diversity_weight=args.diversity_weight,

        # Beta annealing
        beta_annealing_period_epoch_pct=args.beta_annealing_period_epoch_pct,
        beta_annealing_start_first_rise_at_epoch_pct=args.beta_annealing_start_first_rise_at_epoch_pct,
        beta_annealing_per_cycle_rising_ratio=args.beta_annealing_per_cycle_rising_ratio,
        beta_annealing_gap_ratio=args.beta_annealing_gap_ratio,
        beta_annealing_activated=args.beta_annealing_activated,
        beta_level=args.beta_level,

        # Logging
        step_log_frequency_epoch_pct=args.step_log_frequency_epoch_pct,
        step_hit_score_frequency_epoch_pct=args.step_hit_score_frequency_epoch_pct,
        step_piano_roll_frequency_epoch_pct=args.step_piano_roll_frequency_epoch_pct,
        save_model_frequency_epoch_pct=args.save_model_frequency_epoch_pct,

        # Training control
        start_shuffle_on_epoch=args.start_shuffle_on_epoch,
        scale_h_loss=args.scale_h_loss,
        scale_v_loss=args.scale_v_loss,
        scale_o_loss=args.scale_o_loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        device=args.device,

        # Data parameters
        dataset_root_path=args.dataset_root_path,
        dataset_files=args.dataset_files,
        augment_with_no_inputs=args.augment_with_no_inputs,

        # Checkpoint resume parameters
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_wandb_run_id=args.checkpoint_wandb_run_id,
        checkpoint_step=args.checkpoint_step,
        checkpoint_artifact_name=args.checkpoint_artifact_name,
        reset_optimizer=args.reset_optimizer,
        reset_discriminator=args.reset_discriminator,
        reset_beta_scheduler=args.reset_beta_scheduler
    )

# Device availability checks (same as original)
if args.device == 'mps' and not torch.has_mps:
    logger.warning("\n\n MPS is not available. Falling back to CPU.")
    hparams["device"] = 'cpu'

if args.device == 'cuda' and not torch.cuda.is_available():
    logger.warning("\n\n CUDA is not available. Falling back to CPU.")
    hparams["device"] = 'cpu'

is_testing = hparams.get("is_testing", False) or args.is_testing

# Validate control configuration (same as original)
from train_FlexControlTripleStreamsVAE import validate_control_configuration

validate_control_configuration(hparams)

# Print configuration
print("\n\n|" + "=" * 80 + "|")
print(f"\n\tHyperparameters for adversarial training run:")
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
        settings=wandb.Settings(code_dir="train_FlexControlTripleStreamsVAE_Adversarial.py")
    )

    if loaded_via_config:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_FlexControlTripleStreamsVAE_Adversarial.py")
        wandb.run.log_artifact(model_code)

    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Initialize the model
    model_cpu = FlexControlTripleStreamsVAE(config)
    model_on_device = model_cpu.to(config.device)

    # Instantiate loss functions and optimizer
    hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    velocity_loss_fn = torch.nn.MSELoss(reduction='mean')
    offset_loss_fn = torch.nn.MSELoss(reduction='mean')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Initialize adversarial training components
    adversarial_trainer = None
    if config.get('use_adversarial_training', False):
        print(f"\n\n|Initializing Adversarial Training Components|\n\n")

        # Create adversarial configuration
        adversarial_config = {
            'max_len': config.get('max_len', 32),
            'discriminator_hidden_dim': config.get('discriminator_hidden_dim', 256),
            'discriminator_dropout': config.get('discriminator_dropout', 0.3),
            'discriminator_lr': config.get('discriminator_lr', 0.0001),
            'discriminator_betas': config.get('discriminator_betas', (0.5, 0.999)),
            'generator_loss_weight': config.get('adversarial_weight', 0.1),
            'feature_matching_weight': config.get('feature_matching_weight', 10.0),
            'diversity_weight': config.get('diversity_weight', 0.5),
            'd_train_frequency': config.get('d_train_frequency', 1),
            'gan_loss_type': config.get('gan_loss_type', 'standard'),
            'use_feature_matching': config.get('use_feature_matching', True)
        }

        discriminator, discriminator_optimizer, adversarial_trainer = create_adversarial_components(adversarial_config)
        discriminator = discriminator.to(config.device)

        print(f"Discriminator architecture:")
        print(f"  Hidden dimension: {adversarial_config['discriminator_hidden_dim']}")
        print(f"  Dropout: {adversarial_config['discriminator_dropout']}")
        print(f"  Learning rate: {adversarial_config['discriminator_lr']}")
        print(f"  Training frequency: {adversarial_config['d_train_frequency']}")
        print(f"  Generator loss weight: {adversarial_config['generator_loss_weight']}")
        print(f"  Feature matching weight: {adversarial_config['feature_matching_weight']}")
        print(f"  Diversity weight: {adversarial_config['diversity_weight']}")

    # Load Training and Testing Datasets (same as original)
    training_dataset = get_flexcontrol_triplestream_dataset(
        config=config,
        subset_tag="train",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
        move_all_to_cuda=args.move_all_to_cuda,
        augment_with_no_inputs=config.augment_with_no_inputs,
        print_logs=True
    )

    test_dataset = get_flexcontrol_triplestream_dataset(
        config=config,
        subset_tag="test",
        use_cached=True,
        downsampled_size=1000 if is_testing else None,
        augment_with_no_inputs=config.augment_with_no_inputs,
        print_logs=True
    )

    print(f"\n\n|{len(training_dataset)} training samples and {len(test_dataset)} testing samples loaded|\n\n")

    # Calculate steps per epoch and convert epoch percentages to steps
    from train_FlexControlTripleStreamsVAE import convert_epoch_percentages_to_steps

    steps_per_epoch = len(DataLoader(training_dataset, batch_size=config.batch_size, shuffle=False))
    converted_params = convert_epoch_percentages_to_steps(config, steps_per_epoch)

    # Calculate adversarial warmup steps
    if adversarial_trainer is not None:
        adversarial_warmup_steps = int(steps_per_epoch * config.get('adversarial_warmup_steps_epoch_pct', 10.0) / 100.0)
        adversarial_warmup_steps = max(adversarial_warmup_steps, 1) # At least 1 step (only happens in testing)
        print(
            f"\n\n|Adversarial training will start after {adversarial_warmup_steps} steps ({config.get('adversarial_warmup_steps_epoch_pct', 10.0)}% of first epoch)|\n\n")
    else:
        adversarial_warmup_steps = 0

    # Setup step-based beta annealing (same as original)
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

    # Setup resumable training with adversarial components
    print("\n\n|Setting up resumable training if needed|\n\n")
    if adversarial_trainer is not None:
        starting_step, model_on_device, optimizer, adversarial_trainer, beta_scheduler = setup_resumable_training_adversarial(
            config, model_on_device, optimizer, adversarial_trainer, beta_scheduler, wandb_run
        )
    else:
        starting_step, model_on_device, optimizer, beta_scheduler = train_utils.setup_resumable_training_flexcontrol(
            config, model_on_device, optimizer, beta_scheduler, wandb_run
        )

    # Batch Data IO Extractor (same as original)
    from train_FlexControlTripleStreamsVAE import create_dataloader_with_conditional_shuffle


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


    # Setup evaluation callbacks (same as original)
    def quick_hit_scores_train(step):
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
        if adversarial_trainer is not None:
            return save_model_checkpoint_enhanced_adversarial(
                model_on_device, optimizer, beta_scheduler, adversarial_trainer, step,
                args.save_model_dir, config.wandb_project, run_name, run_id
            )
        else:
            return train_utils.save_model_checkpoint_enhanced(
                model_on_device, optimizer, beta_scheduler, step,
                args.save_model_dir, config.wandb_project, run_name, run_id
            )


    # Training loop with adversarial training
    step_ = starting_step

    for epoch in range(config.epochs):
        print("\n\n|" + "=" * 70 + "|")
        print(f"\t\tEpoch {epoch} of {config.epochs}")
        print(f"\t\tSteps so far: {step_}")
        if starting_step > 0 and epoch == 0:
            print(f"\t\tResumed from step {starting_step}")

        if adversarial_trainer is not None:
            adv_status = "ON" if step_ >= adversarial_warmup_steps else f"OFF (starts at step {adversarial_warmup_steps})"
            print(f"\t\tAdversarial training: {adv_status}")

        # Check shuffle status for this epoch
        shuffle_enabled = step_ >= converted_params['start_shuffle_on_step']
        shuffle_status = "ON" if shuffle_enabled else "OFF"
        print(f"\t\tDataLoader shuffle: {shuffle_status}")
        print("|" + "=" * 70 + "|")

        # Create DataLoaders with conditional shuffle
        train_dataloader = create_dataloader_with_conditional_shuffle(
            training_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )
        test_dataloader = create_dataloader_with_conditional_shuffle(
            test_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )

        # Define evaluation callbacks
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

        if adversarial_trainer is not None:
            train_log_metrics, step_ = train_loop_step_based_adversarial(
                train_dataloader=train_dataloader,
                forward_method=forward_using_batch_data,
                optimizer=optimizer,
                hit_loss_fn=hit_loss_fn,
                velocity_loss_fn=velocity_loss_fn,
                offset_loss_fn=offset_loss_fn,
                adversarial_trainer=adversarial_trainer,
                starting_step=step_,
                beta_scheduler=beta_scheduler,
                scale_h_loss=config.scale_h_loss,
                scale_v_loss=config.scale_v_loss,
                scale_o_loss=config.scale_o_loss,
                adversarial_weight=config.get('adversarial_weight', 0.1),
                warmup_steps=adversarial_warmup_steps,
                log_frequency=converted_params['step_log_frequency'],
                eval_callbacks=eval_callbacks
            )
        else:
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

        test_dataloader_eval = create_dataloader_with_conditional_shuffle(
            test_dataset, config.batch_size, step_, converted_params['start_shuffle_on_step']
        )

        if adversarial_trainer is not None:
            test_log_metrics, _ = test_loop_step_based_adversarial(
                test_dataloader=test_dataloader_eval,
                forward_method=forward_using_batch_data,
                hit_loss_fn=hit_loss_fn,
                velocity_loss_fn=velocity_loss_fn,
                offset_loss_fn=offset_loss_fn,
                adversarial_trainer=adversarial_trainer,
                starting_step=step_,
                beta_scheduler=beta_scheduler,
                scale_h_loss=config.scale_h_loss,
                scale_v_loss=config.scale_v_loss,
                scale_o_loss=config.scale_o_loss,
                log_frequency=converted_params['step_log_frequency']
            )
        else:
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

        # Log epoch-level information (including adversarial metrics)
        epoch_info = {
            "epoch": epoch,
            "shuffle_enabled": step_ >= converted_params['start_shuffle_on_step'],
            "steps_per_epoch": steps_per_epoch,
            "current_epoch_from_steps": step_ / steps_per_epoch,
        }

        if adversarial_trainer is not None:
            epoch_info.update({
                "adversarial_training_active": step_ >= adversarial_warmup_steps,
                "discriminator_step": adversarial_trainer.step_count,
            })

        if beta_scheduler is not None:
            cycle_info = beta_scheduler.get_cycle_info(step_)
            epoch_info.update({
                "beta_cycle_number": cycle_info['cycle_number'],
                "beta_phase": cycle_info['phase'],
                "current_beta": cycle_info['beta']
            })

        wandb.log(epoch_info, step=step_)

        # Log training summary
        train_loss = train_log_metrics.get('Train_Epoch_Metrics/loss_total_rec_w_kl', 'N/A')
        test_loss = test_log_metrics.get('Test_Epoch_Metrics/loss_total_rec_w_kl', 'N/A')
        logger.info(f"Epoch {epoch} completed - Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

        if adversarial_trainer is not None and step_ >= adversarial_warmup_steps:
            adv_loss = train_log_metrics.get('Train_Epoch_Metrics/loss_adversarial', 'N/A')
            d_loss = train_log_metrics.get('Train_Epoch_Metrics/discriminator_loss', 'N/A')
            logger.info(f"  Adversarial - Gen loss: {adv_loss:.4f}, Disc loss: {d_loss:.4f}")

        if config.device == 'cuda':
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n\n{'=' * 80}")
    print(f"ADVERSARIAL TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Final step: {step_}")
    print(f"Total epochs completed: {config.epochs}")
    print(f"Final epoch from steps: {step_ / steps_per_epoch:.2f}")
    if adversarial_trainer is not None:
        print(f"Final discriminator step: {adversarial_trainer.step_count}")
        print(f"Adversarial training was active: {step_ >= adversarial_warmup_steps}")
    if beta_scheduler is not None:
        final_cycle_info = beta_scheduler.get_cycle_info(step_)
        print(f"Final beta cycle: {final_cycle_info['cycle_number']}")
        print(f"Final beta phase: {final_cycle_info['phase']}")
        print(f"Final beta value: {final_cycle_info['beta']:.4f}")
    print(f"{'=' * 80}\n")

    wandb.finish()