import torch
import numpy as np
import tqdm

from logging import getLogger

logger = getLogger("train_utils")
logger.setLevel("DEBUG")


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification problems to address class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass for computing the focal loss."""
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        if self.reduction.lower() == 'mean':
            return torch.mean(F_loss)
        elif self.reduction.lower() == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1., reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, h_logits, h_targets):
        h_probs = torch.sigmoid(h_logits).reshape(-1)
        targets = h_targets.reshape(-1)
        intersection = (h_probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (h_probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class AdaptiveBetaScheduler:
    """Adaptive beta scheduler that targets a specific KL divergence"""

    def __init__(self, target_kl=4.0, beta_min=0.0, beta_max=1.0, adjustment_rate=0.001):
        self.target_kl = target_kl
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.adjustment_rate = adjustment_rate
        self.current_beta = beta_min
        self.kl_history = []

    def update(self, current_kl):
        """Update beta based on current KL divergence"""
        self.kl_history.append(current_kl)

        # Use moving average of recent KL values
        recent_kl = np.mean(self.kl_history[-10:]) if len(self.kl_history) >= 10 else current_kl

        if recent_kl < self.target_kl:
            # KL too low, decrease beta to allow more deviation from prior
            self.current_beta = max(self.beta_min, self.current_beta - self.adjustment_rate)
        else:
            # KL too high, increase beta to push closer to prior
            self.current_beta = min(self.beta_max, self.current_beta + self.adjustment_rate)

        return self.current_beta

    def get_beta(self, step=None):
        return self.current_beta


class BetaAnnealingScheduler:
    """Beta annealing scheduler that works with training steps - optimized version"""

    def __init__(self, total_steps, period_steps, rise_ratio, start_first_rise_at_step=0, beta_level=1.0,
                 gap_ratio=0.0):
        self.total_steps = total_steps
        self.period_steps = period_steps
        self.rise_ratio = rise_ratio
        self.gap_ratio = gap_ratio  # New parameter for gap between cycles
        self.start_first_rise_at_step = start_first_rise_at_step
        self.beta_level = beta_level
        self.current_step = 0

        # Pre-compute the single cycle for efficiency
        self.single_cycle = self._generate_single_beta_cycle(period_steps, rise_ratio, gap_ratio)

        print(f"BetaAnnealingScheduler initialized:")
        print(f"  total_steps: {total_steps}")
        print(f"  period_steps: {period_steps}")
        print(f"  rise_ratio: {rise_ratio}")
        print(f"  gap_ratio: {gap_ratio}")
        print(f"  start_first_rise_at_step: {start_first_rise_at_step}")
        print(f"  beta_level: {beta_level}")
        print(f"  single_cycle shape: {self.single_cycle.shape}")

    def _f(self, x, K):
        """Sigmoid-like function for smooth transitions"""
        if x == 0:
            return 0
        elif x == K:
            return 1
        else:
            return 1 / (1 + np.exp(-10 * (x - K / 2) / K))

    def _generate_rising_curve(self, K):
        """Generate a rising curve of length K"""
        curve = []
        for i in range(K):
            curve.append(self._f(i, K - 1))
        return np.array(curve)

    def _generate_single_beta_cycle(self, period, rise_ratio, gap_ratio):
        """Generate a single beta cycle with optional gap at the end"""
        cycle = np.ones(period)

        # Calculate the number of steps for each phase
        rise_steps = int(period * rise_ratio)
        gap_steps = int(period * gap_ratio)
        plateau_steps = period - rise_steps - gap_steps

        # Ensure we don't exceed the period
        if rise_steps + gap_steps > period:
            gap_steps = max(0, period - rise_steps)
            plateau_steps = 0

        # Generate the cycle
        if rise_steps > 0:
            rising_curve = self._generate_rising_curve(rise_steps)
            cycle[:rise_steps] = rising_curve

        # Plateau phase (beta = 1)
        if plateau_steps > 0:
            cycle[rise_steps:rise_steps + plateau_steps] = 1.0

        # Gap phase (beta = 0)
        if gap_steps > 0:
            cycle[rise_steps + plateau_steps:] = 0.0

        return cycle

    def get_beta(self, step=None):
        """Get beta value for current or specified step using modular arithmetic"""
        if step is None:
            step = self.current_step

        # Handle steps before the first rise
        if step < self.start_first_rise_at_step:
            return 0.0

        # Calculate position within the cycling phase
        effective_step = step - self.start_first_rise_at_step

        # Use modulo to find position within current cycle
        cycle_position = effective_step % self.period_steps

        # Get beta value from the pre-computed single cycle
        if cycle_position < len(self.single_cycle):
            beta_value = self.single_cycle[cycle_position]
        else:
            # Fallback (shouldn't happen if cycle is generated correctly)
            beta_value = 1.0

        return beta_value * self.beta_level

    def step(self):
        """Increment the step counter"""
        self.current_step += 1
        return self.get_beta()

    def get_cycle_info(self, step=None):
        """Get debugging information about current cycle position"""
        if step is None:
            step = self.current_step

        if step < self.start_first_rise_at_step:
            return {
                'step': step,
                'phase': 'warmup',
                'cycle_number': 0,
                'cycle_position': 0,
                'beta': 0.0
            }

        effective_step = step - self.start_first_rise_at_step
        cycle_number = effective_step // self.period_steps
        cycle_position = effective_step % self.period_steps

        # Determine which phase we're in within the cycle
        rise_steps = int(self.period_steps * self.rise_ratio)
        gap_steps = int(self.period_steps * self.gap_ratio)
        plateau_steps = self.period_steps - rise_steps - gap_steps

        if cycle_position < rise_steps:
            phase_name = 'rising'
        elif cycle_position < rise_steps + plateau_steps:
            phase_name = 'plateau'
        else:
            phase_name = 'gap'

        return {
            'step': step,
            'phase': phase_name,
            'cycle_number': cycle_number,
            'cycle_position': cycle_position,
            'rise_steps': rise_steps,
            'plateau_steps': plateau_steps,
            'gap_steps': gap_steps,
            'beta': self.get_beta(step)
        }

def load_checkpoint_from_wandb(wandb_project, run_id, artifact_name, step=None, wandb_run=None):
    """
    Load model checkpoint from WandB artifacts using the newer API

    Args:
        wandb_project: WandB project name
        run_id: WandB run ID (for backward compatibility, not used with new API)
        artifact_name: Name of the artifact (e.g., 'model_step_10000')
        step: Step number (optional, can be inferred from artifact_name)
        wandb_run: Active wandb run instance (if None, will use current run)

    Returns:
        checkpoint_data, checkpoint_step
    """
    import wandb

    # Use the newer wandb.run.use_artifact() API
    try:
        # Construct the full artifact path
        # Format: 'entity/project/artifact_name:version'
        artifact_path = f"behzadhaki/{wandb_project}/{artifact_name}"

        # Use the current run's use_artifact method
        if wandb_run is None:
            if wandb.run is None:
                raise RuntimeError("No active wandb run found. Please ensure wandb.init() has been called.")
            artifact = wandb.run.use_artifact(artifact_path, type='model')
        else:
            artifact = wandb_run.use_artifact(artifact_path, type='model')

        artifact_dir = artifact.download()
        logger.info(f"Downloaded artifact {artifact_name} to {artifact_dir}")
    except Exception as e:
        logger.error(f"Failed to download artifact {artifact_name}: {e}")
        logger.error(f"Tried to access: behzadhaki/{wandb_project}/{artifact_name}")
        raise

    # Find the model file
    import glob
    import os
    model_files = glob.glob(os.path.join(artifact_dir, "*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No .pth files found in artifact {artifact_name}")

    model_path = model_files[0]
    logger.info(f"Loading checkpoint from {model_path}")

    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract step number if not provided
    if step is None:
        if 'step_' in artifact_name:
            try:
                # Handle artifact names with versions (e.g., "model_step_19000:v1")
                # First remove version if present, then extract step
                artifact_base = artifact_name.split(':')[0]  # Remove ":v1" part
                step_part = artifact_base.split('step_')[-1]  # Get "19000"
                step = int(step_part)
                logger.info(f"Parsed step {step} from artifact name {artifact_name}")
            except ValueError:
                step = 0
                logger.warning(f"Could not parse step from artifact name '{artifact_name}', defaulting to 0")
        else:
            step = 0
            logger.warning("Could not determine step from artifact name, defaulting to 0")

    logger.info(f"Checkpoint loaded from step {step}")
    return checkpoint, step


def validate_architecture_compatibility(old_config, new_config):
    """
    Validate that the new config is compatible with the old model architecture

    Args:
        old_config: Configuration from the checkpoint (dict or wandb.Config)
        new_config: New configuration (dict or wandb.Config)

    Returns:
        bool: True if compatible, raises ValueError if not
    """
    # Define architecture-critical parameters
    architecture_params = [
        'd_model_enc', 'd_model_dec', 'embedding_size_src', 'embedding_size_tgt',
        'nhead_enc', 'nhead_dec', 'dim_feedforward_enc', 'dim_feedforward_dec',
        'num_encoder_layers', 'num_decoder_layers', 'latent_dim', 'max_len_enc', 'max_len_dec',
        'n_encoding_control1_tokens', 'n_encoding_control2_tokens',
        'n_decoding_control1_tokens', 'n_decoding_control2_tokens', 'n_decoding_control3_tokens'
    ]

    incompatible_params = []

    for param in architecture_params:
        old_val = old_config.get(param) if hasattr(old_config, 'get') else getattr(old_config, param, None)
        new_val = new_config.get(param) if hasattr(new_config, 'get') else getattr(new_config, param, None)

        if old_val is not None and new_val is not None:
            if old_val != new_val:
                incompatible_params.append(param)
                logger.error(f"  {param}: checkpoint={old_val} vs current={new_val}")

    if incompatible_params:
        raise ValueError(f"Architecture incompatible! Different values for: {incompatible_params}")

    logger.info("Architecture compatibility validated successfully")
    return True


def setup_resumable_training(config, model, optimizer, beta_scheduler=None, wandb_run=None):
    """
    Setup training resumption from checkpoint (Option 3: Simple approach)

    Args:
        config: Current configuration (uses CLI args/config file, no inheritance)
        model: Model instance
        optimizer: Optimizer instance
        beta_scheduler: Beta scheduler instance (optional)
        wandb_run: Active wandb run instance (optional)

    Returns:
        starting_step, updated_model, updated_optimizer, updated_beta_scheduler
    """
    starting_step = 0

    if getattr(config, 'resume_from_checkpoint', False):
        if not getattr(config, 'checkpoint_artifact_name', None):
            raise ValueError("Must specify checkpoint_artifact_name when resuming")

        logger.info(f"Resuming from checkpoint: {config.checkpoint_artifact_name}")

        # Load checkpoint using the new API
        checkpoint_data, checkpoint_step = load_checkpoint_from_wandb(
            config.wandb_project,
            getattr(config, 'checkpoint_wandb_run_id', None),  # Not used with new API but kept for compatibility
            config.checkpoint_artifact_name,
            getattr(config, 'checkpoint_step', None),
            wandb_run
        )

        # Get old configuration from WandB for architecture validation
        # We still need the API for accessing run config
        if getattr(config, 'checkpoint_wandb_run_id', None):
            import wandb
            api = wandb.Api()
            old_run = api.run(f"behzadhaki/{config.wandb_project}/{config.checkpoint_wandb_run_id}")
            old_config = old_run.config

            # Validate architecture compatibility
            validate_architecture_compatibility(old_config, config)
        else:
            logger.warning("No checkpoint_wandb_run_id provided - skipping architecture validation")

        # Load model state
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            # Enhanced checkpoint format
            model.load_state_dict(checkpoint_data['model_state_dict'])
            logger.info("Loaded model state from enhanced checkpoint")

            # Load optimizer state if available and not resetting
            if 'optimizer_state_dict' in checkpoint_data and not getattr(config, 'reset_optimizer', False):
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    logger.info("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}. Continuing with fresh optimizer.")
            else:
                logger.info("Optimizer state reset or not available")

            # Load beta scheduler state if available and not resetting
            if beta_scheduler is not None and 'beta_scheduler_state' in checkpoint_data and not getattr(config,
                                                                                                        'reset_beta_scheduler',
                                                                                                        False):
                try:
                    beta_scheduler.current_step = checkpoint_data['beta_scheduler_state'].get('current_step',
                                                                                              checkpoint_step)
                    if hasattr(beta_scheduler, 'kl_history'):
                        beta_scheduler.kl_history = checkpoint_data['beta_scheduler_state'].get('kl_history', [])
                    logger.info("Loaded beta scheduler state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load beta scheduler state: {e}. Continuing with fresh scheduler.")
            elif beta_scheduler is not None and getattr(config, 'reset_beta_scheduler', False):
                logger.info("Beta scheduler state reset")

        else:
            # Legacy checkpoint format (just model state dict)
            model.load_state_dict(checkpoint_data)
            logger.info("Loaded model state from legacy checkpoint format")

        starting_step = checkpoint_step
        logger.info(f"Resuming training from step {starting_step}")

        # Log the resumption details
        import wandb
        resume_info = {
            "resumed_from_artifact": config.checkpoint_artifact_name,
            "resumed_from_step": starting_step,
            "reset_optimizer": getattr(config, 'reset_optimizer', False),
            "reset_beta_scheduler": getattr(config, 'reset_beta_scheduler', False)
        }
        if getattr(config, 'checkpoint_wandb_run_id', None):
            resume_info["resumed_from_run"] = config.checkpoint_wandb_run_id

        wandb.log(resume_info)

    else:
        logger.info("Starting training from scratch")

    return starting_step, model, optimizer, beta_scheduler


def save_model_checkpoint_enhanced(model, optimizer, beta_scheduler, step, save_dir, wandb_project, run_name, run_id):
    """Enhanced model saving with optimizer and scheduler state"""
    if step > 0:
        import wandb
        import os

        model_artifact = wandb.Artifact(f'model_step_{step}', type='model')
        model_path = f"{save_dir}/{wandb_project}/{run_name}_{run_id}/step_{step}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save comprehensive checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }

        # Add beta scheduler state if available
        if beta_scheduler is not None:
            checkpoint['beta_scheduler_state'] = {
                'current_step': getattr(beta_scheduler, 'current_step', step),
                'current_beta': getattr(beta_scheduler, 'current_beta', 1.0)
            }

            # Add kl_history if it exists (for AdaptiveBetaScheduler)
            if hasattr(beta_scheduler, 'kl_history'):
                checkpoint['beta_scheduler_state']['kl_history'] = beta_scheduler.kl_history

        torch.save(checkpoint, model_path)
        model_artifact.add_file(model_path)
        wandb.run.log_artifact(model_artifact)
        logger.info(f"Enhanced checkpoint saved to {model_path}")

    return {}


def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function):
    assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss) or isinstance(hit_loss_function,
                                                                                   FocalLoss) or isinstance(
        hit_loss_function,
        DiceLoss), f"hit_loss_function must be an instance of torch.nn.BCEWithLogitsLoss or FocalLoss or DiceLoss. Got {type(hit_loss_function)}"
    loss_h = hit_loss_function(hit_logits, hit_targets)
    hit_mask = None
    if hit_loss_function.reduction == 'none':
        # put more weight on the hits
        hit_mask = (hit_targets > 0).float() * 3 + 1  # hits weighted almost 4x more than misses
        loss_h = loss_h * hit_mask
        loss_h = loss_h.mean()

    return loss_h, hit_mask


def calculate_velocity_loss(vel_logits, vel_targets, vel_loss_function, hit_mask=None):
    vel_activated = torch.tanh(vel_logits)
    if hit_mask is None:
        return (vel_loss_function(vel_activated, vel_targets - 0.5)).mean()
    else:
        return (vel_loss_function(vel_activated, vel_targets - 0.5) * hit_mask).mean()


def calculate_offset_loss(offset_logits, offset_targets, offset_loss_function, hit_mask=None):
    offset_activated = torch.tanh(offset_logits)
    if hit_mask is None:
        return offset_loss_function(offset_activated, offset_targets).mean()
    else:
        return (offset_loss_function(offset_activated, offset_targets) * hit_mask).mean()


def calculate_kld_loss(mu, log_var, free_bits=4.0):
    """Calculate KLD loss with free bits to prevent posterior collapse"""
    mu = mu.view(mu.shape[0], -1)
    log_var = log_var.view(log_var.shape[0], -1)
    log_var = torch.clamp(log_var, min=-10, max=10)

    # Standard KL divergence
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    # Apply free bits - only penalize KL above the threshold
    if free_bits > 0:
        kld_loss = torch.clamp(kld_loss - free_bits / kld_loss.shape[-1], min=0.0)

    return kld_loss.mean()


def batch_loop_step_based(dataloader_, forward_method, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
                          optimizer=None, starting_step=0, beta_scheduler=None,
                          scale_h_loss=1.0, scale_v_loss=1.0, scale_o_loss=1.0,
                          log_frequency=100, eval_callbacks=None, is_training=True, wandb_log=True):
    """
    Step-based batch loop with integrated logging and evaluation callbacks

    :param dataloader_: torch.utils.data.DataLoader for the dataset
    :param forward_method: function that handles model forward pass
    :param hit_loss_fn: loss function for hits
    :param velocity_loss_fn: loss function for velocities
    :param offset_loss_fn: loss function for offsets
    :param optimizer: optimizer for training (None for evaluation)
    :param starting_step: initial step count
    :param beta_scheduler: BetaAnnealingScheduler instance
    :param scale_h_loss: scaling factor for hit loss
    :param scale_v_loss: scaling factor for velocity loss
    :param scale_o_loss: scaling factor for offset loss
    :param log_frequency: how often to log metrics to wandb
    :param eval_callbacks: dict of evaluation functions to run at specified frequencies
    :param is_training: whether this is training or evaluation
    :param wandb_log: whether to log to wandb
    :return: metrics dict and final step count
    """
    import wandb

    # Prepare metric trackers
    step_metrics = {
        'loss_total': [], 'loss_h': [], 'loss_v': [], 'loss_o': [],
        'loss_KL': [], 'loss_recon': [], 'kl_beta': []
    }

    total_batches = len(dataloader_)
    current_step = starting_step

    # Default evaluation callbacks if none provided
    if eval_callbacks is None:
        eval_callbacks = {}

    for batch_count, batch_data in (pbar := tqdm.tqdm(enumerate(dataloader_), total=total_batches)):

        # Get current beta value
        if beta_scheduler is not None:
            kl_beta = beta_scheduler.get_beta(current_step)
        else:
            kl_beta = 1.0

        # Forward pass
        if optimizer is None:
            with torch.no_grad():
                h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)
        else:
            h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(target_outputs, int(target_outputs.shape[2] / 3), 2)

        # Compute losses
        batch_loss_h, hit_mask = calculate_hit_loss(
            hit_logits=h_logits, hit_targets=h_targets, hit_loss_function=hit_loss_fn)
        batch_loss_h = batch_loss_h * scale_h_loss

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn,
            hit_mask=hit_mask) * scale_v_loss

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn,
            hit_mask=hit_mask) * scale_o_loss

        batch_loss_KL = calculate_kld_loss(mu, log_var)
        batch_loss_KL_Beta_Scaled = batch_loss_KL * kl_beta

        # Backpropagation (only if training)
        if optimizer is not None:
            optimizer.zero_grad()
            (batch_loss_h + batch_loss_KL_Beta_Scaled).backward(retain_graph=True)
            batch_loss_v.backward(retain_graph=True)
            batch_loss_o.backward(retain_graph=True)
            optimizer.step()

        # Store metrics
        current_loss_h = batch_loss_h.item()
        current_loss_v = batch_loss_v.item()
        current_loss_o = batch_loss_o.item()
        current_loss_KL = batch_loss_KL.item()
        current_loss_recon = current_loss_h + current_loss_v + current_loss_o
        current_loss_total = current_loss_recon + batch_loss_KL_Beta_Scaled.item()

        # Add to running averages
        step_metrics['loss_h'].append(current_loss_h)
        step_metrics['loss_v'].append(current_loss_v)
        step_metrics['loss_o'].append(current_loss_o)
        step_metrics['loss_KL'].append(current_loss_KL)
        step_metrics['loss_recon'].append(current_loss_recon)
        step_metrics['loss_total'].append(current_loss_total)
        step_metrics['kl_beta'].append(kl_beta)

        # Step-based logging to wandb
        if wandb_log and current_step % log_frequency == 0:
            recent_steps = min(log_frequency, len(step_metrics['loss_total']))
            step_avg_metrics = {
                f"Step_Metrics/{'train' if is_training else 'eval'}_{k}": np.mean(v[-recent_steps:])
                for k, v in step_metrics.items()
            }
            step_avg_metrics['global_step'] = current_step
            wandb.log(step_avg_metrics, step=current_step)

        # Run evaluation callbacks at specified frequencies
        for callback_name, callback_config in eval_callbacks.items():
            if current_step % callback_config['frequency'] == 0 and current_step > 0:
                try:
                    callback_metrics = callback_config['function'](current_step)
                    if callback_metrics and wandb_log:
                        wandb.log(callback_metrics, step=current_step)
                except Exception as e:
                    logger.warning(f"Evaluation callback {callback_name} failed at step {current_step}: {e}")

        # Update progress bar
        pbar.set_postfix({
            "step": current_step,
            "beta": f"{kl_beta:.4f}",
            "l_total": f"{current_loss_total:.4f}",
            "l_h": f"{current_loss_h:.4f}",
            "l_v": f"{current_loss_v:.4f}",
            "l_o": f"{current_loss_o:.4f}",
            "l_KL": f"{current_loss_KL:.4f}",
        })

        # Only increment step counter during training
        if is_training:
            current_step += 1
            # Only increment beta scheduler during training
            if beta_scheduler is not None:
                beta_scheduler.step()

    # Return aggregated metrics for the entire pass
    aggregated_metrics = {
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_total_rec_w_kl": np.mean(step_metrics['loss_total']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_h": np.mean(step_metrics['loss_h']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_v": np.mean(step_metrics['loss_v']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_o": np.mean(step_metrics['loss_o']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_KL": np.mean(step_metrics['loss_KL']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_recon": np.mean(step_metrics['loss_recon'])
    }

    return aggregated_metrics, current_step


def train_loop_step_based(train_dataloader, forward_method, optimizer, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
                          starting_step, beta_scheduler, scale_h_loss, scale_v_loss, scale_o_loss,
                          log_frequency=100, eval_callbacks=None):
    """Step-based training loop"""
    return batch_loop_step_based(
        dataloader_=train_dataloader,
        forward_method=forward_method,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        optimizer=optimizer,
        starting_step=starting_step,
        beta_scheduler=beta_scheduler,
        scale_h_loss=scale_h_loss,
        scale_v_loss=scale_v_loss,
        scale_o_loss=scale_o_loss,
        log_frequency=log_frequency,
        eval_callbacks=eval_callbacks,
        is_training=True
    )


def test_loop_step_based(test_dataloader, forward_method, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
                         starting_step, beta_scheduler, scale_h_loss, scale_v_loss, scale_o_loss,
                         log_frequency=100):
    """Step-based test loop - doesn't increment step counter or log step-by-step"""
    return batch_loop_step_based(
        dataloader_=test_dataloader,
        forward_method=forward_method,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        optimizer=None,
        starting_step=starting_step,
        beta_scheduler=beta_scheduler,
        scale_h_loss=scale_h_loss,
        scale_v_loss=scale_v_loss,
        scale_o_loss=scale_o_loss,
        log_frequency=log_frequency,
        is_training=False,
        wandb_log=False  # No step-by-step logging during test
    )