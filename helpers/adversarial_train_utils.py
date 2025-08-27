import torch
import numpy as np
import tqdm
import wandb
from logging import getLogger

logger = getLogger("adversarial_train_utils")


def extract_patterns_for_discriminator(h_logits, v_logits, o_logits, use_sigmoid=True):
    """
    Extract patterns in the format expected by discriminator

    Args:
        h_logits, v_logits, o_logits: Model outputs [batch, seq_len, features]
        use_sigmoid: Whether to apply sigmoid to hits (for generated patterns)

    Returns:
        patterns: [batch, seq_len, 3] patterns for discriminator
    """
    if use_sigmoid:
        h = torch.sigmoid(h_logits)
    else:
        h = h_logits

    # Normalize velocity and offset to [0, 1] and [-1, 1] ranges respectively
    v = torch.tanh(v_logits) * 0.5 + 0.5  # Map to [0, 1]
    o = torch.tanh(o_logits)  # Keep in [-1, 1]

    # Concatenate to form complete patterns
    patterns = torch.cat([h, v, o], dim=-1)
    return patterns


def batch_loop_step_based_adversarial(dataloader_, forward_method, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
                                      adversarial_trainer, optimizer=None, starting_step=0, beta_scheduler=None,
                                      scale_h_loss=1.0, scale_v_loss=1.0, scale_o_loss=1.0,
                                      adversarial_weight=0.1, warmup_steps=1000,
                                      log_frequency=100, eval_callbacks=None, is_training=True, wandb_log=True):
    """
    Enhanced batch loop with adversarial training for diversity improvement

    Key additions:
    - Adversarial loss to encourage realistic and diverse pattern generation
    - Discriminator training on real vs generated patterns
    - Feature matching for training stability
    - Gradual adversarial loss introduction via warmup

    Args:
        adversarial_trainer: AdversarialTrainer instance
        adversarial_weight: Weight for adversarial loss component
        warmup_steps: Steps before adversarial training begins
        ... (other parameters same as original)
    """
    # Prepare metric trackers - add adversarial tracking
    step_metrics = {
        'loss_total': [], 'loss_h': [], 'loss_v': [], 'loss_o': [],
        'loss_KL': [], 'loss_recon': [], 'kl_beta': [],
        'loss_adversarial': [], 'loss_feature_matching': [], 'loss_diversity_reg': [],
        'discriminator_loss': [], 'real_score': [], 'fake_score': []
    }

    total_batches = len(dataloader_)
    current_step = starting_step

    if eval_callbacks is None:
        eval_callbacks = {}

    for batch_count, batch_data in (pbar := tqdm.tqdm(enumerate(dataloader_), total=total_batches)):

        # Get current beta value
        if beta_scheduler is not None:
            kl_beta = beta_scheduler.get_beta(current_step)
        else:
            kl_beta = 1.0

        # Determine adversarial weight with warmup
        if current_step < warmup_steps:
            current_adversarial_weight = 0.0
        else:
            # Gradual ramp-up over warmup_steps
            ramp_progress = min(1.0, (current_step - warmup_steps) / warmup_steps)
            current_adversarial_weight = adversarial_weight * ramp_progress

        # Forward pass
        if optimizer is None:
            with torch.no_grad():
                h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)
        else:
            h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(target_outputs, int(target_outputs.shape[2] / 3), 2)

        # Standard VAE losses
        from helpers.train_utils import calculate_hit_loss_with_diversity, calculate_velocity_loss, \
            calculate_offset_loss, calculate_kld_loss

        batch_loss_h, hit_mask = calculate_hit_loss_with_diversity(
            hit_logits=h_logits,
            hit_targets=h_targets,
            hit_loss_function=hit_loss_fn,
            use_hit_mask=True
        )
        batch_loss_h = batch_loss_h * scale_h_loss

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn,
            hit_mask=hit_mask) * scale_v_loss

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn,
            hit_mask=hit_mask) * scale_o_loss

        batch_loss_KL = calculate_kld_loss(mu, log_var)
        batch_loss_KL_Beta_Scaled = batch_loss_KL * kl_beta

        # Adversarial training
        adversarial_loss = torch.tensor(0.0, device=h_logits.device)
        feature_matching_loss = torch.tensor(0.0, device=h_logits.device)
        diversity_reg_loss = torch.tensor(0.0, device=h_logits.device)
        discriminator_loss = torch.tensor(0.0, device=h_logits.device)
        real_score = 0.0
        fake_score = 0.0

        if current_adversarial_weight > 0 and is_training:
            # Extract patterns for discriminator
            generated_patterns = extract_patterns_for_discriminator(h_logits, v_logits, o_logits, use_sigmoid=True)
            real_patterns = extract_patterns_for_discriminator(h_targets, v_targets, o_targets, use_sigmoid=False)

            # Train discriminator
            if adversarial_trainer.should_train_discriminator():
                d_metrics = adversarial_trainer.train_discriminator(real_patterns, generated_patterns)
                discriminator_loss = torch.tensor(d_metrics['discriminator_loss'])
                real_score = d_metrics['real_score']
                fake_score = d_metrics['fake_score']

            # Compute generator adversarial loss
            gen_adv_loss, adv_components = adversarial_trainer.compute_generator_adversarial_loss(
                generated_patterns, real_patterns
            )

            adversarial_loss = gen_adv_loss * current_adversarial_weight

            # Extract individual components for logging
            if 'feature_matching' in adv_components:
                feature_matching_loss = torch.tensor(adv_components['feature_matching'])
            if 'diversity_reg' in adv_components:
                diversity_reg_loss = torch.tensor(adv_components['diversity_reg'])

        # Combined loss
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss = (batch_loss_h + batch_loss_v + batch_loss_o +
                          batch_loss_KL_Beta_Scaled + adversarial_loss)
            total_loss.backward()

            # Gradient clipping to stabilize adversarial training
            torch.nn.utils.clip_grad_norm_(adversarial_trainer.discriminator.parameters(), max_norm=1.0)

            optimizer.step()

        # Store metrics
        current_loss_h = batch_loss_h.item()
        current_loss_v = batch_loss_v.item()
        current_loss_o = batch_loss_o.item()
        current_loss_KL = batch_loss_KL.item()
        current_loss_recon = current_loss_h + current_loss_v + current_loss_o
        current_loss_adversarial = adversarial_loss.item()
        current_loss_total = (current_loss_recon + batch_loss_KL_Beta_Scaled.item() +
                              current_loss_adversarial)

        # Add to running averages
        step_metrics['loss_h'].append(current_loss_h)
        step_metrics['loss_v'].append(current_loss_v)
        step_metrics['loss_o'].append(current_loss_o)
        step_metrics['loss_KL'].append(current_loss_KL)
        step_metrics['loss_recon'].append(current_loss_recon)
        step_metrics['loss_total'].append(current_loss_total)
        step_metrics['kl_beta'].append(kl_beta)
        step_metrics['loss_adversarial'].append(current_loss_adversarial)
        step_metrics['loss_feature_matching'].append(feature_matching_loss.item())
        step_metrics['loss_diversity_reg'].append(diversity_reg_loss.item())
        step_metrics['discriminator_loss'].append(discriminator_loss.item())
        step_metrics['real_score'].append(real_score)
        step_metrics['fake_score'].append(fake_score)

        # Step-based logging to wandb
        if wandb_log and current_step % log_frequency == 0:
            recent_steps = min(log_frequency, len(step_metrics['loss_total']))
            step_avg_metrics = {
                f"Step_Metrics/{'train' if is_training else 'eval'}_{k}": np.mean(v[-recent_steps:])
                for k, v in step_metrics.items()
            }
            step_avg_metrics['global_step'] = current_step
            step_avg_metrics['adversarial_weight'] = current_adversarial_weight
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
            "adv_w": f"{current_adversarial_weight:.3f}",
            "l_total": f"{current_loss_total:.4f}",
            "l_h": f"{current_loss_h:.4f}",
            "l_adv": f"{current_loss_adversarial:.4f}",
            "d_loss": f"{discriminator_loss.item():.4f}",
            "real": f"{real_score:.3f}",
            "fake": f"{fake_score:.3f}",
        })

        # Only increment step counter during training
        if is_training:
            current_step += 1
            if beta_scheduler is not None:
                beta_scheduler.step()
            adversarial_trainer.step()

    # Return aggregated metrics for the entire pass
    aggregated_metrics = {
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_total_rec_w_kl": np.mean(step_metrics['loss_total']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_h": np.mean(step_metrics['loss_h']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_v": np.mean(step_metrics['loss_v']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_o": np.mean(step_metrics['loss_o']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_KL": np.mean(step_metrics['loss_KL']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_recon": np.mean(step_metrics['loss_recon']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/loss_adversarial": np.mean(
            step_metrics['loss_adversarial']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/discriminator_loss": np.mean(
            step_metrics['discriminator_loss']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/real_score": np.mean(step_metrics['real_score']),
        f"{'Train' if is_training else 'Test'}_Epoch_Metrics/fake_score": np.mean(step_metrics['fake_score']),
    }

    return aggregated_metrics, current_step


def train_loop_step_based_adversarial(train_dataloader, forward_method, optimizer, hit_loss_fn, velocity_loss_fn,
                                      offset_loss_fn, adversarial_trainer, starting_step, beta_scheduler,
                                      scale_h_loss, scale_v_loss, scale_o_loss, adversarial_weight=0.1,
                                      warmup_steps=1000, log_frequency=100, eval_callbacks=None):
    """Step-based training loop with adversarial training"""
    return batch_loop_step_based_adversarial(
        dataloader_=train_dataloader,
        forward_method=forward_method,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        adversarial_trainer=adversarial_trainer,
        optimizer=optimizer,
        starting_step=starting_step,
        beta_scheduler=beta_scheduler,
        scale_h_loss=scale_h_loss,
        scale_v_loss=scale_v_loss,
        scale_o_loss=scale_o_loss,
        adversarial_weight=adversarial_weight,
        warmup_steps=warmup_steps,
        log_frequency=log_frequency,
        eval_callbacks=eval_callbacks,
        is_training=True
    )


def test_loop_step_based_adversarial(test_dataloader, forward_method, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
                                     adversarial_trainer, starting_step, beta_scheduler, scale_h_loss, scale_v_loss,
                                     scale_o_loss, log_frequency=100):
    """Step-based test loop with adversarial components (but no discriminator training)"""
    return batch_loop_step_based_adversarial(
        dataloader_=test_dataloader,
        forward_method=forward_method,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        adversarial_trainer=adversarial_trainer,
        optimizer=None,
        starting_step=starting_step,
        beta_scheduler=beta_scheduler,
        scale_h_loss=scale_h_loss,
        scale_v_loss=scale_v_loss,
        scale_o_loss=scale_o_loss,
        adversarial_weight=0.0,  # No adversarial loss during testing
        warmup_steps=0,
        log_frequency=log_frequency,
        is_training=False,
        wandb_log=False
    )


def save_model_checkpoint_enhanced_adversarial(model, optimizer, beta_scheduler, adversarial_trainer, step,
                                               save_dir, wandb_project, run_name, run_id):
    """Enhanced model saving with adversarial training components"""
    if step > 0:
        import wandb
        import os

        model_artifact = wandb.Artifact(f'model_step_{step}', type='model')
        model_path = f"{save_dir}/{wandb_project}/{run_name}_{run_id}/step_{step}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save comprehensive checkpoint including discriminator
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'discriminator_state_dict': adversarial_trainer.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': adversarial_trainer.discriminator_optimizer.state_dict(),
            'adversarial_trainer_step': adversarial_trainer.step_count,
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
        logger.info(f"Enhanced adversarial checkpoint saved to {model_path}")

    return {}


def setup_resumable_training_adversarial(config, model, optimizer, adversarial_trainer, beta_scheduler=None,
                                         wandb_run=None):
    """
    Setup training resumption from checkpoint with adversarial training components

    Args:
        config: Current configuration
        model: Model instance
        optimizer: Optimizer instance
        adversarial_trainer: AdversarialTrainer instance
        beta_scheduler: Beta scheduler instance (optional)
        wandb_run: Active wandb run instance (optional)

    Returns:
        starting_step, updated_model, updated_optimizer, updated_adversarial_trainer, updated_beta_scheduler
    """
    starting_step = 0

    if getattr(config, 'resume_from_checkpoint', False):
        if not getattr(config, 'checkpoint_artifact_name', None):
            raise ValueError("Must specify checkpoint_artifact_name when resuming")

        logger.info(f"Resuming from adversarial checkpoint: {config.checkpoint_artifact_name}")

        # Load checkpoint using the existing method from train_utils
        from helpers.train_utils import load_checkpoint_from_wandb, validate_flexcontrol_architecture_compatibility

        checkpoint_data, checkpoint_step = load_checkpoint_from_wandb(
            config.wandb_project,
            getattr(config, 'checkpoint_wandb_run_id', None),
            config.checkpoint_artifact_name,
            getattr(config, 'checkpoint_step', None),
            wandb_run
        )

        # Get old configuration from WandB for architecture validation
        if getattr(config, 'checkpoint_wandb_run_id', None):
            import wandb
            api = wandb.Api()
            old_run = api.run(f"behzadhaki/{config.wandb_project}/{config.checkpoint_wandb_run_id}")
            old_config = old_run.config

            # Validate architecture compatibility
            validate_flexcontrol_architecture_compatibility(old_config, config)
        else:
            logger.warning("No checkpoint_wandb_run_id provided - skipping architecture validation")

        # Load model and optimizer states
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

            # Load discriminator state if available
            if 'discriminator_state_dict' in checkpoint_data and not getattr(config, 'reset_discriminator', False):
                try:
                    adversarial_trainer.discriminator.load_state_dict(checkpoint_data['discriminator_state_dict'])
                    logger.info("Loaded discriminator state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load discriminator state: {e}. Continuing with fresh discriminator.")

            # Load discriminator optimizer state if available
            if 'discriminator_optimizer_state_dict' in checkpoint_data and not getattr(config, 'reset_discriminator',
                                                                                       False):
                try:
                    adversarial_trainer.discriminator_optimizer.load_state_dict(
                        checkpoint_data['discriminator_optimizer_state_dict'])
                    logger.info("Loaded discriminator optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(
                        f"Could not load discriminator optimizer state: {e}. Continuing with fresh optimizer.")

            # Load adversarial trainer step count
            if 'adversarial_trainer_step' in checkpoint_data:
                adversarial_trainer.step_count = checkpoint_data['adversarial_trainer_step']
                logger.info(f"Restored adversarial trainer step count: {adversarial_trainer.step_count}")

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

        starting_step = checkpoint_step
        logger.info(f"Resuming adversarial training from step {starting_step}")

        # Log the resumption details
        import wandb
        resume_info = {
            "resumed_from_artifact": config.checkpoint_artifact_name,
            "resumed_from_step": starting_step,
            "reset_optimizer": getattr(config, 'reset_optimizer', False),
            "reset_discriminator": getattr(config, 'reset_discriminator', False),
            "reset_beta_scheduler": getattr(config, 'reset_beta_scheduler', False)
        }
        if getattr(config, 'checkpoint_wandb_run_id', None):
            resume_info["resumed_from_run"] = config.checkpoint_wandb_run_id

        wandb.log(resume_info)

    else:
        logger.info("Starting adversarial training from scratch")

    return starting_step, model, optimizer, adversarial_trainer, beta_scheduler