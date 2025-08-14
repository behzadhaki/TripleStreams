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

    """Beta annealing scheduler that works with training steps"""

    def __init__(self, total_steps, period_steps, rise_ratio, start_first_rise_at_step=0, beta_level=1.0):
        self.beta_curve = self.generate_beta_curve_step_based(
            total_steps, period_steps, rise_ratio, start_first_rise_at_step
        )
        self.beta_level = beta_level
        self.current_step = 0

    def generate_beta_curve_step_based(self, total_steps, period_steps, rise_ratio, start_first_rise_at_step=0):
        """Generate a beta curve based on training steps"""

        def f(x, K):
            if x == 0:
                return 0
            elif x == K:
                return 1
            else:
                return 1 / (1 + np.exp(-10 * (x - K / 2) / K))

        def generate_rising_curve(K):
            curve = []
            for i in range(K):
                curve.append(f(i, K - 1))
            return np.array(curve)

        def generate_single_beta_cycle(period, rise_ratio):
            cycle = np.ones(period)
            curve_steps = int(period * rise_ratio)
            rising_curve = generate_rising_curve(curve_steps)
            cycle[:rising_curve.shape[0]] = rising_curve[:cycle.shape[0]]
            return cycle

        beta_curve = np.zeros(start_first_rise_at_step)
        effective_steps = total_steps - start_first_rise_at_step
        n_cycles = np.ceil(effective_steps / period_steps)

        single_cycle = generate_single_beta_cycle(period_steps, rise_ratio)

        for c in np.arange(int(n_cycles)):
            beta_curve = np.append(beta_curve, single_cycle)

        return beta_curve[:total_steps]

    def get_beta(self, step=None):
        """Get beta value for current or specified step"""
        if step is None:
            step = self.current_step

        if step >= len(self.beta_curve):
            return self.beta_level

        return self.beta_curve[step] * self.beta_level

    def step(self):
        """Increment the step counter"""
        self.current_step += 1
        return self.get_beta()


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
    :param beta_scheduler: StepBasedBetaScheduler instance
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