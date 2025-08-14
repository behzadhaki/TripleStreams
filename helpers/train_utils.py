import torch
import numpy as np
import tqdm

from logging import getLogger
logger = getLogger("train_utils")
logger.setLevel("DEBUG")


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification problems to address class imbalance.

    Focal Loss=−α(1−pt)^γ * log(pt)

    Attributes:
    -----------
    alpha : float
        A weighting factor for balancing the importance of positive/negative classes.
        Typically in the range [0, 1]. Higher values of alpha give more weight to the positive class.

    gamma : float
        The focusing parameter. Gamma >= 0. Reduces the relative loss for well-classified examples,
        putting more focus on hard, misclassified examples. Higher values of gamma make the model focus more
        on hard examples.

    Methods:
    --------
    forward(inputs, targets):
        Compute the focal loss for given inputs and targets.

    Parameters:
    -----------
    inputs : torch.Tensor
        The logits (raw model outputs) of shape (N, *) where * means any number of additional dimensions.
        For binary classification, this is typically of shape (N, 1).

    targets : torch.Tensor
        The ground truth values (labels) with the same shape as inputs.

    Returns:
    --------
    torch.Tensor
        The computed focal loss.

    Example:
    --------
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for computing the focal loss.
        """
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Computes the probability of the correct class (Prevents nans when probability 0)
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
        # Flatten label and prediction tensors
        h_probs = torch.sigmoid(h_logits).reshape(-1)
        targets = h_targets.reshape(-1)

        intersection = (h_probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (h_probs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def generate_beta_curve(n_epochs, period_epochs, rise_ratio, start_first_rise_at_epoch=0):
    """
    Generate a beta curve for the given parameters

    Args:
        n_epochs:            The number of epochs to generate the curve for
        period_epochs:       The period of the curve in epochs (for multiple cycles)
        rise_ratio:         The ratio of the period to be used for the rising part of the curve
        start_first_rise_at_epoch:  The epoch to start the first rise at (useful for warmup)

    Returns:

    """
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

        curve_steps_in_epochs = int(period * rise_ratio)

        rising_curve = generate_rising_curve(curve_steps_in_epochs)

        cycle[:rising_curve.shape[0]] = rising_curve[:cycle.shape[0]]

        return cycle

    beta_curve = np.zeros((start_first_rise_at_epoch))
    effective_epochs = n_epochs - start_first_rise_at_epoch
    n_cycles = np.ceil(effective_epochs / period_epochs)

    single_cycle = generate_single_beta_cycle(period_epochs, rise_ratio)

    for c in np.arange(n_cycles):
        beta_curve = np.append(beta_curve, single_cycle)

    return beta_curve[:n_epochs]


def calculate_hit_loss(hit_logits, hit_targets, hit_loss_function):
    assert isinstance(hit_loss_function, torch.nn.BCEWithLogitsLoss) or isinstance(hit_loss_function, FocalLoss) or isinstance(hit_loss_function, DiceLoss), f"hit_loss_function must be an instance of torch.nn.BCEWithLogitsLoss or FocalLoss or DiceLoss. Got {type(hit_loss_function)}"
    loss_h = hit_loss_function(hit_logits, hit_targets)           # batch, time steps, voices (10 is a scaling factor to match the other losses)
    hit_mask = None
    if hit_loss_function.reduction == 'none':
        # put more weight on the hits
        hit_mask = (hit_targets > 0).float() * 3 + 1 # hits weighted almost 10 times more than the misses (in reality, 4 to 1 ratio)
        loss_h = loss_h * hit_mask
        loss_h = loss_h.mean()

    return loss_h, hit_mask       # batch_size,  time_steps, n_voices


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


def calculate_kld_loss(mu, log_var):
    """ calculate the KLD loss for the given mu and log_var values against a standard normal distribution
    :param mu:  (torch.Tensor)  the mean values of the latent space
    :param log_var: (torch.Tensor)  the log variance values of the latent space
    :return:    kld_loss (torch.Tensor)  the KLD loss value (unreduced) shape: (batch_size,  time_steps, n_voices)

    """
    mu = mu.view(mu.shape[0], -1)
    log_var = log_var.view(log_var.shape[0], -1)
    log_var = torch.clamp(log_var, min=-10, max=10)
    kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    return kld_loss.mean()     # batch_size,  time_steps, n_voices


def batch_loop(dataloader_, forward_method, hit_loss_fn, velocity_loss_fn,  offset_loss_fn,
               optimizer=None, starting_step=None, kl_beta=1.0, scale_h_loss=1.0, scale_v_loss=1.0, scale_o_loss=1.0):

    """
    This function iteratively loops over the given dataloader and calculates the loss for each batch. If an optimizer is
    provided, it will also perform the backward pass and update the model parameters. The loss values are accumulated
    and returned at the end of the loop.

    **Can be used for both training and testing. In testing however, backpropagation will not be performed**


    :param dataloader_:     (torch.utils.data.DataLoader)  dataloader for the dataset
    :param forward_method:  (function)  the forward method of the model (takes care of io extraction and model forward pass, also returns the targets)
    :param hit_loss_fn:     (str)  "bce"
    :param velocity_loss_fn:    (str)  "bce"
    :param offset_loss_fn:  (str)  "HuberLoss"
    :param optimizer:   (torch.optim.Optimizer)  the optimizer to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KLD loss
    :param scale_h_loss: (float)  the scaling factor for the hit loss
    :param scale_v_loss: (float)  the scaling factor for the velocity loss
    :param scale_o_loss: (float)  the scaling factor for the offset loss
    :return:    (dict)  a dictionary containing the loss values for the current batch

                metrics = {
                    "loss_total": np.mean(loss_total),
                    "loss_h": np.mean(loss_h),
                    "loss_v": np.mean(loss_v),
                    "loss_o": np.mean(loss_o),
                    "loss_KL": np.mean(loss_KL)}

                (int)  the current step of the optimizer (if provided)
    """

    # Prepare the metric trackers for the new epoch
    # ------------------------------------------------------------------------------------------
    loss_total, loss_recon, loss_h, loss_v, loss_o, loss_KL, loss_KL_beta_scaled = [], [], [], [], [], [], []

    # Iterate over batches
    # ------------------------------------------------------------------------------------------
    total_batches = len(dataloader_)
    for batch_count, (batch_data) in (pbar := tqdm.tqdm(enumerate(dataloader_), total=total_batches)):

        if optimizer is None:
            with torch.no_grad():
                # Set the model to evaluation mode
                h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)
        else:
            # Set the model to training mode
            h_logits, v_logits, o_logits, mu, log_var, latent_z, target_outputs = forward_method(batch_data)

        # Prepare targets for loss calculation
        h_targets, v_targets, o_targets = torch.split(target_outputs, int(target_outputs.shape[2] / 3), 2)

        # Compute losses for the model
        # ---------------------------------------------------------------------------------------
        batch_loss_h, hit_mask = calculate_hit_loss(
            hit_logits=h_logits, hit_targets=h_targets, hit_loss_function=hit_loss_fn)
        batch_loss_h = batch_loss_h * scale_h_loss

        batch_loss_v = calculate_velocity_loss(
            vel_logits=v_logits, vel_targets=v_targets, vel_loss_function=velocity_loss_fn, hit_mask=hit_mask) * scale_v_loss

        batch_loss_o = calculate_offset_loss(
            offset_logits=o_logits, offset_targets=o_targets, offset_loss_function=offset_loss_fn, hit_mask=hit_mask) * scale_o_loss

        batch_loss_KL = calculate_kld_loss(mu, log_var)

        batch_loss_KL_Beta_Scaled = batch_loss_KL * kl_beta


        # Backpropagation and optimization step (if training)
        # ---------------------------------------------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
            (batch_loss_h + batch_loss_KL_Beta_Scaled).backward(retain_graph=True)
            batch_loss_v.backward(retain_graph=True)
            batch_loss_o.backward(retain_graph=True)
            optimizer.step()


        # Update the per batch loss trackers
        # -----------------------------------------------------------------
        loss_h.append(batch_loss_h.item())
        loss_v.append(batch_loss_v.item())
        loss_o.append(batch_loss_o.item())
        loss_KL.append(batch_loss_KL.item())
        loss_KL_beta_scaled.append(batch_loss_KL_Beta_Scaled.item())
        loss_recon.append(loss_h[-1] + loss_v[-1] + loss_o[-1])
        loss_total.append(loss_recon[-1] + loss_KL_beta_scaled[-1])

        # Debugging information
        # ---------------------------------------------------------------------------------------
        assert not np.isnan(loss_total[-1]), print(batch_data[-2])

        # Update the progress bar
        # ---------------------------------------------------------------------------------------
        pbar.set_postfix(
            {"l_total_recon_kl": f"{np.mean(loss_total):.4f}",
                "l_recon": f"{np.mean(loss_recon):.4f}",
                "l_h": f"{np.mean(loss_h):.4f}",
                "l_v": f"{np.mean(loss_v):.4f}",
                "l_o": f"{np.mean(loss_o):.4f}",
                "l_KL": f"{np.mean(loss_KL):.4f}",
                })

        # Increment the step counter
        # ---------------------------------------------------------------------------------------
        if starting_step is not None:
            starting_step += 1

    metrics = {
        "loss_total_rec_w_kl": np.mean(loss_total),
        "loss_h": np.mean(loss_h),
        "loss_v": np.mean(loss_v),
        "loss_o": np.mean(loss_o),
        "loss_KL": np.mean(loss_KL),
        "loss_KL_beta_scaled": np.mean(loss_KL_beta_scaled),
        "loss_recon": np.mean(loss_recon)
    }

    if starting_step is not None:
        return metrics, starting_step
    else:
        return metrics


def train_loop(train_dataloader, forward_method,
               optimizer, hit_loss_fn, velocity_loss_fn, offset_loss_fn,
               starting_step, kl_beta, scale_h_loss, scale_v_loss, scale_o_loss):
    """
    This function performs the training loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward and backward pass for each batch. The loss values are accumulated and the average is
    returned at the end of the loop.

    :param train_dataloader:    (torch.utils.data.DataLoader)  dataloader for the training dataset
    :param model:  (GenreGlobalDensityWithVoiceMutesVAE)  the model
    :param batch_data_extractor:  (function)  a function to extract the batch data
    :param optimizer:  (torch.optim.Optimizer)  the optimizer to use for the model (sgd or adam)
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:  (torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:      (torch.nn.HuberLoss)
    :param device:  (str)  the device to use for the model
    :param starting_step:   (int)  the starting step for the optimizer
    :param kl_beta: (float)  the beta value for the KL loss (maybe flat or annealed)

    :return:    (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "train/loss_total": np.mean(loss_total),
                    "train/loss_h": np.mean(loss_h),
                    "train/loss_v": np.mean(loss_v),
                    "train/loss_o": np.mean(loss_o),
                    "train/loss_KL": np.mean(loss_KL)}
    """
    # Run the batch loop
    metrics, starting_step = batch_loop(
        dataloader_=train_dataloader,
        forward_method=forward_method,
        hit_loss_fn=hit_loss_fn,
        velocity_loss_fn=velocity_loss_fn,
        offset_loss_fn=offset_loss_fn,
        optimizer=optimizer,
        starting_step=starting_step,
        kl_beta=kl_beta,
        scale_h_loss=scale_h_loss,
        scale_v_loss=scale_v_loss,
        scale_o_loss=scale_o_loss)

    metrics = {f"Loss_Criteria/{key}_train": value for key, value in sorted(metrics.items())}
    return metrics, starting_step


def test_loop(test_dataloader, forward_method,
              hit_loss_fn, velocity_loss_fn, offset_loss_fn,
              kl_beta, scale_h_loss, scale_v_loss, scale_o_loss):
    """
    This function performs the test loop for the given model and dataloader. It will iterate over the dataloader
    and perform the forward pass for each batch. The loss values are accumulated and the average is returned at the end
    of the loop.

    :param test_dataloader:   (torch.utils.data.DataLoader)  dataloader for the test dataset
    :param model:  (GenreGlobalDensityWithVoiceMutesVAE)  the model
    :param batch_data_extractor:  (function)  a function to extract the batch data
    :param hit_loss_fn:     ("dice" or torch.nn.BCEWithLogitsLoss)
    :param velocity_loss_fn:    (torch.nn.BCEWithLogitsLoss)
    :param offset_loss_fn:    (torch.nn.HuberLoss)
    :param device:  (str)  the device to use for the model
    :param kl_beta: (float)  the beta value for the KL loss

    :return:   (dict)  a dictionary containing the loss values for the current batch

            metrics = {
                    "test/loss_total": np.mean(loss_total),
                    "test/loss_h": np.mean(loss_h),
                    "test/loss_v": np.mean(loss_v),
                    "test/loss_o": np.mean(loss_o),
                    "test/loss_KL": np.mean(loss_KL)}
    """

    with torch.no_grad():
        # Run the batch loop
        metrics = batch_loop(
            dataloader_=test_dataloader,
            forward_method=forward_method,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            optimizer=None,
            kl_beta=kl_beta,
            scale_h_loss=scale_h_loss,
            scale_v_loss=scale_v_loss,
            scale_o_loss=scale_o_loss)

    metrics = {f"Loss_Criteria/{key}_test": value for key, value in sorted(metrics.items())}

    return metrics




