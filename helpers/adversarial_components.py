import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

logger = getLogger("adversarial_components")


class PatternDiscriminator(nn.Module):
    """
    Discriminator that distinguishes between real and generated musical patterns.
    Focuses on temporal and rhythmic characteristics to encourage diversity.
    """

    def __init__(self, config):
        super(PatternDiscriminator, self).__init__()

        # Input: [batch, 32, 3] (steps, features: hit/velocity/offset)
        input_dim = config.get('max_len', 32) * config.get('embedding_size_tgt', 9)  # 32 * 9 = 288
        hidden_dim = config.get('discriminator_hidden_dim', 256)
        dropout = config.get('discriminator_dropout', 0.3)

        self.flatten_input = True

        # Main discriminator network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Feature extractors for different aspects
        self.rhythm_extractor = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=3, padding=1),  # Changed from 3 to 9
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, 64)
        )

        self.use_feature_matching = config.get('use_feature_matching', True)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [batch, 32, 3] musical patterns
            return_features: whether to return intermediate features for feature matching

        Returns:
            logits: [batch, 1] real/fake prediction
            features: dict of intermediate features (if return_features=True)
        """
        batch_size = x.shape[0]

        # Extract rhythm features
        rhythm_features = self.rhythm_extractor(x.transpose(1, 2))

        # Flatten main input
        x_flat = x.view(batch_size, -1)

        # Get intermediate features for feature matching
        features = {}
        if return_features or self.use_feature_matching:
            h1 = F.leaky_relu(self.network[0](x_flat), 0.2)
            features['layer_1'] = h1
            h2 = F.leaky_relu(self.network[3](self.network[2](h1)), 0.2)
            features['layer_2'] = h2
            h3 = F.leaky_relu(self.network[6](self.network[5](h2)), 0.2)
            features['layer_3'] = h3
            features['rhythm'] = rhythm_features

            logits = self.network[-1](h3)
        else:
            logits = self.network(x_flat)

        if return_features:
            return logits, features
        return logits

    def init_weights(self, init_range=0.02):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, 0.0, init_range)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class DiversityDiscriminator(nn.Module):
    """
    Alternative discriminator that specifically focuses on pattern diversity
    within batches to combat mode collapse.
    """

    def __init__(self, config):
        super(DiversityDiscriminator, self).__init__()

        max_len = config.get('max_len', 32)
        hidden_dim = config.get('diversity_discriminator_hidden_dim', 128)

        # Encoder for individual patterns
        self.pattern_encoder = nn.Sequential(
            nn.Linear(max_len * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)  # Compact representation
        )

        # Diversity analyzer - takes batch of encoded patterns
        self.diversity_analyzer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 32, 3] musical patterns

        Returns:
            diversity_score: [1] scalar indicating batch diversity (0=low, 1=high)
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Encode each pattern
        encoded = self.pattern_encoder(x_flat)  # [batch, 32]

        # Compute pairwise distances within batch
        distances = torch.cdist(encoded, encoded, p=2)  # [batch, batch]

        # Average distance (excluding diagonal)
        mask = ~torch.eye(batch_size, dtype=bool, device=x.device)
        avg_distance = distances[mask].mean()

        # Convert to diversity score
        diversity_input = avg_distance.unsqueeze(0).unsqueeze(0)  # [1, 1]
        diversity_score = self.diversity_analyzer(
            torch.cat([diversity_input, torch.tensor([[batch_size]], device=x.device, dtype=torch.float)], dim=1)
        )

        return diversity_score.squeeze()


def adversarial_generator_loss(fake_logits, loss_type='standard'):
    """
    Generator loss for adversarial training

    Args:
        fake_logits: [batch, 1] discriminator predictions for generated samples
        loss_type: 'standard', 'least_squares', or 'wasserstein'
    """
    if loss_type == 'standard':
        # Standard GAN loss: generator wants discriminator to output 1 (real)
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    elif loss_type == 'least_squares':
        # LSGAN loss
        return 0.5 * torch.mean((fake_logits - 1.0) ** 2)
    elif loss_type == 'wasserstein':
        # WGAN loss
        return -torch.mean(fake_logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def adversarial_discriminator_loss(real_logits, fake_logits, loss_type='standard'):
    """
    Discriminator loss for adversarial training

    Args:
        real_logits: [batch, 1] discriminator predictions for real samples
        fake_logits: [batch, 1] discriminator predictions for generated samples
        loss_type: 'standard', 'least_squares', or 'wasserstein'
    """
    if loss_type == 'standard':
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) * 0.5
    elif loss_type == 'least_squares':
        real_loss = 0.5 * torch.mean((real_logits - 1.0) ** 2)
        fake_loss = 0.5 * torch.mean(fake_logits ** 2)
        return real_loss + fake_loss
    elif loss_type == 'wasserstein':
        return torch.mean(fake_logits) - torch.mean(real_logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def feature_matching_loss(real_features, fake_features, feature_weights=None):
    """
    Feature matching loss to stabilize training and encourage diversity

    Args:
        real_features: dict of features from real samples
        fake_features: dict of features from generated samples
        feature_weights: dict of weights for different feature layers
    """
    if feature_weights is None:
        feature_weights = {}

    total_loss = 0
    count = 0

    for key in real_features.keys():
        if key in fake_features:
            weight = feature_weights.get(key, 1.0)
            loss = F.mse_loss(fake_features[key], real_features[key].detach())
            total_loss += weight * loss
            count += 1

    return total_loss / max(count, 1)


def diversity_regularization_loss(generated_patterns, diversity_weight=1.0):
    """
    Encourage diversity within generated batches

    Args:
        generated_patterns: [batch, seq_len, features] generated patterns
        diversity_weight: weight for diversity term
    """
    batch_size = generated_patterns.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=generated_patterns.device)

    # Flatten patterns
    patterns_flat = generated_patterns.view(batch_size, -1)

    # Compute pairwise similarities
    normalized_patterns = F.normalize(patterns_flat, dim=1)
    similarity_matrix = torch.mm(normalized_patterns, normalized_patterns.t())

    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(batch_size, dtype=bool, device=generated_patterns.device)
    similarities = similarity_matrix[mask]

    # Penalize high similarities (encourage diversity)
    diversity_loss = torch.clamp(similarities - 0.3, min=0).mean()

    return diversity_weight * diversity_loss


class AdversarialTrainer:
    """
    Helper class to manage adversarial training process
    """

    def __init__(self, discriminator, discriminator_optimizer, config):
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.config = config

        # Training parameters
        self.d_train_frequency = config.get('d_train_frequency', 1)  # Train D every N generator steps
        self.generator_loss_weight = config.get('generator_loss_weight', 0.1)
        self.feature_matching_weight = config.get('feature_matching_weight', 10.0)
        self.diversity_weight = config.get('diversity_weight', 0.5)
        self.loss_type = config.get('gan_loss_type', 'standard')

        self.step_count = 0

    def train_discriminator(self, real_patterns, fake_patterns):
        """Train discriminator on real and fake patterns"""
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()

        # Get discriminator predictions
        real_logits, real_features = self.discriminator(real_patterns, return_features=True)
        fake_logits, fake_features = self.discriminator(fake_patterns.detach(), return_features=True)

        # Compute discriminator loss
        d_loss = adversarial_discriminator_loss(real_logits, fake_logits, self.loss_type)

        d_loss.backward()
        self.discriminator_optimizer.step()

        return {
            'discriminator_loss': d_loss.item(),
            'real_score': torch.sigmoid(real_logits).mean().item(),
            'fake_score': torch.sigmoid(fake_logits).mean().item()
        }

    def compute_generator_adversarial_loss(self, fake_patterns, real_patterns=None):
        """Compute adversarial loss for generator"""
        self.discriminator.eval()

        # Get discriminator predictions for generated patterns
        fake_logits, fake_features = self.discriminator(fake_patterns, return_features=True)

        # Generator adversarial loss
        gen_adv_loss = adversarial_generator_loss(fake_logits, self.loss_type)

        total_loss = self.generator_loss_weight * gen_adv_loss
        loss_components = {'adversarial': gen_adv_loss.item()}

        # Feature matching loss
        if real_patterns is not None and self.feature_matching_weight > 0:
            with torch.no_grad():
                _, real_features = self.discriminator(real_patterns, return_features=True)

            fm_loss = feature_matching_loss(real_features, fake_features)
            total_loss += self.feature_matching_weight * fm_loss
            loss_components['feature_matching'] = fm_loss.item()

        # Diversity regularization
        if self.diversity_weight > 0:
            div_loss = diversity_regularization_loss(fake_patterns, self.diversity_weight)
            total_loss += div_loss
            loss_components['diversity_reg'] = div_loss.item()

        return total_loss, loss_components

    def should_train_discriminator(self):
        """Determine if discriminator should be trained this step"""
        return self.step_count % self.d_train_frequency == 0

    def step(self):
        """Increment step counter"""
        self.step_count += 1


def create_adversarial_components(config):
    """
    Factory function to create adversarial training components

    Args:
        config: Configuration dict containing adversarial training parameters

    Returns:
        tuple: (discriminator, discriminator_optimizer, adversarial_trainer)
    """
    # Create discriminator
    discriminator = PatternDiscriminator(config)

    # Initialize weights
    discriminator.init_weights()

    # Create optimizer for discriminator
    d_lr = config.get('discriminator_lr', 0.0002)
    d_betas = config.get('discriminator_betas', (0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=d_lr,
        betas=d_betas
    )

    # Create trainer
    adversarial_trainer = AdversarialTrainer(
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        config=config
    )

    return discriminator, discriminator_optimizer, adversarial_trainer


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'max_len': 32,
        'discriminator_hidden_dim': 256,
        'discriminator_dropout': 0.3,
        'discriminator_lr': 0.0002,
        'generator_loss_weight': 0.1,
        'feature_matching_weight': 10.0,
        'diversity_weight': 0.5,
        'd_train_frequency': 1,
        'gan_loss_type': 'standard'
    }

    # Create components
    discriminator, d_optimizer, trainer = create_adversarial_components(config)

    # Test with dummy data
    batch_size = 4
    real_patterns = torch.randn(batch_size, 32, 3)
    fake_patterns = torch.randn(batch_size, 32, 3)

    # Test discriminator training
    if trainer.should_train_discriminator():
        d_metrics = trainer.train_discriminator(real_patterns, fake_patterns)
        print("Discriminator metrics:", d_metrics)

    # Test generator loss computation
    gen_loss, gen_components = trainer.compute_generator_adversarial_loss(
        fake_patterns, real_patterns
    )
    print("Generator loss:", gen_loss.item())
    print("Loss components:", gen_components)

    trainer.step()