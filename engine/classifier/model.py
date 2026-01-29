"""CNN model for beatbox sound classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# Drum classes mapping to General MIDI drum notes
DRUM_CLASSES = {
    'kick': 36,      # Bass Drum 1
    'snare': 38,     # Acoustic Snare
    'hihat': 42,     # Closed Hi-Hat
    'clap': 39,      # Hand Clap
    'tom': 45,       # Low Tom
}

CLASS_NAMES = list(DRUM_CLASSES.keys())
NUM_CLASSES = len(CLASS_NAMES)


class BeatboxCNN(nn.Module):
    """
    Small CNN for real-time beatbox classification.

    Input: Mel spectrogram (1, n_mels, time_frames)
    Output: Class probabilities (num_classes,)

    Target inference time: <5ms on CPU
    """

    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3
    ):
        super().__init__()

        self.n_mels = n_mels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Make a prediction with confidence.

        Args:
            x: Input tensor

        Returns:
            Tuple of (class_index, confidence)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)
            return predicted.item(), confidence.item()


class BeatboxCNNLite(nn.Module):
    """
    Even smaller CNN for very fast inference.

    Trade-off: Lower accuracy for faster inference.
    Target inference time: <2ms on CPU
    """

    def __init__(
        self,
        n_mels: int = 40,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        self.n_mels = n_mels
        self.num_classes = num_classes

        # Simplified architecture
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """Factory function to create a model."""
    if model_type == 'standard':
        return BeatboxCNN(**kwargs)
    elif model_type == 'lite':
        return BeatboxCNNLite(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing BeatboxCNN...")
    model = BeatboxCNN()
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 1
    n_mels = 64
    time_frames = 32  # ~100ms of audio at typical hop length

    x = torch.randn(batch_size, 1, n_mels, time_frames)

    import time

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Benchmark
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        output = model(x)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Output shape: {output.shape}")
    print(f"Inference time: {elapsed:.2f}ms")

    # Test lite model
    print("\nTesting BeatboxCNNLite...")
    model_lite = BeatboxCNNLite(n_mels=40)
    print(f"Parameters: {count_parameters(model_lite):,}")

    x_lite = torch.randn(batch_size, 1, 40, time_frames)

    # Warmup
    for _ in range(10):
        _ = model_lite(x_lite)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        output = model_lite(x_lite)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Output shape: {output.shape}")
    print(f"Inference time: {elapsed:.2f}ms")
