"""
Training script for the BeatBox classifier model.

Usage:
    python train.py --data_dir dataset/ --epochs 50 --batch_size 32
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa

import sys
sys.path.append(str(Path(__file__).parent.parent))

from engine.classifier.model import BeatboxCNN, BeatboxCNNLite, CLASS_NAMES, create_model


class BeatboxDataset(Dataset):
    """Dataset for beatbox sound classification."""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        duration_ms: int = 100,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration_samples = int(duration_ms * sample_rate / 1000)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

        # Load file list
        self.samples = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for audio_file in class_dir.glob("*.wav"):
                    self.samples.append((audio_file, class_idx))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            count = sum(1 for s in self.samples if s[1] == class_idx)
            print(f"  {class_name}: {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        audio_path, class_idx = self.samples[idx]

        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Pad or trim to fixed duration
        if len(audio) < self.duration_samples:
            audio = np.pad(audio, (0, self.duration_samples - len(audio)))
        else:
            # Center crop
            start = (len(audio) - self.duration_samples) // 2
            audio = audio[start:start + self.duration_samples]

        # Apply augmentation
        if self.augment:
            audio = self._augment(audio)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Convert to tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dim

        return mel_spec, class_idx

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Time shift
        if np.random.random() < 0.5:
            shift = np.random.randint(-len(audio) // 10, len(audio) // 10)
            audio = np.roll(audio, shift)

        # Add noise
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.01)
            audio = audio + np.random.randn(len(audio)) * noise_level

        # Volume change
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain

        return audio


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train BeatBox classifier")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="../engine/classifier/weights",
                        help="Directory to save model weights")
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "lite"],
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--n_mels", type=int, default=64,
                        help="Number of mel bands")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    n_mels = 40 if args.model_type == "lite" else args.n_mels
    full_dataset = BeatboxDataset(
        args.data_dir,
        n_mels=n_mels,
        augment=args.augment,
    )

    if len(full_dataset) == 0:
        print("Error: No training data found!")
        print(f"Please add audio files to: {args.data_dir}/[class_name]/*.wav")
        print(f"Classes: {CLASS_NAMES}")
        return

    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    model = create_model(args.model_type, n_mels=n_mels)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_dir / f"beatbox_{args.model_type}_best.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (acc: {val_acc:.1f}%)")

    # Save final model
    final_path = output_dir / f"beatbox_{args.model_type}_final.pt"
    torch.save(model.state_dict(), final_path)

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
