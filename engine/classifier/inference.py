"""Inference module for real-time beatbox classification."""

import numpy as np
import torch
import librosa
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from .model import BeatboxCNN, BeatboxCNNLite, CLASS_NAMES, DRUM_CLASSES, create_model


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    sample_rate: int = 44100
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    window_duration_ms: int = 100  # Analysis window
    model_type: str = 'standard'  # 'standard' or 'lite'
    confidence_threshold: float = 0.5  # Minimum confidence for detection
    device: str = 'cpu'


class BeatboxClassifier:
    """Real-time beatbox sound classifier."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)

        # Calculate window size in samples
        self.window_samples = int(
            self.config.window_duration_ms * self.config.sample_rate / 1000
        )

        # Initialize model
        n_mels = 40 if self.config.model_type == 'lite' else self.config.n_mels
        self.model = create_model(
            model_type=self.config.model_type,
            n_mels=n_mels
        )
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=n_mels,
            fmin=20,
            fmax=self.config.sample_rate // 2
        )

        self._weights_loaded = False

    def load_weights(self, weights_path: str) -> bool:
        """Load pre-trained weights."""
        path = Path(weights_path)
        if not path.exists():
            print(f"Weights not found: {weights_path}")
            return False

        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._weights_loaded = True
            print(f"Loaded weights from {weights_path}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        # Ensure correct length
        if len(audio) < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - len(audio)))
        elif len(audio) > self.window_samples:
            audio = audio[:self.window_samples]

        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window='hann'
        )

        # Convert to mel spectrogram
        mel_spec = np.dot(self.mel_basis, np.abs(stft) ** 2)

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

        return mel_spec

    def classify(self, audio: np.ndarray) -> Tuple[Optional[str], float, dict]:
        """
        Classify a beatbox sound.

        Args:
            audio: Audio samples (mono, float32)

        Returns:
            Tuple of (class_name or None, confidence, all_probabilities)
        """
        # Compute features
        mel_spec = self._compute_mel_spectrogram(audio)

        # Convert to tensor
        x = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        x = x.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)

        # Extract results
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        all_probs = {name: probs[0, i].item() for i, name in enumerate(CLASS_NAMES)}

        # Apply threshold
        if confidence < self.config.confidence_threshold:
            return None, confidence, all_probs

        class_name = CLASS_NAMES[predicted_idx]
        return class_name, confidence, all_probs

    def get_midi_note(self, class_name: str) -> int:
        """Get MIDI note number for a drum class."""
        return DRUM_CLASSES.get(class_name, 36)  # Default to kick


class RuleBasedClassifier:
    """
    Simple rule-based classifier for initial development.

    Uses spectral features to distinguish between:
    - Kick: Low frequency energy
    - Snare: Mid-high frequency with noise
    - Hi-hat: High frequency, short attack
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def classify(self, audio: np.ndarray) -> Tuple[str, float, dict]:
        """Classify based on spectral features."""
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)

        # Frequency band energies
        low_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 2000)
        high_mask = freqs >= 2000

        low_energy = np.sum(spectrum[low_mask] ** 2)
        mid_energy = np.sum(spectrum[mid_mask] ** 2)
        high_energy = np.sum(spectrum[high_mask] ** 2)

        total_energy = low_energy + mid_energy + high_energy + 1e-8

        # Normalize
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        # Simple decision rules
        probs = {
            'kick': low_ratio * 2,
            'snare': mid_ratio * 1.5 + high_ratio * 0.5,
            'hihat': high_ratio * 2,
            'clap': mid_ratio,
            'tom': low_ratio + mid_ratio * 0.5,
        }

        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Get best class
        best_class = max(probs, key=probs.get)
        confidence = probs[best_class]

        return best_class, confidence, probs

    def get_midi_note(self, class_name: str) -> int:
        """Get MIDI note number for a drum class."""
        return DRUM_CLASSES.get(class_name, 36)


if __name__ == "__main__":
    import time

    # Test inference speed
    config = InferenceConfig()
    classifier = BeatboxClassifier(config)

    # Generate test audio
    test_audio = np.random.randn(int(0.1 * config.sample_rate)).astype(np.float32) * 0.1

    # Warmup
    for _ in range(10):
        _ = classifier.classify(test_audio)

    # Benchmark
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        result = classifier.classify(test_audio)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"Classification result: {result[0]} (conf: {result[1]:.2f})")
    print(f"All probabilities: {result[2]}")
    print(f"Inference time: {elapsed:.2f}ms")

    # Test rule-based classifier
    print("\n--- Rule-based classifier ---")
    rule_classifier = RuleBasedClassifier()
    result = rule_classifier.classify(test_audio)
    print(f"Classification result: {result[0]} (conf: {result[1]:.2f})")
