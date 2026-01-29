"""
Audio augmentation utilities for dataset expansion.

Supports various augmentation techniques for drum sounds:
- Pitch shifting
- Time stretching
- Noise injection
- Room simulation (reverb)
- EQ variation
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Optional
import random


class AudioAugmenter:
    """Collection of audio augmentation methods."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def pitch_shift(
        self, audio: np.ndarray, semitones: float
    ) -> np.ndarray:
        """Shift pitch by specified semitones."""
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=semitones
        )

    def time_stretch(
        self, audio: np.ndarray, rate: float
    ) -> np.ndarray:
        """Time stretch by specified rate (>1 = faster, <1 = slower)."""
        return librosa.effects.time_stretch(audio, rate=rate)

    def add_noise(
        self, audio: np.ndarray, noise_level: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise

    def add_colored_noise(
        self, audio: np.ndarray, noise_level: float = 0.01, color: str = "pink"
    ) -> np.ndarray:
        """Add colored noise (pink, brown, or white)."""
        n = len(audio)

        if color == "white":
            noise = np.random.randn(n)
        elif color == "pink":
            # Pink noise: 1/f spectrum
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            white = np.random.randn(n)
            pink_fft = np.fft.rfft(white) * pink_filter
            noise = np.fft.irfft(pink_fft, n)
        elif color == "brown":
            # Brown noise: 1/f^2 spectrum (random walk)
            noise = np.cumsum(np.random.randn(n))
        else:
            noise = np.random.randn(n)

        # Normalize noise
        noise = noise / np.abs(noise).max()

        return audio + noise * noise_level

    def apply_reverb(
        self,
        audio: np.ndarray,
        decay: float = 0.3,
        room_size: float = 0.5,
    ) -> np.ndarray:
        """Apply simple reverb effect."""
        # Create impulse response
        ir_length = int(room_size * self.sample_rate)
        ir = np.random.randn(ir_length) * np.exp(-np.linspace(0, 10, ir_length))

        # Convolve
        wet = signal.convolve(audio, ir, mode="full")[: len(audio)]
        wet = wet / np.abs(wet).max()

        # Mix dry and wet
        return audio * (1 - decay) + wet * decay

    def apply_eq(
        self,
        audio: np.ndarray,
        low_gain: float = 1.0,
        mid_gain: float = 1.0,
        high_gain: float = 1.0,
    ) -> np.ndarray:
        """Apply 3-band EQ."""
        # Design filters
        nyq = self.sample_rate // 2

        # Low pass for bass
        b_low, a_low = signal.butter(4, 200 / nyq, btype="low")
        # Band pass for mids
        b_mid, a_mid = signal.butter(4, [200 / nyq, 2000 / nyq], btype="band")
        # High pass for highs
        b_high, a_high = signal.butter(4, 2000 / nyq, btype="high")

        # Apply filters
        low = signal.filtfilt(b_low, a_low, audio) * low_gain
        mid = signal.filtfilt(b_mid, a_mid, audio) * mid_gain
        high = signal.filtfilt(b_high, a_high, audio) * high_gain

        return low + mid + high

    def change_volume(
        self, audio: np.ndarray, gain_db: float
    ) -> np.ndarray:
        """Change volume by dB amount."""
        gain = 10 ** (gain_db / 20)
        return audio * gain

    def random_augment(
        self, audio: np.ndarray, intensity: float = 0.5
    ) -> np.ndarray:
        """Apply random combination of augmentations."""
        result = audio.copy()

        # Pitch shift (50% chance)
        if random.random() < 0.5:
            semitones = random.uniform(-2 * intensity, 2 * intensity)
            result = self.pitch_shift(result, semitones)

        # Time stretch (30% chance)
        if random.random() < 0.3:
            rate = random.uniform(0.9, 1.1)
            result = self.time_stretch(result, rate)
            # Adjust length back
            if len(result) > len(audio):
                result = result[: len(audio)]
            else:
                result = np.pad(result, (0, len(audio) - len(result)))

        # Add noise (40% chance)
        if random.random() < 0.4:
            noise_level = random.uniform(0.001, 0.02 * intensity)
            noise_type = random.choice(["white", "pink"])
            result = self.add_colored_noise(result, noise_level, noise_type)

        # Volume change (60% chance)
        if random.random() < 0.6:
            gain_db = random.uniform(-6 * intensity, 6 * intensity)
            result = self.change_volume(result, gain_db)

        # EQ variation (30% chance)
        if random.random() < 0.3:
            eq_range = 0.5 * intensity
            result = self.apply_eq(
                result,
                low_gain=random.uniform(1 - eq_range, 1 + eq_range),
                mid_gain=random.uniform(1 - eq_range, 1 + eq_range),
                high_gain=random.uniform(1 - eq_range, 1 + eq_range),
            )

        # Reverb (20% chance)
        if random.random() < 0.2:
            decay = random.uniform(0.1, 0.3 * intensity)
            result = self.apply_reverb(result, decay=decay)

        # Normalize
        result = result / (np.abs(result).max() + 1e-8)

        return result


def augment_dataset(
    input_dir: str,
    output_dir: str,
    num_augmentations: int = 5,
    sample_rate: int = 44100,
):
    """
    Augment all audio files in input directory.

    Directory structure:
        input_dir/
            kick/
                kick_001.wav
                ...
            snare/
                ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    augmenter = AudioAugmenter(sample_rate)

    total_generated = 0

    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(exist_ok=True)

        audio_files = list(class_dir.glob("*.wav"))
        print(f"\nProcessing {class_name}: {len(audio_files)} files")

        for audio_file in audio_files:
            # Load original
            audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

            # Save original to output
            original_name = f"{audio_file.stem}_orig.wav"
            sf.write(output_class_dir / original_name, audio, sample_rate)

            # Generate augmentations
            for i in range(num_augmentations):
                augmented = augmenter.random_augment(audio, intensity=0.5)

                aug_name = f"{audio_file.stem}_aug{i + 1:02d}.wav"
                sf.write(output_class_dir / aug_name, augmented, sample_rate)
                total_generated += 1

        class_total = len(audio_files) * (num_augmentations + 1)
        print(f"  Generated {class_total} samples for {class_name}")

    print(f"\nTotal augmented samples generated: {total_generated}")


def main():
    parser = argparse.ArgumentParser(description="Augment beatbox dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset/raw",
        help="Input directory with raw samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/augmented",
        help="Output directory for augmented samples",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=5,
        help="Number of augmentations per sample",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=44100, help="Sample rate"
    )
    args = parser.parse_args()

    augment_dataset(
        args.input_dir,
        args.output_dir,
        args.num_augmentations,
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
