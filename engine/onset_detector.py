"""Onset detection for identifying drum hit timings."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque
import librosa


@dataclass
class OnsetConfig:
    """Configuration for onset detection."""
    sample_rate: int = 44100
    hop_length: int = 512
    # Adaptive threshold parameters
    threshold_ratio: float = 1.5  # Multiplier over local median
    min_threshold: float = 0.01  # Minimum absolute threshold
    # Temporal constraints
    min_onset_interval_ms: float = 50  # Minimum time between onsets
    pre_max_ms: float = 30  # Look-ahead for peak picking
    post_max_ms: float = 30  # Look-behind for peak picking


class OnsetDetector:
    """Real-time onset detection using spectral flux."""

    def __init__(self, config: Optional[OnsetConfig] = None):
        self.config = config or OnsetConfig()

        # Buffers for streaming processing
        self.audio_buffer = deque(maxlen=4096)  # Rolling audio buffer
        self.flux_history = deque(maxlen=50)  # Recent spectral flux values
        self.last_onset_sample = -10000  # Sample index of last detected onset
        self.total_samples = 0

        # Pre-compute FFT parameters
        self.n_fft = 1024
        self.prev_spectrum: Optional[np.ndarray] = None

        # Minimum samples between onsets
        self.min_onset_samples = int(
            self.config.min_onset_interval_ms * self.config.sample_rate / 1000
        )

    def _compute_spectral_flux(self, audio: np.ndarray) -> float:
        """Compute spectral flux for onset detection."""
        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=self.n_fft))

        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum
            return 0.0

        # Half-wave rectified spectral flux (only increases matter)
        diff = spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(0, diff))

        self.prev_spectrum = spectrum
        return flux

    def _adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on recent flux history."""
        if len(self.flux_history) < 5:
            return self.config.min_threshold

        median_flux = np.median(list(self.flux_history))
        threshold = max(
            self.config.min_threshold,
            median_flux * self.config.threshold_ratio
        )
        return threshold

    def process(self, audio: np.ndarray) -> list[Tuple[int, float]]:
        """
        Process audio buffer and detect onsets.

        Args:
            audio: Audio samples (mono, float32)

        Returns:
            List of (sample_offset, strength) tuples for detected onsets
        """
        onsets = []

        # Add to rolling buffer
        self.audio_buffer.extend(audio)

        # Need enough samples for FFT
        if len(self.audio_buffer) < self.n_fft:
            self.total_samples += len(audio)
            return onsets

        # Process in hop-size chunks
        hop = self.config.hop_length
        buffer_array = np.array(self.audio_buffer)

        for i in range(0, len(audio), hop):
            if len(self.audio_buffer) < self.n_fft:
                break

            # Get analysis frame
            frame_start = max(0, len(buffer_array) - self.n_fft)
            frame = buffer_array[frame_start:frame_start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Compute spectral flux
            flux = self._compute_spectral_flux(frame)
            self.flux_history.append(flux)

            # Check for onset
            threshold = self._adaptive_threshold()
            current_sample = self.total_samples + i

            if flux > threshold:
                # Check minimum interval
                if current_sample - self.last_onset_sample >= self.min_onset_samples:
                    # Peak picking - ensure this is a local maximum
                    if len(self.flux_history) >= 3:
                        recent = list(self.flux_history)[-3:]
                        if recent[1] >= recent[0] and recent[1] >= recent[2]:
                            onsets.append((i, flux / threshold))  # Normalized strength
                            self.last_onset_sample = current_sample

        self.total_samples += len(audio)

        # Trim buffer to prevent unbounded growth
        while len(self.audio_buffer) > self.n_fft * 2:
            self.audio_buffer.popleft()

        return onsets

    def reset(self) -> None:
        """Reset detector state."""
        self.audio_buffer.clear()
        self.flux_history.clear()
        self.prev_spectrum = None
        self.last_onset_sample = -10000
        self.total_samples = 0


class OnsetDetectorLibrosa:
    """Alternative onset detector using librosa (higher latency, better accuracy)."""

    def __init__(self, config: Optional[OnsetConfig] = None):
        self.config = config or OnsetConfig()
        self.audio_buffer = []
        self.buffer_duration_sec = 0.1  # Process in 100ms chunks
        self.buffer_samples = int(self.buffer_duration_sec * self.config.sample_rate)
        self.processed_samples = 0

    def process(self, audio: np.ndarray) -> list[Tuple[int, float]]:
        """Process audio and return detected onsets."""
        self.audio_buffer.extend(audio)
        onsets = []

        if len(self.audio_buffer) >= self.buffer_samples:
            # Process accumulated buffer
            buffer_array = np.array(self.audio_buffer[:self.buffer_samples])
            self.audio_buffer = self.audio_buffer[self.buffer_samples:]

            # Use librosa onset detection
            onset_frames = librosa.onset.onset_detect(
                y=buffer_array,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                backtrack=False,
                units='samples'
            )

            # Get onset strengths
            onset_env = librosa.onset.onset_strength(
                y=buffer_array,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )

            for onset_sample in onset_frames:
                frame_idx = onset_sample // self.config.hop_length
                if frame_idx < len(onset_env):
                    strength = float(onset_env[frame_idx])
                else:
                    strength = 1.0
                onsets.append((onset_sample + self.processed_samples, strength))

            self.processed_samples += self.buffer_samples

        return onsets

    def reset(self) -> None:
        """Reset detector state."""
        self.audio_buffer = []
        self.processed_samples = 0


if __name__ == "__main__":
    # Test onset detector with synthetic signal
    import matplotlib.pyplot as plt

    sr = 44100
    duration = 2.0

    # Create test signal with clear transients
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)

    # Add impulses at known times
    impulse_times = [0.2, 0.5, 0.8, 1.1, 1.5, 1.8]
    for it in impulse_times:
        idx = int(it * sr)
        # Short burst (simulating drum hit)
        burst_len = int(0.05 * sr)
        if idx + burst_len < len(signal):
            signal[idx:idx + burst_len] = np.random.randn(burst_len) * np.exp(-np.linspace(0, 5, burst_len))

    # Test detector
    detector = OnsetDetector()
    buffer_size = 512
    detected_onsets = []

    for i in range(0, len(signal), buffer_size):
        buffer = signal[i:i + buffer_size]
        if len(buffer) < buffer_size:
            buffer = np.pad(buffer, (0, buffer_size - len(buffer)))

        onsets = detector.process(buffer)
        for offset, strength in onsets:
            detected_onsets.append((i + offset) / sr)

    print(f"Expected onsets at: {impulse_times}")
    print(f"Detected onsets at: {[f'{t:.3f}' for t in detected_onsets]}")
