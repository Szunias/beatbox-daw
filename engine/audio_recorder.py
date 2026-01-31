"""Audio recorder module for recording audio to files."""

import numpy as np
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
import os


@dataclass
class RecordingConfig:
    """Configuration for audio recording."""
    sample_rate: int = 44100
    channels: int = 1  # Mono recording
    dtype: np.dtype = np.float32
    max_duration: float = 3600.0  # Maximum recording duration in seconds (1 hour)


@dataclass
class RecordingMetadata:
    """Metadata for a recording."""
    start_time: float = 0.0
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 1
    num_samples: int = 0
    created_at: str = ""


class AudioRecorder:
    """Records audio buffers to memory and exports to WAV files."""

    def __init__(self, config: Optional[RecordingConfig] = None):
        self.config = config or RecordingConfig()

        # Recording state
        self.is_recording = False
        self.recording_start_time: Optional[float] = None

        # Audio buffer storage
        self.audio_buffers: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # Recording metadata
        self.metadata: Optional[RecordingMetadata] = None

        # Callbacks for recording events
        self.on_recording_started: Optional[Callable[[], None]] = None
        self.on_recording_stopped: Optional[Callable[[RecordingMetadata], None]] = None
        self.on_buffer_recorded: Optional[Callable[[int], None]] = None

    def start_recording(self, start_time: float = 0.0) -> bool:
        """Start recording audio.

        Args:
            start_time: Timeline position where recording starts (in seconds)

        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            return False

        with self._buffer_lock:
            self.audio_buffers = []

        self.recording_start_time = time.perf_counter()
        self.metadata = RecordingMetadata(
            start_time=start_time,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            created_at=datetime.now().isoformat()
        )
        self.is_recording = True

        if self.on_recording_started:
            try:
                self.on_recording_started()
            except Exception as e:
                print(f"Error in recording started callback: {e}")

        print(f"Audio recording started at timeline position: {start_time:.3f}s")
        return True

    def stop_recording(self) -> Optional[RecordingMetadata]:
        """Stop recording and return metadata.

        Returns:
            Recording metadata or None if not recording
        """
        if not self.is_recording:
            return None

        self.is_recording = False

        # Calculate duration
        if self.recording_start_time is not None and self.metadata is not None:
            self.metadata.duration = time.perf_counter() - self.recording_start_time

            with self._buffer_lock:
                if self.audio_buffers:
                    total_samples = sum(len(buf) for buf in self.audio_buffers)
                    self.metadata.num_samples = total_samples
                    # Recalculate duration based on actual samples
                    self.metadata.duration = total_samples / self.config.sample_rate

        metadata = self.metadata

        if self.on_recording_stopped and metadata:
            try:
                self.on_recording_stopped(metadata)
            except Exception as e:
                print(f"Error in recording stopped callback: {e}")

        print(f"Audio recording stopped: {metadata.duration:.3f}s, "
              f"{metadata.num_samples} samples")
        return metadata

    def add_audio_buffer(self, audio_data: np.ndarray) -> None:
        """Add an audio buffer to the recording.

        This method is thread-safe and can be called from audio callbacks.

        Args:
            audio_data: Audio buffer (numpy array of samples)
        """
        if not self.is_recording:
            return

        # Check max duration
        if self.metadata and self.recording_start_time:
            elapsed = time.perf_counter() - self.recording_start_time
            if elapsed > self.config.max_duration:
                print(f"Max recording duration reached: {self.config.max_duration}s")
                self.stop_recording()
                return

        # Copy the data to avoid issues with buffer reuse
        buffer_copy = audio_data.copy()

        with self._buffer_lock:
            self.audio_buffers.append(buffer_copy)
            buffer_count = len(self.audio_buffers)

        if self.on_buffer_recorded:
            try:
                self.on_buffer_recorded(buffer_count)
            except Exception as e:
                print(f"Error in buffer recorded callback: {e}")

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get the complete recorded audio data.

        Returns:
            Numpy array of all recorded audio or None if no data
        """
        with self._buffer_lock:
            if not self.audio_buffers:
                return None
            return np.concatenate(self.audio_buffers)

    def get_duration(self) -> float:
        """Get the current recording duration in seconds."""
        with self._buffer_lock:
            if not self.audio_buffers:
                return 0.0
            total_samples = sum(len(buf) for buf in self.audio_buffers)
            return total_samples / self.config.sample_rate

    def export_wav(self, filename: str, normalize: bool = True) -> bool:
        """Export recorded audio to a WAV file.

        Args:
            filename: Output WAV file path
            normalize: If True, normalize audio to prevent clipping

        Returns:
            True if export successful, False otherwise
        """
        audio_data = self.get_audio_data()

        if audio_data is None or len(audio_data) == 0:
            print("No audio data to export")
            return False

        try:
            # Import scipy here to avoid import errors if not installed
            from scipy.io import wavfile

            # Normalize if requested
            if normalize:
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.95  # Leave some headroom

            # Ensure directory exists
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Convert to appropriate format for wavfile
            # scipy.io.wavfile expects int16 or float32
            if self.config.dtype == np.float32:
                # Keep as float32, values should be in [-1, 1]
                audio_data = audio_data.astype(np.float32)
            else:
                # Convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)

            # Handle mono/stereo
            if self.config.channels == 1:
                # Ensure 1D array for mono
                audio_data = audio_data.flatten()

            wavfile.write(filename, self.config.sample_rate, audio_data)

            print(f"Exported WAV file: {filename} "
                  f"({len(audio_data)} samples, {self.config.sample_rate}Hz)")
            return True

        except ImportError:
            print("scipy is required for WAV export. Install with: pip install scipy")
            return False
        except Exception as e:
            print(f"Error exporting WAV: {e}")
            return False

    def clear(self) -> None:
        """Clear all recorded audio data."""
        with self._buffer_lock:
            self.audio_buffers = []
        self.metadata = None
        self.recording_start_time = None
        print("Audio recording cleared")

    def get_buffer_count(self) -> int:
        """Get the number of recorded buffers."""
        with self._buffer_lock:
            return len(self.audio_buffers)

    def get_recording_info(self) -> dict:
        """Get information about the current recording.

        Returns:
            Dictionary with recording information
        """
        with self._buffer_lock:
            buffer_count = len(self.audio_buffers)
            total_samples = sum(len(buf) for buf in self.audio_buffers) if self.audio_buffers else 0

        return {
            'is_recording': self.is_recording,
            'buffer_count': buffer_count,
            'total_samples': total_samples,
            'duration': total_samples / self.config.sample_rate if total_samples > 0 else 0,
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'metadata': {
                'start_time': self.metadata.start_time if self.metadata else 0,
                'created_at': self.metadata.created_at if self.metadata else None
            } if self.metadata else None
        }


if __name__ == "__main__":
    # Test AudioRecorder
    print("Testing AudioRecorder...")

    # Create recorder
    config = RecordingConfig(sample_rate=44100, channels=1)
    recorder = AudioRecorder(config)

    # Simulate recording with test data
    print("\nStarting test recording...")
    recorder.start_recording(start_time=0.0)

    # Generate test audio (1 second of 440Hz sine wave)
    duration = 1.0
    t = np.linspace(0, duration, int(config.sample_rate * duration), dtype=np.float32)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add audio in chunks (simulating real-time capture)
    chunk_size = 512
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i + chunk_size]
        recorder.add_audio_buffer(chunk)

    # Stop recording
    metadata = recorder.stop_recording()

    # Print info
    print(f"\nRecording info: {recorder.get_recording_info()}")

    # Export to WAV
    test_filename = "test_recording.wav"
    if recorder.export_wav(test_filename):
        print(f"\nTest WAV file created: {test_filename}")
        # Clean up test file
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print("Test file cleaned up")

    print("\nAudioRecorder test complete!")
