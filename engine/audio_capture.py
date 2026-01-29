"""Audio capture module using sounddevice for low-latency input."""

import numpy as np
import sounddevice as sd
from typing import Callable, Optional
from dataclasses import dataclass
import threading
import queue


@dataclass
class AudioConfig:
    """Configuration for audio capture."""
    sample_rate: int = 44100
    buffer_size: int = 512  # ~11ms latency at 44100Hz
    channels: int = 1  # Mono input
    dtype: np.dtype = np.float32


class AudioCapture:
    """Real-time audio capture with callback support."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.callbacks: list[Callable[[np.ndarray], None]] = []
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self.current_device_id: Optional[int] = None

    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add a callback to be called with each audio buffer."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """Internal callback called by sounddevice."""
        if status:
            print(f"Audio status: {status}")

        # Copy the data to avoid issues with buffer reuse
        audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # Put in queue for async processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Drop frame if queue is full

        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(audio_data)
            except Exception as e:
                print(f"Error in audio callback: {e}")

    def set_device(self, device_id: int) -> bool:
        """Set the audio input device. Restarts capture if running."""
        was_running = self.is_running
        if was_running:
            self.stop()

        self.current_device_id = device_id
        print(f"Audio device set to: {device_id}")

        if was_running:
            self.start()

        return True

    def start(self) -> None:
        """Start audio capture."""
        if self.is_running:
            return

        # Build stream kwargs
        stream_kwargs = {
            'samplerate': self.config.sample_rate,
            'blocksize': self.config.buffer_size,
            'channels': self.config.channels,
            'dtype': self.config.dtype,
            'callback': self._audio_callback
        }

        # Add device if specified
        if self.current_device_id is not None:
            stream_kwargs['device'] = self.current_device_id

        self.stream = sd.InputStream(**stream_kwargs)
        self.stream.start()
        self.is_running = True
        device_name = f"device {self.current_device_id}" if self.current_device_id is not None else "default device"
        print(f"Audio capture started on {device_name}: {self.config.sample_rate}Hz, "
              f"buffer={self.config.buffer_size} samples")

    def stop(self) -> None:
        """Stop audio capture."""
        if not self.is_running:
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("Audio capture stopped")

    def get_audio_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the next audio frame from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'type': 'input'
                })
        return input_devices

    @staticmethod
    def list_output_devices() -> list[dict]:
        """List available audio output devices."""
        devices = sd.query_devices()
        output_devices = []
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                output_devices.append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'type': 'output'
                })
        return output_devices

    @staticmethod
    def list_all_devices() -> list[dict]:
        """List all available audio devices (input and output)."""
        devices = sd.query_devices()
        all_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 or dev['max_output_channels'] > 0:
                device_type = []
                if dev['max_input_channels'] > 0:
                    device_type.append('input')
                if dev['max_output_channels'] > 0:
                    device_type.append('output')

                all_devices.append({
                    'id': i,
                    'name': dev['name'],
                    'input_channels': dev['max_input_channels'],
                    'output_channels': dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'type': device_type[0] if len(device_type) == 1 else 'both'
                })
        return all_devices

    @staticmethod
    def get_default_device() -> Optional[dict]:
        """Get the default input device."""
        try:
            device_id = sd.default.device[0]
            if device_id is not None:
                dev = sd.query_devices(device_id)
                return {
                    'id': device_id,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                }
        except Exception:
            pass
        return None


if __name__ == "__main__":
    # Test audio capture
    print("Available input devices:")
    for dev in AudioCapture.list_devices():
        print(f"  [{dev['id']}] {dev['name']} ({dev['channels']}ch)")

    print("\nDefault device:", AudioCapture.get_default_device())

    # Test capture for 3 seconds
    capture = AudioCapture()

    def print_level(audio: np.ndarray):
        level = np.abs(audio).max()
        bars = int(level * 50)
        print(f"\rLevel: {'█' * bars}{' ' * (50 - bars)} {level:.3f}", end='')

    capture.add_callback(print_level)
    capture.start()

    import time
    time.sleep(3)

    capture.stop()
    print("\nDone!")
