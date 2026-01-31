"""
Audio Playback Module
Handles playback of audio clips synchronized with transport timing.
"""

import time
import threading
import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path

from transport import Transport, TransportState


@dataclass
class AudioPlaybackConfig:
    """Configuration for audio playback."""
    sample_rate: int = 44100
    buffer_size: int = 512  # ~11ms latency at 44100Hz
    channels: int = 2  # Stereo output
    dtype: np.dtype = np.float32


@dataclass
class AudioClipData:
    """Loaded audio data for a clip."""
    clip_id: str
    track_id: str
    audio_data: np.ndarray  # Audio samples (mono or stereo)
    sample_rate: int
    start_tick: int
    duration_ticks: int
    volume: float = 1.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    muted: bool = False

    @property
    def duration_samples(self) -> int:
        """Get duration in samples."""
        return len(self.audio_data) if self.audio_data.ndim == 1 else self.audio_data.shape[0]

    @property
    def is_stereo(self) -> bool:
        """Check if audio data is stereo."""
        return self.audio_data.ndim > 1 and self.audio_data.shape[1] == 2


@dataclass
class ScheduledAudioEvent:
    """An audio clip scheduled for playback."""
    tick: int
    clip_data: AudioClipData
    event_type: str  # 'start', 'stop'

    def __lt__(self, other):
        return self.tick < other.tick


class AudioPlayback:
    """
    Plays audio clips synchronized with transport timing.
    Works with Transport to handle timing and playback position.
    """

    def __init__(self, transport: Transport, config: Optional[AudioPlaybackConfig] = None):
        self.transport = transport
        self.config = config or AudioPlaybackConfig()

        # Audio output stream
        self.stream: Optional[sd.OutputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.current_device_id: Optional[int] = None

        # Loaded audio clips
        self._clips: Dict[str, AudioClipData] = {}  # clip_id -> AudioClipData
        self._events: List[ScheduledAudioEvent] = []
        self._event_index = 0

        # Active playback tracking
        self._active_clips: Dict[str, int] = {}  # clip_id -> current sample position
        self._lock = threading.Lock()

        # Track settings cache
        self._track_settings: Dict[str, Dict] = {}  # track_id -> {volume, pan, muted}

        # Master volume
        self.master_volume: float = 1.0

        # Callbacks
        self._clip_started_callback: Optional[Callable[[str, str], None]] = None  # clip_id, track_id
        self._clip_ended_callback: Optional[Callable[[str, str], None]] = None  # clip_id, track_id

        # Register transport callbacks
        transport.add_state_callback(self._on_transport_state_change)

    def set_clip_started_callback(self, callback: Callable[[str, str], None]):
        """Set callback for clip start events (clip_id, track_id)."""
        self._clip_started_callback = callback

    def set_clip_ended_callback(self, callback: Callable[[str, str], None]):
        """Set callback for clip end events (clip_id, track_id)."""
        self._clip_ended_callback = callback

    def load_audio_file(self, filepath: str, clip_id: str, track_id: str,
                        start_tick: int, duration_ticks: int,
                        volume: float = 1.0, pan: float = 0.0) -> bool:
        """
        Load an audio file for playback.

        Args:
            filepath: Path to audio file (WAV format)
            clip_id: Unique ID for this clip
            track_id: ID of the track this clip belongs to
            start_tick: Timeline position where clip starts
            duration_ticks: Duration in ticks
            volume: Clip volume (0.0-1.0)
            pan: Stereo pan (-1.0 left, 0.0 center, 1.0 right)

        Returns:
            True if loaded successfully
        """
        try:
            import scipy.io.wavfile as wav

            sample_rate, audio_data = wav.read(filepath)

            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Resample if needed (simple approach - should use librosa for quality)
            if sample_rate != self.config.sample_rate:
                ratio = self.config.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
                audio_data = audio_data[indices]

            clip_data = AudioClipData(
                clip_id=clip_id,
                track_id=track_id,
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                volume=volume,
                pan=pan
            )

            with self._lock:
                self._clips[clip_id] = clip_data
                self._rebuild_events()

            return True

        except Exception as e:
            print(f"Failed to load audio file {filepath}: {e}")
            return False

    def load_audio_data(self, audio_data: np.ndarray, sample_rate: int,
                        clip_id: str, track_id: str, start_tick: int,
                        duration_ticks: int, volume: float = 1.0,
                        pan: float = 0.0) -> bool:
        """
        Load audio data directly (e.g., from a recording).

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            clip_id: Unique ID for this clip
            track_id: ID of the track this clip belongs to
            start_tick: Timeline position where clip starts
            duration_ticks: Duration in ticks
            volume: Clip volume (0.0-1.0)
            pan: Stereo pan (-1.0 left, 0.0 center, 1.0 right)

        Returns:
            True if loaded successfully
        """
        try:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)

            # Resample if needed
            if sample_rate != self.config.sample_rate:
                ratio = self.config.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
                audio_data = audio_data[indices]

            clip_data = AudioClipData(
                clip_id=clip_id,
                track_id=track_id,
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                volume=volume,
                pan=pan
            )

            with self._lock:
                self._clips[clip_id] = clip_data
                self._rebuild_events()

            return True

        except Exception as e:
            print(f"Failed to load audio data: {e}")
            return False

    def unload_clip(self, clip_id: str) -> bool:
        """Remove a loaded audio clip."""
        with self._lock:
            if clip_id in self._clips:
                del self._clips[clip_id]
                # Remove from active clips if playing
                self._active_clips.pop(clip_id, None)
                self._rebuild_events()
                return True
        return False

    def clear(self):
        """Clear all loaded clips."""
        with self._lock:
            self._clips.clear()
            self._events.clear()
            self._active_clips.clear()
            self._event_index = 0

    def set_track_settings(self, track_id: str, volume: float = None,
                           pan: float = None, muted: bool = None):
        """Update track mixer settings."""
        if track_id not in self._track_settings:
            self._track_settings[track_id] = {'volume': 1.0, 'pan': 0.0, 'muted': False}

        if volume is not None:
            self._track_settings[track_id]['volume'] = max(0.0, min(1.0, volume))
        if pan is not None:
            self._track_settings[track_id]['pan'] = max(-1.0, min(1.0, pan))
        if muted is not None:
            self._track_settings[track_id]['muted'] = muted

        # Update any loaded clips for this track
        with self._lock:
            for clip in self._clips.values():
                if clip.track_id == track_id:
                    if muted is not None:
                        clip.muted = muted

    def set_clip_muted(self, clip_id: str, muted: bool):
        """Set clip mute state."""
        with self._lock:
            if clip_id in self._clips:
                self._clips[clip_id].muted = muted

    def set_device(self, device_id: int) -> bool:
        """Set the audio output device. Restarts playback if running."""
        was_running = self._running
        if was_running:
            self._stop_playback()

        self.current_device_id = device_id
        print(f"Audio output device set to: {device_id}")

        if was_running:
            self._start_playback()

        return True

    def _rebuild_events(self):
        """Rebuild the event list from loaded clips."""
        self._events.clear()

        for clip in self._clips.values():
            # Start event
            self._events.append(ScheduledAudioEvent(
                tick=clip.start_tick,
                clip_data=clip,
                event_type='start'
            ))

            # End event
            end_tick = clip.start_tick + clip.duration_ticks
            self._events.append(ScheduledAudioEvent(
                tick=end_tick,
                clip_data=clip,
                event_type='stop'
            ))

        # Sort by tick
        self._events.sort()
        self._reset_index()

    def _reset_index(self):
        """Reset event index to match current transport position."""
        current_tick = self.transport.current_tick
        self._event_index = 0

        # Find first event at or after current position
        for i, event in enumerate(self._events):
            if event.tick >= current_tick:
                self._event_index = i
                break
        else:
            self._event_index = len(self._events)

        # Clear active clips and recalculate which should be playing
        with self._lock:
            self._active_clips.clear()

            # Find clips that should already be playing at current position
            for clip in self._clips.values():
                if (clip.start_tick <= current_tick <
                    clip.start_tick + clip.duration_ticks):
                    # Calculate how far into the clip we are
                    ticks_into_clip = current_tick - clip.start_tick
                    seconds_into_clip = self.transport._ticks_to_seconds(ticks_into_clip)
                    samples_into_clip = int(seconds_into_clip * self.config.sample_rate)
                    samples_into_clip = min(samples_into_clip, clip.duration_samples - 1)
                    self._active_clips[clip.clip_id] = samples_into_clip

    def _on_transport_state_change(self, state: TransportState):
        """Handle transport state changes."""
        if state == TransportState.PLAYING or state == TransportState.RECORDING:
            self._reset_index()
            self._start_playback()
        elif state in (TransportState.STOPPED, TransportState.PAUSED):
            self._stop_playback()

    def _start_playback(self):
        """Start audio output stream."""
        if self._running:
            return

        self._running = True

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

        try:
            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            device_name = f"device {self.current_device_id}" if self.current_device_id is not None else "default device"
            print(f"Audio playback started on {device_name}: {self.config.sample_rate}Hz, "
                  f"buffer={self.config.buffer_size} samples")
        except Exception as e:
            print(f"Failed to start audio playback: {e}")
            self._running = False

    def _stop_playback(self):
        """Stop audio output stream."""
        self._running = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            self.stream = None

        # Clear active clips
        with self._lock:
            self._active_clips.clear()

        print("Audio playback stopped")

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """Internal callback called by sounddevice for output."""
        if status:
            print(f"Audio playback status: {status}")

        # Fill with silence first
        outdata.fill(0)

        if not self._running:
            return

        current_tick = self.transport.current_tick

        # Process scheduled events
        with self._lock:
            # Check for new clips to start
            while (self._event_index < len(self._events) and
                   self._events[self._event_index].tick <= current_tick):

                event = self._events[self._event_index]

                if event.event_type == 'start':
                    if not event.clip_data.muted:
                        # Calculate starting sample position
                        ticks_past = current_tick - event.clip_data.start_tick
                        seconds_past = self.transport._ticks_to_seconds(ticks_past)
                        sample_offset = int(seconds_past * self.config.sample_rate)
                        sample_offset = max(0, min(sample_offset, event.clip_data.duration_samples - 1))

                        self._active_clips[event.clip_data.clip_id] = sample_offset

                        if self._clip_started_callback:
                            try:
                                self._clip_started_callback(event.clip_data.clip_id,
                                                           event.clip_data.track_id)
                            except Exception as e:
                                print(f"Clip started callback error: {e}")

                elif event.event_type == 'stop':
                    if event.clip_data.clip_id in self._active_clips:
                        del self._active_clips[event.clip_data.clip_id]

                        if self._clip_ended_callback:
                            try:
                                self._clip_ended_callback(event.clip_data.clip_id,
                                                         event.clip_data.track_id)
                            except Exception as e:
                                print(f"Clip ended callback error: {e}")

                self._event_index += 1

            # Mix all active clips
            clips_to_remove = []

            for clip_id, sample_pos in self._active_clips.items():
                clip = self._clips.get(clip_id)
                if clip is None or clip.muted:
                    continue

                # Get track settings
                track_settings = self._track_settings.get(clip.track_id,
                    {'volume': 1.0, 'pan': 0.0, 'muted': False})

                if track_settings['muted']:
                    continue

                # Calculate how many samples we can read
                samples_remaining = clip.duration_samples - sample_pos
                samples_to_read = min(frames, samples_remaining)

                if samples_to_read <= 0:
                    clips_to_remove.append(clip_id)
                    continue

                # Get audio data
                audio_chunk = clip.audio_data[sample_pos:sample_pos + samples_to_read]

                # Apply volume (clip * track * master)
                total_volume = clip.volume * track_settings['volume'] * self.master_volume

                # Calculate pan gains
                clip_pan = clip.pan + track_settings['pan']
                clip_pan = max(-1.0, min(1.0, clip_pan))
                left_gain = total_volume * (1.0 - max(0, clip_pan))
                right_gain = total_volume * (1.0 + min(0, clip_pan))

                # Mix into output buffer
                if clip.is_stereo:
                    # Stereo clip
                    outdata[:samples_to_read, 0] += audio_chunk[:, 0] * left_gain
                    outdata[:samples_to_read, 1] += audio_chunk[:, 1] * right_gain
                else:
                    # Mono clip - pan to stereo
                    outdata[:samples_to_read, 0] += audio_chunk * left_gain
                    outdata[:samples_to_read, 1] += audio_chunk * right_gain

                # Update position
                self._active_clips[clip_id] = sample_pos + samples_to_read

                # Check if clip ended
                if sample_pos + samples_to_read >= clip.duration_samples:
                    clips_to_remove.append(clip_id)

            # Remove finished clips
            for clip_id in clips_to_remove:
                if clip_id in self._active_clips:
                    del self._active_clips[clip_id]
                    clip = self._clips.get(clip_id)
                    if clip and self._clip_ended_callback:
                        try:
                            self._clip_ended_callback(clip_id, clip.track_id)
                        except Exception as e:
                            print(f"Clip ended callback error: {e}")

        # Clip output to prevent distortion
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def get_active_clips(self) -> List[str]:
        """Get list of currently playing clip IDs."""
        with self._lock:
            return list(self._active_clips.keys())

    def get_clip_info(self, clip_id: str) -> Optional[Dict]:
        """Get information about a loaded clip."""
        with self._lock:
            clip = self._clips.get(clip_id)
            if clip:
                return {
                    'clip_id': clip.clip_id,
                    'track_id': clip.track_id,
                    'start_tick': clip.start_tick,
                    'duration_ticks': clip.duration_ticks,
                    'duration_samples': clip.duration_samples,
                    'volume': clip.volume,
                    'pan': clip.pan,
                    'muted': clip.muted,
                    'is_stereo': clip.is_stereo
                }
        return None

    def get_loaded_clips(self) -> List[Dict]:
        """Get information about all loaded clips."""
        with self._lock:
            return [self.get_clip_info(clip_id) for clip_id in self._clips.keys()]

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
    def get_default_output_device() -> Optional[dict]:
        """Get the default output device."""
        try:
            device_id = sd.default.device[1]  # [1] is output device
            if device_id is not None:
                dev = sd.query_devices(device_id)
                return {
                    'id': device_id,
                    'name': dev['name'],
                    'channels': dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate']
                }
        except Exception:
            pass
        return None


# Standalone test
if __name__ == "__main__":
    import sys

    print("Audio Playback Test")
    print("=" * 40)

    # List output devices
    print("\nAvailable output devices:")
    for dev in AudioPlayback.list_output_devices():
        print(f"  [{dev['id']}] {dev['name']} ({dev['channels']}ch)")

    print("\nDefault output device:", AudioPlayback.get_default_output_device())

    # Create transport and playback
    transport = Transport()
    playback = AudioPlayback(transport)

    # Callbacks
    def on_clip_started(clip_id, track_id):
        print(f"Clip started: {clip_id} on track {track_id}")

    def on_clip_ended(clip_id, track_id):
        print(f"Clip ended: {clip_id} on track {track_id}")

    playback.set_clip_started_callback(on_clip_started)
    playback.set_clip_ended_callback(on_clip_ended)

    # Generate a test tone
    print("\nGenerating test tone...")
    duration_seconds = 2.0
    frequency = 440.0  # A4
    sample_rate = 44100
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    test_tone = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)

    # Load test tone as clip
    ppqn = transport.config.ppqn
    ticks_per_second = (transport.config.bpm / 60.0) * ppqn
    duration_ticks = int(duration_seconds * ticks_per_second)

    playback.load_audio_data(
        audio_data=test_tone,
        sample_rate=sample_rate,
        clip_id="test_clip_1",
        track_id="test_track_1",
        start_tick=0,
        duration_ticks=duration_ticks,
        volume=0.8,
        pan=0.0
    )

    print(f"Loaded test clip: {duration_seconds}s, {duration_ticks} ticks")
    print(f"Loaded clips: {playback.get_loaded_clips()}")

    # Play
    print("\nStarting playback...")
    transport.play()

    try:
        time.sleep(3)  # Play for 3 seconds
    except KeyboardInterrupt:
        pass

    transport.stop()
    print("\nPlayback stopped")
    print("Done!")
