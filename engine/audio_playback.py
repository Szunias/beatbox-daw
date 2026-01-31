"""
Audio Playback Module
Handles playback of audio clips synchronized with transport timing.
Includes per-track and master effects processing.
"""

import time
import threading
import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from pathlib import Path

from transport import Transport, TransportState
from effects_processor import EffectsProcessor, EffectsProcessorConfig


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

        # === Effects Processing ===
        # Per-track effects processors
        self._track_effects: Dict[str, EffectsProcessor] = {}  # track_id -> EffectsProcessor

        # Master effects processor (applied to final mix)
        effects_config = EffectsProcessorConfig(
            sample_rate=self.config.sample_rate,
            buffer_size=self.config.buffer_size
        )
        self._master_effects = EffectsProcessor(effects_config)

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
        """Internal callback called by sounddevice for output.

        Audio processing chain:
        1. Mix clips per-track into track buffers
        2. Apply per-track effects to each track buffer
        3. Sum all track outputs with volume/pan
        4. Apply master effects to final mix
        5. Apply master volume and clip output
        """
        if status:
            pass  # Silently ignore status messages in production

        # Fill with silence first
        outdata.fill(0)

        if not self._running:
            return

        current_tick = self.transport.current_tick

        # Process scheduled events and mix audio
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
                            except Exception:
                                pass  # Silently ignore callback errors in audio thread

                elif event.event_type == 'stop':
                    if event.clip_data.clip_id in self._active_clips:
                        del self._active_clips[event.clip_data.clip_id]

                        if self._clip_ended_callback:
                            try:
                                self._clip_ended_callback(event.clip_data.clip_id,
                                                         event.clip_data.track_id)
                            except Exception:
                                pass  # Silently ignore callback errors in audio thread

                self._event_index += 1

            # Group clips by track for per-track effects processing
            track_clips: Dict[str, List[Tuple[AudioClipData, int]]] = {}
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

                # Group by track
                if clip.track_id not in track_clips:
                    track_clips[clip.track_id] = []
                track_clips[clip.track_id].append((clip, sample_pos))

                # Update position
                self._active_clips[clip_id] = sample_pos + samples_to_read

                # Check if clip ended
                if sample_pos + samples_to_read >= clip.duration_samples:
                    clips_to_remove.append(clip_id)

            # Process each track's audio through its effects chain
            for track_id, clips in track_clips.items():
                # Create track buffer
                track_buffer = np.zeros((frames, 2), dtype=np.float32)

                # Get track settings
                track_settings = self._track_settings.get(track_id,
                    {'volume': 1.0, 'pan': 0.0, 'muted': False})

                # Mix all clips for this track (pre-effects, pre-track-volume)
                for clip, sample_pos in clips:
                    samples_remaining = clip.duration_samples - sample_pos
                    samples_to_read = min(frames, samples_remaining)

                    if samples_to_read <= 0:
                        continue

                    # Get audio data
                    audio_chunk = clip.audio_data[sample_pos:sample_pos + samples_to_read]

                    # Apply clip volume only (track volume applied post-effects)
                    clip_volume = clip.volume

                    # Calculate clip pan gains
                    clip_pan = max(-1.0, min(1.0, clip.pan))
                    left_gain = clip_volume * (1.0 - max(0, clip_pan))
                    right_gain = clip_volume * (1.0 + min(0, clip_pan))

                    # Mix into track buffer
                    if clip.is_stereo:
                        track_buffer[:samples_to_read, 0] += audio_chunk[:, 0] * left_gain
                        track_buffer[:samples_to_read, 1] += audio_chunk[:, 1] * right_gain
                    else:
                        track_buffer[:samples_to_read, 0] += audio_chunk * left_gain
                        track_buffer[:samples_to_read, 1] += audio_chunk * right_gain

                # Apply track effects if any
                if track_id in self._track_effects:
                    track_effects = self._track_effects[track_id]
                    if track_effects.effect_count > 0 and not track_effects.is_bypassed():
                        track_buffer = track_effects.process(track_buffer)

                # Apply track volume and pan, then mix into output
                track_volume = track_settings['volume']
                track_pan = max(-1.0, min(1.0, track_settings['pan']))
                left_track_gain = track_volume * (1.0 - max(0, track_pan))
                right_track_gain = track_volume * (1.0 + min(0, track_pan))

                outdata[:, 0] += track_buffer[:, 0] * left_track_gain
                outdata[:, 1] += track_buffer[:, 1] * right_track_gain

            # Apply master effects if any
            if self._master_effects.effect_count > 0 and not self._master_effects.is_bypassed():
                processed = self._master_effects.process(outdata)
                outdata[:] = processed

            # Apply master volume
            outdata *= self.master_volume

            # Remove finished clips
            for clip_id in clips_to_remove:
                if clip_id in self._active_clips:
                    del self._active_clips[clip_id]
                    clip = self._clips.get(clip_id)
                    if clip and self._clip_ended_callback:
                        try:
                            self._clip_ended_callback(clip_id, clip.track_id)
                        except Exception:
                            pass  # Silently ignore callback errors in audio thread

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

    # === Track Effects Management ===

    def _get_or_create_track_effects(self, track_id: str) -> EffectsProcessor:
        """Get or create an effects processor for a track."""
        if track_id not in self._track_effects:
            effects_config = EffectsProcessorConfig(
                sample_rate=self.config.sample_rate,
                buffer_size=self.config.buffer_size
            )
            self._track_effects[track_id] = EffectsProcessor(effects_config)
        return self._track_effects[track_id]

    def add_track_effect(self, track_id: str, effect_type: str,
                         effect_id: Optional[str] = None,
                         position: Optional[int] = None) -> Optional[str]:
        """
        Add an effect to a track's effect chain.

        Args:
            track_id: ID of the track
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect (auto-generated if not provided)
            position: Optional position in chain (appended if not provided)

        Returns:
            Effect ID if successful, None if effect type not found
        """
        processor = self._get_or_create_track_effects(track_id)
        return processor.add_effect(effect_type, effect_id, position)

    def remove_track_effect(self, track_id: str, effect_id: str) -> bool:
        """
        Remove an effect from a track's effect chain.

        Args:
            track_id: ID of the track
            effect_id: ID of the effect to remove

        Returns:
            True if removed, False if not found
        """
        if track_id not in self._track_effects:
            return False
        return self._track_effects[track_id].remove_effect(effect_id)

    def move_track_effect(self, track_id: str, effect_id: str,
                          new_position: int) -> bool:
        """
        Move an effect to a new position in a track's effect chain.

        Args:
            track_id: ID of the track
            effect_id: ID of the effect to move
            new_position: New position in chain

        Returns:
            True if moved, False if not found
        """
        if track_id not in self._track_effects:
            return False
        return self._track_effects[track_id].move_effect(effect_id, new_position)

    def set_track_effect_parameter(self, track_id: str, effect_id: str,
                                    param_name: str, value: float) -> bool:
        """
        Set a parameter on a track's effect.

        Args:
            track_id: ID of the track
            effect_id: ID of the effect
            param_name: Name of the parameter
            value: New value

        Returns:
            True if set, False if not found
        """
        if track_id not in self._track_effects:
            return False
        return self._track_effects[track_id].set_effect_parameter(
            effect_id, param_name, value
        )

    def get_track_effect_parameters(self, track_id: str,
                                     effect_id: str) -> Optional[Dict[str, Any]]:
        """Get parameters of a track's effect."""
        if track_id not in self._track_effects:
            return None
        return self._track_effects[track_id].get_effect_parameters(effect_id)

    def get_track_effects_chain(self, track_id: str) -> List[Dict[str, Any]]:
        """Get information about all effects in a track's chain."""
        if track_id not in self._track_effects:
            return []
        return self._track_effects[track_id].get_chain_info()

    def bypass_track_effects(self, track_id: str, bypassed: bool = True) -> bool:
        """
        Bypass all effects for a track.

        Args:
            track_id: ID of the track
            bypassed: Whether to bypass

        Returns:
            True if track exists
        """
        if track_id not in self._track_effects:
            return False
        self._track_effects[track_id].bypass(bypassed)
        return True

    def reset_track_effects(self, track_id: str) -> bool:
        """
        Reset all effects in a track's chain (clear buffers/state).

        Args:
            track_id: ID of the track

        Returns:
            True if track exists
        """
        if track_id not in self._track_effects:
            return False
        self._track_effects[track_id].reset()
        return True

    def clear_track_effects(self, track_id: str) -> bool:
        """
        Remove all effects from a track's chain.

        Args:
            track_id: ID of the track

        Returns:
            True if track exists
        """
        if track_id not in self._track_effects:
            return False
        self._track_effects[track_id].clear()
        return True

    # === Master Effects Management ===

    def add_master_effect(self, effect_type: str,
                          effect_id: Optional[str] = None,
                          position: Optional[int] = None) -> Optional[str]:
        """
        Add an effect to the master effect chain.

        Args:
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect (auto-generated if not provided)
            position: Optional position in chain (appended if not provided)

        Returns:
            Effect ID if successful, None if effect type not found
        """
        return self._master_effects.add_effect(effect_type, effect_id, position)

    def remove_master_effect(self, effect_id: str) -> bool:
        """
        Remove an effect from the master effect chain.

        Args:
            effect_id: ID of the effect to remove

        Returns:
            True if removed, False if not found
        """
        return self._master_effects.remove_effect(effect_id)

    def move_master_effect(self, effect_id: str, new_position: int) -> bool:
        """
        Move an effect to a new position in the master effect chain.

        Args:
            effect_id: ID of the effect to move
            new_position: New position in chain

        Returns:
            True if moved, False if not found
        """
        return self._master_effects.move_effect(effect_id, new_position)

    def set_master_effect_parameter(self, effect_id: str,
                                     param_name: str, value: float) -> bool:
        """
        Set a parameter on a master effect.

        Args:
            effect_id: ID of the effect
            param_name: Name of the parameter
            value: New value

        Returns:
            True if set, False if not found
        """
        return self._master_effects.set_effect_parameter(effect_id, param_name, value)

    def get_master_effect_parameters(self, effect_id: str) -> Optional[Dict[str, Any]]:
        """Get parameters of a master effect."""
        return self._master_effects.get_effect_parameters(effect_id)

    def get_master_effects_chain(self) -> List[Dict[str, Any]]:
        """Get information about all effects in the master chain."""
        return self._master_effects.get_chain_info()

    def bypass_master_effects(self, bypassed: bool = True) -> None:
        """Bypass all master effects."""
        self._master_effects.bypass(bypassed)

    def reset_master_effects(self) -> None:
        """Reset all master effects (clear buffers/state)."""
        self._master_effects.reset()

    def clear_master_effects(self) -> None:
        """Remove all effects from the master chain."""
        self._master_effects.clear()

    def is_master_effects_bypassed(self) -> bool:
        """Check if master effects are bypassed."""
        return self._master_effects.is_bypassed()

    def get_available_effect_types(self) -> List[str]:
        """Get list of available effect types."""
        return list(EffectsProcessor.EFFECT_TYPES.keys())


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
