"""
Audio Export Module
Handles offline rendering and mixdown of audio clips to audio files.
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
from enum import Enum

from effects_processor import EffectsProcessor, EffectsProcessorConfig


class ExportState(Enum):
    """Export operation states."""
    IDLE = "idle"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ExportConfig:
    """Configuration for audio export."""
    sample_rate: int = 44100
    channels: int = 2  # Stereo output
    bit_depth: int = 16  # 16-bit PCM for WAV
    buffer_size: int = 4096  # Larger buffers for offline rendering
    normalize: bool = True  # Normalize output to prevent clipping
    normalize_headroom_db: float = -0.3  # Headroom for normalization


@dataclass
class ExportClipData:
    """Audio data for a clip to be exported."""
    clip_id: str
    track_id: str
    audio_data: np.ndarray  # Audio samples (mono or stereo)
    sample_rate: int
    start_sample: int  # Position in output (in samples)
    volume: float = 1.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    muted: bool = False

    @property
    def duration_samples(self) -> int:
        """Get duration in samples."""
        return len(self.audio_data) if self.audio_data.ndim == 1 else self.audio_data.shape[0]

    @property
    def end_sample(self) -> int:
        """Get end position in samples."""
        return self.start_sample + self.duration_samples

    @property
    def is_stereo(self) -> bool:
        """Check if audio data is stereo."""
        return self.audio_data.ndim > 1 and self.audio_data.shape[1] == 2


@dataclass
class ExportProgress:
    """Progress information for export operation."""
    state: ExportState = ExportState.IDLE
    progress: float = 0.0  # 0.0 to 1.0
    current_sample: int = 0
    total_samples: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None


class AudioExporter:
    """
    Exports audio clips to audio files with offline rendering.
    Performs mixdown with volume, pan, and effects processing.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

        # Clips to export
        self._clips: Dict[str, ExportClipData] = {}  # clip_id -> ExportClipData

        # Track settings
        self._track_settings: Dict[str, Dict[str, Any]] = {}  # track_id -> settings

        # Track effects processors
        self._track_effects: Dict[str, EffectsProcessor] = {}  # track_id -> EffectsProcessor

        # Master effects processor
        effects_config = EffectsProcessorConfig(
            sample_rate=self.config.sample_rate,
            buffer_size=self.config.buffer_size
        )
        self._master_effects = EffectsProcessor(effects_config)

        # Master volume
        self.master_volume: float = 1.0

        # Export state
        self._state = ExportState.IDLE
        self._progress = ExportProgress()
        self._cancel_requested = False
        self._lock = threading.Lock()

        # Callbacks
        self._progress_callback: Optional[Callable[[ExportProgress], None]] = None
        self._completed_callback: Optional[Callable[[str, bool, Optional[str]], None]] = None

    def set_progress_callback(self, callback: Callable[[ExportProgress], None]):
        """Set callback for progress updates (progress: ExportProgress)."""
        self._progress_callback = callback

    def set_completed_callback(self, callback: Callable[[str, bool, Optional[str]], None]):
        """Set callback for export completion (filepath, success, error_message)."""
        self._completed_callback = callback

    def add_clip(self, clip_id: str, track_id: str, audio_data: np.ndarray,
                 sample_rate: int, start_sample: int, volume: float = 1.0,
                 pan: float = 0.0, muted: bool = False) -> bool:
        """
        Add an audio clip for export.

        Args:
            clip_id: Unique ID for this clip
            track_id: ID of the track this clip belongs to
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            start_sample: Start position in output (in samples)
            volume: Clip volume (0.0-1.0)
            pan: Stereo pan (-1.0 left, 0.0 center, 1.0 right)
            muted: Whether clip is muted

        Returns:
            True if added successfully
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
                if audio_data.ndim == 1:
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
                    audio_data = audio_data[indices]
                else:
                    new_length = int(audio_data.shape[0] * ratio)
                    indices = np.linspace(0, audio_data.shape[0] - 1, new_length).astype(int)
                    audio_data = audio_data[indices]

            clip_data = ExportClipData(
                clip_id=clip_id,
                track_id=track_id,
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
                start_sample=start_sample,
                volume=volume,
                pan=pan,
                muted=muted
            )

            with self._lock:
                self._clips[clip_id] = clip_data

            return True

        except Exception:
            return False

    def remove_clip(self, clip_id: str) -> bool:
        """Remove a clip from the export list."""
        with self._lock:
            if clip_id in self._clips:
                del self._clips[clip_id]
                return True
        return False

    def clear_clips(self):
        """Clear all clips from the export list."""
        with self._lock:
            self._clips.clear()

    def set_track_settings(self, track_id: str, volume: float = None,
                           pan: float = None, muted: bool = None):
        """Update track mixer settings for export."""
        if track_id not in self._track_settings:
            self._track_settings[track_id] = {'volume': 1.0, 'pan': 0.0, 'muted': False}

        if volume is not None:
            self._track_settings[track_id]['volume'] = max(0.0, min(1.0, volume))
        if pan is not None:
            self._track_settings[track_id]['pan'] = max(-1.0, min(1.0, pan))
        if muted is not None:
            self._track_settings[track_id]['muted'] = muted

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
        Add an effect to a track's effect chain for export.

        Args:
            track_id: ID of the track
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect
            position: Optional position in chain

        Returns:
            Effect ID if successful, None if effect type not found
        """
        processor = self._get_or_create_track_effects(track_id)
        return processor.add_effect(effect_type, effect_id, position)

    def set_track_effect_parameter(self, track_id: str, effect_id: str,
                                    param_name: str, value: float) -> bool:
        """Set a parameter on a track's effect."""
        if track_id not in self._track_effects:
            return False
        return self._track_effects[track_id].set_effect_parameter(
            effect_id, param_name, value
        )

    def add_master_effect(self, effect_type: str,
                          effect_id: Optional[str] = None,
                          position: Optional[int] = None) -> Optional[str]:
        """Add an effect to the master effect chain."""
        return self._master_effects.add_effect(effect_type, effect_id, position)

    def set_master_effect_parameter(self, effect_id: str,
                                     param_name: str, value: float) -> bool:
        """Set a parameter on a master effect."""
        return self._master_effects.set_effect_parameter(effect_id, param_name, value)

    def clear_track_effects(self, track_id: str) -> bool:
        """Remove all effects from a track's chain."""
        if track_id not in self._track_effects:
            return False
        self._track_effects[track_id].clear()
        return True

    def clear_master_effects(self):
        """Remove all effects from the master chain."""
        self._master_effects.clear()

    def get_state(self) -> ExportState:
        """Get current export state."""
        return self._state

    def get_progress(self) -> ExportProgress:
        """Get current export progress."""
        return self._progress

    def cancel(self):
        """Request cancellation of current export operation."""
        self._cancel_requested = True

    def _calculate_total_duration(self) -> int:
        """Calculate total duration needed for export in samples."""
        if not self._clips:
            return 0

        max_end = 0
        for clip in self._clips.values():
            if not clip.muted:
                track_settings = self._track_settings.get(clip.track_id,
                    {'volume': 1.0, 'pan': 0.0, 'muted': False})
                if not track_settings['muted']:
                    max_end = max(max_end, clip.end_sample)

        return max_end

    def _render_buffer(self, output_buffer: np.ndarray, start_sample: int) -> None:
        """
        Render a buffer of audio samples.

        Args:
            output_buffer: Output buffer to fill (stereo, float32)
            start_sample: Starting sample position in the export
        """
        buffer_size = output_buffer.shape[0]
        end_sample = start_sample + buffer_size

        # Group clips by track
        track_clips: Dict[str, List[ExportClipData]] = {}

        for clip in self._clips.values():
            if clip.muted:
                continue

            # Check if clip overlaps with this buffer
            if clip.start_sample >= end_sample or clip.end_sample <= start_sample:
                continue

            # Get track settings
            track_settings = self._track_settings.get(clip.track_id,
                {'volume': 1.0, 'pan': 0.0, 'muted': False})

            if track_settings['muted']:
                continue

            # Group by track
            if clip.track_id not in track_clips:
                track_clips[clip.track_id] = []
            track_clips[clip.track_id].append(clip)

        # Process each track
        for track_id, clips in track_clips.items():
            # Create track buffer
            track_buffer = np.zeros((buffer_size, 2), dtype=np.float32)

            # Get track settings
            track_settings = self._track_settings.get(track_id,
                {'volume': 1.0, 'pan': 0.0, 'muted': False})

            # Mix all clips for this track
            for clip in clips:
                # Calculate overlap region
                clip_buffer_start = max(0, clip.start_sample - start_sample)
                clip_buffer_end = min(buffer_size, clip.end_sample - start_sample)

                clip_start_in_clip = max(0, start_sample - clip.start_sample)
                samples_to_read = clip_buffer_end - clip_buffer_start

                if samples_to_read <= 0:
                    continue

                # Get audio data
                audio_chunk = clip.audio_data[clip_start_in_clip:clip_start_in_clip + samples_to_read]

                # Apply clip volume and pan
                clip_volume = clip.volume
                clip_pan = max(-1.0, min(1.0, clip.pan))
                left_gain = clip_volume * (1.0 - max(0, clip_pan))
                right_gain = clip_volume * (1.0 + min(0, clip_pan))

                # Mix into track buffer
                if clip.is_stereo:
                    track_buffer[clip_buffer_start:clip_buffer_end, 0] += audio_chunk[:, 0] * left_gain
                    track_buffer[clip_buffer_start:clip_buffer_end, 1] += audio_chunk[:, 1] * right_gain
                else:
                    track_buffer[clip_buffer_start:clip_buffer_end, 0] += audio_chunk * left_gain
                    track_buffer[clip_buffer_start:clip_buffer_end, 1] += audio_chunk * right_gain

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

            output_buffer[:, 0] += track_buffer[:, 0] * left_track_gain
            output_buffer[:, 1] += track_buffer[:, 1] * right_track_gain

        # Apply master effects if any
        if self._master_effects.effect_count > 0 and not self._master_effects.is_bypassed():
            processed = self._master_effects.process(output_buffer)
            output_buffer[:] = processed

        # Apply master volume
        output_buffer *= self.master_volume

    def export_to_wav(self, filepath: str, start_sample: int = 0,
                      end_sample: Optional[int] = None) -> bool:
        """
        Export audio to WAV file (blocking).

        Args:
            filepath: Output file path
            start_sample: Start position in samples (default: 0)
            end_sample: End position in samples (default: auto-detect)

        Returns:
            True if export completed successfully
        """
        try:
            import scipy.io.wavfile as wav

            # Reset state
            self._cancel_requested = False
            self._state = ExportState.EXPORTING

            # Calculate total duration
            total_duration = self._calculate_total_duration()
            if end_sample is None:
                end_sample = total_duration
            else:
                end_sample = min(end_sample, total_duration)

            if end_sample <= start_sample:
                self._state = ExportState.ERROR
                self._progress.error_message = "No audio to export"
                return False

            total_samples = end_sample - start_sample
            self._progress = ExportProgress(
                state=ExportState.EXPORTING,
                progress=0.0,
                current_sample=0,
                total_samples=total_samples
            )

            # Allocate output buffer
            output_audio = np.zeros((total_samples, self.config.channels), dtype=np.float32)

            # Reset effects state
            for effects in self._track_effects.values():
                effects.reset()
            self._master_effects.reset()

            start_time = time.perf_counter()
            buffer_size = self.config.buffer_size
            current_pos = 0

            # Render in chunks
            while current_pos < total_samples:
                if self._cancel_requested:
                    self._state = ExportState.CANCELLED
                    self._progress.state = ExportState.CANCELLED
                    if self._completed_callback:
                        self._completed_callback(filepath, False, "Export cancelled")
                    return False

                # Calculate chunk size
                chunk_size = min(buffer_size, total_samples - current_pos)
                chunk_buffer = np.zeros((chunk_size, 2), dtype=np.float32)

                # Render this chunk
                self._render_buffer(chunk_buffer, start_sample + current_pos)

                # Copy to output
                output_audio[current_pos:current_pos + chunk_size] = chunk_buffer[:chunk_size]

                current_pos += chunk_size

                # Update progress
                elapsed = time.perf_counter() - start_time
                self._progress = ExportProgress(
                    state=ExportState.EXPORTING,
                    progress=current_pos / total_samples,
                    current_sample=current_pos,
                    total_samples=total_samples,
                    elapsed_seconds=elapsed
                )

                if self._progress_callback:
                    try:
                        self._progress_callback(self._progress)
                    except Exception:
                        pass

            # Normalize if requested
            if self.config.normalize:
                peak = np.max(np.abs(output_audio))
                if peak > 0:
                    target_peak = 10 ** (self.config.normalize_headroom_db / 20.0)
                    output_audio = output_audio * (target_peak / peak)

            # Clip to prevent overflow
            np.clip(output_audio, -1.0, 1.0, out=output_audio)

            # Convert to target bit depth
            if self.config.bit_depth == 16:
                output_int = (output_audio * 32767).astype(np.int16)
            elif self.config.bit_depth == 24:
                # scipy doesn't support 24-bit, use 32-bit
                output_int = (output_audio * 2147483647).astype(np.int32)
            elif self.config.bit_depth == 32:
                output_int = (output_audio * 2147483647).astype(np.int32)
            else:
                output_int = (output_audio * 32767).astype(np.int16)

            # Ensure output directory exists
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write WAV file
            wav.write(str(output_path), self.config.sample_rate, output_int)

            # Update state
            elapsed = time.perf_counter() - start_time
            self._state = ExportState.COMPLETED
            self._progress = ExportProgress(
                state=ExportState.COMPLETED,
                progress=1.0,
                current_sample=total_samples,
                total_samples=total_samples,
                elapsed_seconds=elapsed
            )

            if self._progress_callback:
                try:
                    self._progress_callback(self._progress)
                except Exception:
                    pass

            if self._completed_callback:
                try:
                    self._completed_callback(filepath, True, None)
                except Exception:
                    pass

            return True

        except Exception as e:
            self._state = ExportState.ERROR
            self._progress.state = ExportState.ERROR
            self._progress.error_message = str(e)

            if self._completed_callback:
                try:
                    self._completed_callback(filepath, False, str(e))
                except Exception:
                    pass

            return False

    def export_to_wav_async(self, filepath: str, start_sample: int = 0,
                            end_sample: Optional[int] = None) -> threading.Thread:
        """
        Export audio to WAV file (non-blocking).

        Args:
            filepath: Output file path
            start_sample: Start position in samples (default: 0)
            end_sample: End position in samples (default: auto-detect)

        Returns:
            Thread object for the export operation
        """
        thread = threading.Thread(
            target=self.export_to_wav,
            args=(filepath, start_sample, end_sample),
            daemon=True
        )
        thread.start()
        return thread

    def get_export_info(self) -> Dict[str, Any]:
        """Get information about the current export setup."""
        total_duration = self._calculate_total_duration()
        duration_seconds = total_duration / self.config.sample_rate

        return {
            'clip_count': len(self._clips),
            'track_count': len(set(clip.track_id for clip in self._clips.values())),
            'total_samples': total_duration,
            'duration_seconds': duration_seconds,
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'bit_depth': self.config.bit_depth,
            'normalize': self.config.normalize,
            'master_volume': self.master_volume,
            'state': self._state.value,
        }


# Standalone test
if __name__ == "__main__":
    print("Audio Export Test")
    print("=" * 40)

    # Create exporter
    config = ExportConfig(sample_rate=44100, channels=2, bit_depth=16)
    exporter = AudioExporter(config)

    # Generate test tones
    duration_seconds = 2.0
    sample_rate = 44100
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, False)

    # Tone 1: 440Hz sine wave
    tone1 = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Tone 2: 660Hz sine wave (starting later)
    tone2 = (np.sin(2 * np.pi * 660 * t) * 0.3).astype(np.float32)

    # Add clips
    exporter.add_clip(
        clip_id="clip1",
        track_id="track1",
        audio_data=tone1,
        sample_rate=sample_rate,
        start_sample=0,
        volume=0.8,
        pan=-0.5  # Left
    )

    exporter.add_clip(
        clip_id="clip2",
        track_id="track2",
        audio_data=tone2,
        sample_rate=sample_rate,
        start_sample=sample_rate,  # Start at 1 second
        volume=0.6,
        pan=0.5  # Right
    )

    # Set track settings
    exporter.set_track_settings("track1", volume=0.9)
    exporter.set_track_settings("track2", volume=0.7)

    # Progress callback
    def on_progress(progress: ExportProgress):
        print(f"Progress: {progress.progress * 100:.1f}% ({progress.current_sample}/{progress.total_samples})")

    def on_completed(filepath: str, success: bool, error: Optional[str]):
        if success:
            print(f"Export completed: {filepath}")
        else:
            print(f"Export failed: {error}")

    exporter.set_progress_callback(on_progress)
    exporter.set_completed_callback(on_completed)

    # Get export info
    info = exporter.get_export_info()
    print(f"\nExport Info:")
    print(f"  Clips: {info['clip_count']}")
    print(f"  Tracks: {info['track_count']}")
    print(f"  Duration: {info['duration_seconds']:.2f}s")
    print(f"  Sample Rate: {info['sample_rate']}Hz")

    # Export
    print("\nExporting...")
    success = exporter.export_to_wav("test_export.wav")

    if success:
        print("\nExport successful!")
        print("AudioExporter imported successfully")
    else:
        print(f"\nExport failed: {exporter.get_progress().error_message}")
