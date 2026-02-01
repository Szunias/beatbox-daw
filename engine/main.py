"""
BeatBox DAW Engine

Main entry point for the Python audio processing engine.
Provides WebSocket server for communication with Tauri frontend.
Supports both BeatBox-to-MIDI conversion and full DAW functionality.
"""

import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

from audio_capture import AudioCapture, AudioConfig
from audio_recorder import AudioRecorder, RecordingConfig, RecordingMetadata
from audio_playback import AudioPlayback, AudioPlaybackConfig
from audio_export import AudioExporter, ExportConfig, ExportProgress, ExportState
from onset_detector import OnsetDetector, OnsetConfig
from classifier.inference import BeatboxClassifier, RuleBasedClassifier, InferenceConfig
from midi_output import MidiOutput, MidiEvent
from transport import Transport, TransportState
from project import Project, ProjectManager, Track, Clip, MidiNote
from scheduler import MidiScheduler


class EngineState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class EngineConfig:
    """Engine configuration."""
    sample_rate: int = 44100
    buffer_size: int = 512
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    use_ml_classifier: bool = False  # Use rule-based by default until model is trained
    confidence_threshold: float = 0.5


@dataclass
class DrumEvent:
    """Drum event to send to frontend."""
    drum_class: str
    confidence: float
    midi_note: int
    velocity: int
    timestamp: float


class BeatBoxDawEngine:
    """
    Main engine coordinating audio capture, classification, MIDI output,
    and full DAW functionality (transport, project, scheduler).
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.state = EngineState.STOPPED

        # === BeatBox Detection Components ===

        # Audio capture
        audio_config = AudioConfig(
            sample_rate=self.config.sample_rate,
            buffer_size=self.config.buffer_size
        )
        self.audio_capture = AudioCapture(audio_config)

        # Onset detection
        onset_config = OnsetConfig(sample_rate=self.config.sample_rate)
        self.onset_detector = OnsetDetector(onset_config)

        # Classification
        if self.config.use_ml_classifier:
            inference_config = InferenceConfig(
                sample_rate=self.config.sample_rate,
                confidence_threshold=self.config.confidence_threshold
            )
            self.classifier = BeatboxClassifier(inference_config)
        else:
            self.classifier = RuleBasedClassifier(self.config.sample_rate)

        # MIDI output
        self.midi_output = MidiOutput()

        # Audio buffer for classification
        self.classification_buffer = []
        self.classification_window_samples = int(0.1 * self.config.sample_rate)  # 100ms

        # Statistics
        self.events_detected = 0
        self.start_time: Optional[float] = None

        # === DAW Components ===

        # Transport
        self.transport = Transport()

        # Project management
        self.project_manager = ProjectManager()
        self._project: Optional[Project] = None

        # MIDI Scheduler
        self.scheduler = MidiScheduler(self.transport)
        self.scheduler.set_note_on_callback(self._on_scheduled_note_on)
        self.scheduler.set_note_off_callback(self._on_scheduled_note_off)
        self.scheduler.set_click_callback(self._on_click)

        # Audio Playback
        playback_config = AudioPlaybackConfig(
            sample_rate=self.config.sample_rate,
            buffer_size=self.config.buffer_size,
            channels=2  # Stereo output
        )
        self.audio_playback = AudioPlayback(self.transport, playback_config)
        self.audio_playback.set_clip_started_callback(self._on_audio_clip_started)
        self.audio_playback.set_clip_ended_callback(self._on_audio_clip_ended)

        # Connect scheduler audio clip callbacks to audio playback
        self.scheduler.set_audio_clip_start_callback(self._on_scheduled_audio_clip_start)
        self.scheduler.set_audio_clip_stop_callback(self._on_scheduled_audio_clip_stop)

        # Recording buffer for beatbox events
        self._beatbox_recording: list = []
        self._beatbox_recording_start: Optional[float] = None

        # === Audio Track Recording ===

        # Audio recorder for track recording
        recording_config = RecordingConfig(
            sample_rate=self.config.sample_rate,
            channels=1,  # Mono recording
            max_duration=3600.0  # 1 hour max
        )
        self.audio_recorder = AudioRecorder(recording_config)

        # Track recording state
        self._recording_track_id: Optional[str] = None
        self._audio_recording_active = False

        # Setup audio recorder callbacks
        self.audio_recorder.on_recording_stopped = self._on_audio_recording_stopped

        # === Audio Export ===

        # Audio exporter for offline rendering and mixdown
        export_config = ExportConfig(
            sample_rate=self.config.sample_rate,
            channels=2,  # Stereo output
            bit_depth=16,
            normalize=True
        )
        self.audio_exporter = AudioExporter(export_config)

        # Setup exporter callbacks
        self.audio_exporter.set_progress_callback(self._on_export_progress)
        self.audio_exporter.set_completed_callback(self._on_export_completed)

        # WebSocket clients
        self.clients: Set[WebSocketServerProtocol] = set()

        # Create default project
        self._project = self.project_manager.new_project("New Project")

        # Setup transport callbacks for broadcasting position to clients
        self.transport.add_position_callback(self._on_transport_position)
        self.transport.add_state_callback(self._on_transport_state_change)

        # Position broadcast throttling (broadcast every ~50ms)
        # Use perf_counter for higher precision timing (matches transport.py pattern)
        self._last_position_broadcast = time.perf_counter()
        self._position_broadcast_interval = 0.05  # 50ms

        # Valid state transitions for transport controls
        self._valid_state_transitions = {
            TransportState.STOPPED: [TransportState.PLAYING, TransportState.RECORDING, TransportState.PRE_ROLL],
            TransportState.PLAYING: [TransportState.PAUSED, TransportState.STOPPED, TransportState.RECORDING, TransportState.PRE_ROLL],
            TransportState.PAUSED: [TransportState.PLAYING, TransportState.STOPPED, TransportState.RECORDING, TransportState.PRE_ROLL],
            TransportState.RECORDING: [TransportState.STOPPED, TransportState.PAUSED],
            TransportState.PRE_ROLL: [TransportState.STOPPED, TransportState.RECORDING],
        }

        # Store event loop reference for thread-safe callbacks
        # Will be set when server starts
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # === Audio Click Synthesis (fallback when MIDI unavailable) ===
        self._click_samples = self._generate_click_samples()
        self._click_stream: Optional[sd.OutputStream] = None

    # === Audio Click Synthesis ===

    def _generate_click_samples(self) -> dict:
        """Generate audio samples for metronome clicks.

        Creates short percussive sounds for downbeat (beat 1) and regular beats.
        Uses sine waves with exponential decay for clean, audible clicks.

        Returns:
            Dictionary with 'downbeat' and 'beat' audio samples as np.ndarray.
        """
        sample_rate = self.config.sample_rate
        duration = 0.05  # 50ms click duration

        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Exponential decay envelope
        decay = np.exp(-t * 50)  # Fast decay for percussive sound

        # Downbeat click: higher frequency (1200 Hz), higher amplitude
        downbeat_freq = 1200.0
        downbeat = (np.sin(2 * np.pi * downbeat_freq * t) * decay * 0.6).astype(np.float32)

        # Regular beat click: slightly lower frequency (900 Hz), lower amplitude
        beat_freq = 900.0
        beat = (np.sin(2 * np.pi * beat_freq * t) * decay * 0.4).astype(np.float32)

        return {
            'downbeat': downbeat,
            'beat': beat
        }

    def _play_click_audio(self, is_downbeat: bool) -> None:
        """Play audio click using sounddevice (fallback when MIDI unavailable).

        Args:
            is_downbeat: True for beat 1 (downbeat), False for other beats.
        """
        try:
            # Select appropriate click sample
            sample_key = 'downbeat' if is_downbeat else 'beat'
            audio_data = self._click_samples[sample_key]

            # Play the click asynchronously (non-blocking)
            sd.play(audio_data, self.config.sample_rate, blocking=False)
        except Exception:
            # Silently ignore audio playback errors
            pass

    # === BeatBox Detection ===

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        message_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_json) for client in self.clients],
            return_exceptions=True
        )

    def _audio_callback(self, audio: np.ndarray) -> None:
        """Process incoming audio buffer for beatbox detection and recording."""
        # Feed audio to AudioRecorder if track recording is active
        if self._audio_recording_active:
            self.audio_recorder.add_audio_buffer(audio)

        if self.state != EngineState.RUNNING:
            return

        # Add to classification buffer
        self.classification_buffer.extend(audio)

        # Detect onsets
        onsets = self.onset_detector.process(audio)

        for offset, strength in onsets:
            # Extract audio around onset for classification
            if len(self.classification_buffer) >= self.classification_window_samples:
                # Get window centered on onset
                window_start = max(0, len(self.classification_buffer) - self.classification_window_samples)
                classification_audio = np.array(
                    self.classification_buffer[window_start:window_start + self.classification_window_samples]
                )

                # Classify
                drum_class, confidence, probs = self.classifier.classify(classification_audio)

                if drum_class and confidence >= self.config.confidence_threshold:
                    # Calculate velocity from audio amplitude and onset strength
                    amplitude = np.abs(classification_audio).max()
                    velocity = int(min(127, max(30, amplitude * 127 * strength)))

                    # Get MIDI note
                    midi_note = self.classifier.get_midi_note(drum_class)

                    # Send MIDI
                    self.midi_output.send_note(midi_note, velocity)

                    # Create event
                    timestamp = time.time() - (self.start_time or time.time())
                    event = DrumEvent(
                        drum_class=drum_class,
                        confidence=confidence,
                        midi_note=midi_note,
                        velocity=velocity,
                        timestamp=timestamp
                    )

                    self.events_detected += 1

                    # Add to recording buffer if recording
                    if self._beatbox_recording_start is not None:
                        self._beatbox_recording.append(asdict(event))

                    # Broadcast to clients (non-blocking)
                    asyncio.create_task(self.broadcast({
                        'type': 'drum_event',
                        'data': asdict(event)
                    }))

        # Trim buffer to prevent unbounded growth
        max_buffer = self.classification_window_samples * 2
        if len(self.classification_buffer) > max_buffer:
            self.classification_buffer = self.classification_buffer[-max_buffer:]

        # Send audio level for visualization
        level = float(np.abs(audio).max())
        asyncio.create_task(self.broadcast({
            'type': 'audio_level',
            'data': {'level': level}
        }))

    # === Transport Broadcast Callbacks ===

    def _broadcast_threadsafe(self, message: dict):
        """Thread-safe broadcast helper for callbacks from non-async contexts.

        Uses run_coroutine_threadsafe when called from a thread (e.g., transport update thread),
        or create_task when already in an async context.
        """
        if self._event_loop is None:
            return

        try:
            # Use run_coroutine_threadsafe for thread-safe scheduling
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._event_loop)
        except Exception as e:
            # Silently ignore errors during shutdown
            pass

    def _on_transport_position(self, tick: int):
        """Broadcast transport position to all clients (throttled).

        Uses perf_counter for high-precision timing to match transport.py pattern.
        Includes bar/beat information for accurate frontend synchronization.
        """
        current_time = time.perf_counter()
        if current_time - self._last_position_broadcast >= self._position_broadcast_interval:
            self._last_position_broadcast = current_time

            # Get bar and beat for enhanced position data
            bar, beat = self.transport._ticks_to_bar_beat(max(0, tick))

            self._broadcast_threadsafe({
                'type': 'transport_position',
                'data': {
                    'tick': tick,
                    'bar': bar,
                    'beat': beat,
                    'state': self.transport.state.value,
                    'bpm': self.transport.bpm,
                    'timestamp': current_time  # High-precision timestamp for sync
                }
            })

    def _on_transport_state_change(self, state: TransportState):
        """Broadcast transport state change to all clients.

        Includes full position and timing data for accurate frontend synchronization.
        """
        current_tick = self.transport.current_tick
        bar, beat = self.transport._ticks_to_bar_beat(max(0, current_tick))

        self._broadcast_threadsafe({
            'type': 'transport_state',
            'data': {
                'state': state.value,
                'tick': current_tick,
                'bar': bar,
                'beat': beat,
                'bpm': self.transport.bpm,
                'timestamp': time.perf_counter()
            }
        })

    # === DAW Scheduler Callbacks ===

    def _on_scheduled_note_on(self, channel: int, note: int, velocity: int, track_id: str):
        """Handle note on from scheduler during playback."""
        track = self.project_manager.get_track(track_id)
        if track and not track.muted:
            adjusted_velocity = int(velocity * track.volume)
            self.midi_output.send_note(note, adjusted_velocity, channel=channel)

    def _on_scheduled_note_off(self, channel: int, note: int, track_id: str):
        """Handle note off from scheduler."""
        self.midi_output.send_note_off(note, channel=channel)

    def _on_click(self, bar: int, beat: int):
        """Handle metronome click.

        Uses MIDI output if available, otherwise falls back to audio synthesis.
        """
        if self.transport.config.click_enabled:
            is_downbeat = (beat == 1)

            # Try MIDI first, fall back to audio synthesis if unavailable
            if self.midi_output.is_connected:
                velocity = 100 if is_downbeat else 70
                note = 76 if is_downbeat else 77  # Woodblock sounds
                self.midi_output.send_note(note, velocity, duration=0.05, channel=9)
            else:
                # Audio click synthesis fallback
                self._play_click_audio(is_downbeat)

            self._broadcast_threadsafe({
                'type': 'click',
                'data': {'bar': bar, 'beat': beat}
            })

    # === Audio Playback Callbacks ===

    def _on_scheduled_audio_clip_start(self, clip_id: str, track_id: str, file_path: str,
                                        duration_ticks: int, volume: float, pan: float):
        """Handle audio clip start from scheduler - load and play the clip."""
        track = self.project_manager.get_track(track_id)
        if track and track.muted:
            return

        # Get track volume/pan/mute settings
        track_volume = track.volume if track else 1.0
        track_pan = track.pan if track else 0.0

        # Calculate start tick based on clip position
        clip_info = self.scheduler._audio_clips.get(clip_id, {})
        start_tick = clip_info.get('start_tick', 0)

        # Load audio file into playback system if not already loaded
        if not self.audio_playback._clips.get(clip_id):
            self.audio_playback.load_audio_file(
                filepath=file_path,
                clip_id=clip_id,
                track_id=track_id,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                volume=volume * track_volume,
                pan=pan + track_pan
            )

        # Update track settings
        self.audio_playback.set_track_settings(
            track_id=track_id,
            volume=track_volume,
            pan=track_pan,
            muted=track.muted if track else False
        )

    def _on_scheduled_audio_clip_stop(self, clip_id: str, track_id: str):
        """Handle audio clip stop from scheduler."""
        # AudioPlayback handles clip stopping automatically via events
        pass

    def _on_audio_clip_started(self, clip_id: str, track_id: str):
        """Callback when audio clip actually starts playing."""
        asyncio.create_task(self.broadcast({
            'type': 'audio_clip_started',
            'data': {'clip_id': clip_id, 'track_id': track_id}
        }))

    def _on_audio_clip_ended(self, clip_id: str, track_id: str):
        """Callback when audio clip finishes playing."""
        asyncio.create_task(self.broadcast({
            'type': 'audio_clip_ended',
            'data': {'clip_id': clip_id, 'track_id': track_id}
        }))

    # === Engine Controls ===

    def start(self) -> bool:
        """Start the beatbox detection engine."""
        if self.state == EngineState.RUNNING:
            return True

        # Connect MIDI
        if not self.midi_output.connect():
            print("Warning: MIDI output not available")

        # Setup audio callback
        self.audio_capture.add_callback(self._audio_callback)

        # Start capture
        self.audio_capture.start()

        self.state = EngineState.RUNNING
        self.start_time = time.time()
        self.events_detected = 0
        self._beatbox_recording = []
        self._beatbox_recording_start = time.time()

        print("BeatBox engine started")
        return True

    def stop(self) -> None:
        """Stop the beatbox detection engine."""
        if self.state == EngineState.STOPPED:
            return

        self.audio_capture.stop()
        self.onset_detector.reset()
        self.classification_buffer = []

        self.state = EngineState.STOPPED
        print(f"BeatBox engine stopped. Events detected: {self.events_detected}")

    def pause(self) -> None:
        """Pause the engine."""
        self.state = EngineState.PAUSED

    def resume(self) -> None:
        """Resume the engine."""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING

    # === DAW Transport Controls ===

    def _validate_state_transition(self, target_state: TransportState) -> bool:
        """Validate if a state transition is allowed.

        Args:
            target_state: The target state to transition to.

        Returns:
            True if transition is valid, False otherwise.
        """
        current_state = self.transport.state
        if current_state == target_state:
            return True  # Already in target state
        valid_targets = self._valid_state_transitions.get(current_state, [])
        return target_state in valid_targets

    def transport_play(self) -> bool:
        """Start DAW playback.

        Validates state transition before starting playback.
        Returns False if transition is not valid.
        """
        if not self._validate_state_transition(TransportState.PLAYING):
            return False

        if self._project:
            self.scheduler.load_project(self._project)
        return self.transport.play()

    def transport_pause(self) -> bool:
        """Pause DAW playback.

        Validates state transition before pausing.
        Returns True on success, False if transition is not valid.
        """
        if not self._validate_state_transition(TransportState.PAUSED):
            return False

        self.transport.pause()
        return True

    def transport_stop(self) -> bool:
        """Stop DAW playback.

        Always succeeds as stop is valid from any state.
        Returns True on success.
        """
        # Stop is always valid - it resets to initial state
        self.transport.stop()
        return True

    def transport_record(self) -> bool:
        """Toggle DAW recording.

        Validates state transition before starting recording.
        When starting, automatically starts track recording on armed tracks.
        When stopping, automatically stops track recording and creates clips.
        Returns True if recording started, False otherwise.
        """
        current_state = self.transport.state
        # If already recording or in pre-roll, we're stopping
        if current_state in (TransportState.RECORDING, TransportState.PRE_ROLL):
            # Stop track recording if active
            if self._audio_recording_active:
                self.stop_track_recording()
            return self.transport.record()

        # Starting recording - validate transition
        if not self._validate_state_transition(TransportState.RECORDING) and \
           not self._validate_state_transition(TransportState.PRE_ROLL):
            return False

        # Find armed track and start recording on it
        armed_track = self._get_armed_track()
        if armed_track:
            self.start_track_recording(armed_track.id)

        return self.transport.record()

    def _get_armed_track(self) -> Optional[Track]:
        """Get the first armed track in the project.

        Returns:
            The first armed track, or None if no track is armed
        """
        if not self._project:
            return None

        for track in self._project.tracks:
            if track.armed and track.type in ('audio', 'drum', 'midi'):
                return track
        return None

    def set_track_armed(self, track_id: str, armed: bool) -> bool:
        """Set a track's armed state for recording.

        Args:
            track_id: ID of the track to arm/disarm
            armed: True to arm, False to disarm

        Returns:
            True if successful, False if track not found
        """
        track = self.project_manager.get_track(track_id)
        if not track:
            return False

        # If arming this track, disarm all other tracks (only one can be armed at a time)
        if armed and self._project:
            for t in self._project.tracks:
                t.armed = False

        track.armed = armed
        return True

    def transport_seek(self, tick: int) -> bool:
        """Seek to position.

        Validates the tick value before seeking.

        Args:
            tick: Position in ticks to seek to (must be non-negative).

        Returns:
            True on success, False if tick is invalid.
        """
        if tick < 0:
            return False
        self.transport.seek(tick)
        return True

    # === Project Management ===

    def new_project(self, name: str = "New Project") -> dict:
        """Create a new project."""
        self._project = self.project_manager.new_project(name)
        self.scheduler.load_project(self._project)
        return self._project.to_dict()

    def get_project(self) -> Optional[dict]:
        """Get current project data."""
        if self._project:
            return self._project.to_dict()
        return None

    def add_track(self, track_type: str = "drum", name: str = None) -> Optional[dict]:
        """Add a track to the project."""
        if not self._project:
            return None

        track = self.project_manager.add_track(track_type, name)
        return {
            'id': track.id,
            'name': track.name,
            'type': track.type,
            'color': track.color
        }

    def add_beatbox_clip(self, track_id: str, start_tick: int = 0) -> Optional[dict]:
        """Add recorded beatbox events as a clip."""
        if not self._project or not self._beatbox_recording:
            return None

        track = self.project_manager.get_track(track_id)
        if not track:
            return None

        # Convert beatbox events to MIDI notes
        from project import MidiClipData
        notes = []
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm

        for event in self._beatbox_recording:
            tick = int(event.get('timestamp', 0) * (bpm / 60.0) * ppqn)
            notes.append(MidiNote.create(
                pitch=event.get('midi_note', 36),
                velocity=event.get('velocity', 100),
                start_tick=tick,
                duration=ppqn // 4
            ))

        if not notes:
            return None

        # Calculate clip duration
        max_tick = max(n.start_tick + n.duration for n in notes)
        duration = ((max_tick // (ppqn * 4)) + 1) * ppqn * 4

        # Create clip
        clip = Clip.create_midi(
            f"BeatBox {time.strftime('%H:%M:%S')}",
            start_tick,
            duration,
            notes,
            color="#e94560"
        )

        track.clips.append(clip)
        self.scheduler.load_track(track)

        # Clear recording buffer
        self._beatbox_recording = []
        self._beatbox_recording_start = time.time()

        print(f"Added beatbox clip with {len(notes)} notes")
        return {
            'id': clip.id,
            'name': clip.name,
            'notes': len(notes)
        }

    # === Audio Clip Management ===

    def load_audio_clip(self, filepath: str, clip_id: str, track_id: str,
                        start_tick: int, duration_ticks: int,
                        volume: float = 1.0, pan: float = 0.0) -> bool:
        """Load an audio file as a clip for playback.

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
        # Get track settings for volume/pan adjustment
        track = self.project_manager.get_track(track_id)
        track_volume = track.volume if track else 1.0
        track_pan = track.pan if track else 0.0
        track_muted = track.muted if track else False

        # Load into audio playback system
        success = self.audio_playback.load_audio_file(
            filepath=filepath,
            clip_id=clip_id,
            track_id=track_id,
            start_tick=start_tick,
            duration_ticks=duration_ticks,
            volume=volume,
            pan=pan
        )

        if success:
            # Update track settings in playback
            self.audio_playback.set_track_settings(
                track_id=track_id,
                volume=track_volume,
                pan=track_pan,
                muted=track_muted
            )

            # Also register with scheduler for event timing
            self.scheduler.load_audio_clip(
                clip_id=clip_id,
                track_id=track_id,
                file_path=filepath,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                volume=volume,
                pan=pan
            )

        return success

    def load_audio_clip_from_recording(self, clip_id: str, track_id: str,
                                        start_tick: int) -> bool:
        """Load audio from the current recording buffer as a clip.

        Args:
            clip_id: Unique ID for this clip
            track_id: ID of the track this clip belongs to
            start_tick: Timeline position where clip starts

        Returns:
            True if loaded successfully
        """
        if not self.audio_recorder.is_recording and self.audio_recorder.buffer is None:
            return False

        # Get the recorded audio data
        audio_data = self.audio_recorder.get_audio_data()
        if audio_data is None or len(audio_data) == 0:
            return False

        # Calculate duration in ticks
        duration_seconds = len(audio_data) / self.config.sample_rate
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm
        ticks_per_second = (bpm / 60.0) * ppqn
        duration_ticks = int(duration_seconds * ticks_per_second)

        # Get track settings
        track = self.project_manager.get_track(track_id)
        track_volume = track.volume if track else 1.0
        track_pan = track.pan if track else 0.0
        track_muted = track.muted if track else False

        # Load into audio playback system
        success = self.audio_playback.load_audio_data(
            audio_data=audio_data,
            sample_rate=self.config.sample_rate,
            clip_id=clip_id,
            track_id=track_id,
            start_tick=start_tick,
            duration_ticks=duration_ticks,
            volume=1.0,
            pan=0.0
        )

        if success:
            self.audio_playback.set_track_settings(
                track_id=track_id,
                volume=track_volume,
                pan=track_pan,
                muted=track_muted
            )

        return success

    def unload_audio_clip(self, clip_id: str) -> bool:
        """Remove a loaded audio clip.

        Args:
            clip_id: ID of the clip to unload

        Returns:
            True if unloaded successfully
        """
        playback_success = self.audio_playback.unload_clip(clip_id)
        scheduler_success = self.scheduler.unload_audio_clip(clip_id)
        return playback_success or scheduler_success

    def update_track_settings(self, track_id: str, volume: float = None,
                               pan: float = None, muted: bool = None) -> bool:
        """Update track mixer settings for audio playback.

        Args:
            track_id: ID of the track to update
            volume: New volume (0.0-1.0)
            pan: New pan (-1.0 left, 0.0 center, 1.0 right)
            muted: New mute state

        Returns:
            True if updated
        """
        # Update audio playback settings
        self.audio_playback.set_track_settings(
            track_id=track_id,
            volume=volume,
            pan=pan,
            muted=muted
        )

        # Also update the track in project
        track = self.project_manager.get_track(track_id)
        if track:
            if volume is not None:
                track.volume = max(0.0, min(1.0, volume))
            if pan is not None:
                track.pan = max(-1.0, min(1.0, pan))
            if muted is not None:
                track.muted = muted
                # Reload track events if mute state changed
                self.scheduler.load_track(track)

        return True

    def set_master_volume(self, volume: float) -> bool:
        """Set master output volume.

        Args:
            volume: Master volume (0.0-1.0)

        Returns:
            True if set
        """
        self.audio_playback.master_volume = max(0.0, min(1.0, volume))
        return True

    def get_audio_playback_status(self) -> dict:
        """Get audio playback status.

        Returns:
            Dictionary with playback info
        """
        return {
            'active_clips': self.audio_playback.get_active_clips(),
            'loaded_clips': self.audio_playback.get_loaded_clips(),
            'master_volume': self.audio_playback.master_volume,
            'device_id': self.audio_playback.current_device_id
        }

    def set_audio_output_device(self, device_id: int) -> bool:
        """Set audio output device.

        Args:
            device_id: ID of the output device

        Returns:
            True if set successfully
        """
        return self.audio_playback.set_device(device_id)

    # === Track Effects Management ===

    def add_track_effect(self, track_id: str, effect_type: str,
                         effect_id: Optional[str] = None,
                         position: Optional[int] = None) -> Optional[str]:
        """
        Add an effect to a track's effect chain.

        Args:
            track_id: ID of the track
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect
            position: Optional position in chain

        Returns:
            Effect ID if successful, None if effect type not found
        """
        return self.audio_playback.add_track_effect(track_id, effect_type, effect_id, position)

    def remove_track_effect(self, track_id: str, effect_id: str) -> bool:
        """Remove an effect from a track's effect chain."""
        return self.audio_playback.remove_track_effect(track_id, effect_id)

    def move_track_effect(self, track_id: str, effect_id: str, new_position: int) -> bool:
        """Move an effect to a new position in a track's effect chain."""
        return self.audio_playback.move_track_effect(track_id, effect_id, new_position)

    def set_track_effect_parameter(self, track_id: str, effect_id: str,
                                    param_name: str, value: float) -> bool:
        """Set a parameter on a track's effect."""
        return self.audio_playback.set_track_effect_parameter(
            track_id, effect_id, param_name, value
        )

    def get_track_effect_parameters(self, track_id: str, effect_id: str) -> Optional[dict]:
        """Get parameters of a track's effect."""
        return self.audio_playback.get_track_effect_parameters(track_id, effect_id)

    def get_track_effects_chain(self, track_id: str) -> list:
        """Get information about all effects in a track's chain."""
        return self.audio_playback.get_track_effects_chain(track_id)

    def bypass_track_effects(self, track_id: str, bypassed: bool = True) -> bool:
        """Bypass all effects for a track."""
        return self.audio_playback.bypass_track_effects(track_id, bypassed)

    def reset_track_effects(self, track_id: str) -> bool:
        """Reset all effects in a track's chain."""
        return self.audio_playback.reset_track_effects(track_id)

    def clear_track_effects(self, track_id: str) -> bool:
        """Remove all effects from a track's chain."""
        return self.audio_playback.clear_track_effects(track_id)

    # === Master Effects Management ===

    def add_master_effect(self, effect_type: str,
                          effect_id: Optional[str] = None,
                          position: Optional[int] = None) -> Optional[str]:
        """
        Add an effect to the master effect chain.

        Args:
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect
            position: Optional position in chain

        Returns:
            Effect ID if successful, None if effect type not found
        """
        return self.audio_playback.add_master_effect(effect_type, effect_id, position)

    def remove_master_effect(self, effect_id: str) -> bool:
        """Remove an effect from the master effect chain."""
        return self.audio_playback.remove_master_effect(effect_id)

    def move_master_effect(self, effect_id: str, new_position: int) -> bool:
        """Move an effect to a new position in the master effect chain."""
        return self.audio_playback.move_master_effect(effect_id, new_position)

    def set_master_effect_parameter(self, effect_id: str,
                                     param_name: str, value: float) -> bool:
        """Set a parameter on a master effect."""
        return self.audio_playback.set_master_effect_parameter(effect_id, param_name, value)

    def get_master_effect_parameters(self, effect_id: str) -> Optional[dict]:
        """Get parameters of a master effect."""
        return self.audio_playback.get_master_effect_parameters(effect_id)

    def get_master_effects_chain(self) -> list:
        """Get information about all effects in the master chain."""
        return self.audio_playback.get_master_effects_chain()

    def bypass_master_effects(self, bypassed: bool = True) -> None:
        """Bypass all master effects."""
        self.audio_playback.bypass_master_effects(bypassed)

    def reset_master_effects(self) -> None:
        """Reset all master effects."""
        self.audio_playback.reset_master_effects()

    def clear_master_effects(self) -> None:
        """Remove all effects from the master chain."""
        self.audio_playback.clear_master_effects()

    def is_master_effects_bypassed(self) -> bool:
        """Check if master effects are bypassed."""
        return self.audio_playback.is_master_effects_bypassed()

    def get_available_effect_types(self) -> list:
        """Get list of available effect types."""
        return self.audio_playback.get_available_effect_types()

    # === Status ===

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            'state': self.state.value,
            'events_detected': self.events_detected,
            'midi_connected': self.midi_output.is_connected,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'transport': self.transport.get_status(),
            'recorded_events': len(self._beatbox_recording)
        }

    def start_recording(self) -> None:
        """Start MIDI recording."""
        self.midi_output.start_recording()
        self._beatbox_recording = []
        self._beatbox_recording_start = time.time()

    def stop_recording(self) -> list:
        """Stop MIDI recording and return events."""
        events = self.midi_output.stop_recording()
        return [asdict(e) if isinstance(e, MidiEvent) else e for e in events]

    def export_midi(self, filename: str, bpm: int = 120) -> bool:
        """Export recorded MIDI to file."""
        return self.midi_output.export_midi_file(filename, bpm=bpm)

    # === Audio Track Recording ===

    def start_track_recording(self, track_id: str) -> bool:
        """Start recording audio to a specific track.

        Args:
            track_id: ID of the track to record to

        Returns:
            True if recording started successfully, False otherwise
        """
        if self._audio_recording_active:
            return False

        # Verify track exists
        track = self.project_manager.get_track(track_id)
        if not track:
            return False

        # Calculate timeline position in seconds
        current_tick = self.transport.current_tick
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm
        start_time = current_tick / ppqn * (60.0 / bpm)

        # Start audio recording
        if not self.audio_recorder.start_recording(start_time):
            return False

        self._recording_track_id = track_id
        self._audio_recording_active = True

        # Ensure audio capture is running and _audio_callback is registered
        # _audio_callback handles recording when _audio_recording_active is True (line 266-268)
        if self._audio_callback not in self.audio_capture.callbacks:
            self.audio_capture.add_callback(self._audio_callback)

        if not self.audio_capture.is_running:
            self.audio_capture.start()

        self._broadcast_threadsafe({
            'type': 'track_recording_started',
            'data': {
                'track_id': track_id,
                'start_time': start_time
            }
        })

        return True

    def stop_track_recording(self) -> Optional[dict]:
        """Stop audio track recording and return recording info.

        Returns:
            Recording info dictionary or None if not recording
        """
        if not self._audio_recording_active:
            return None

        # Stop audio recording
        metadata = self.audio_recorder.stop_recording()

        # Note: _audio_callback stays registered - it handles multiple purposes
        # and only records when _audio_recording_active is True

        track_id = self._recording_track_id
        self._recording_track_id = None
        self._audio_recording_active = False

        if metadata is None:
            return None

        result = {
            'track_id': track_id,
            'duration': metadata.duration,
            'sample_rate': metadata.sample_rate,
            'num_samples': metadata.num_samples,
            'start_time': metadata.start_time
        }

        self._broadcast_threadsafe({
            'type': 'track_recording_stopped',
            'data': result
        })

        return result

    def export_track_recording(self, filename: str, normalize: bool = True) -> bool:
        """Export the current track recording to a WAV file.

        Args:
            filename: Output file path
            normalize: Whether to normalize audio

        Returns:
            True if export successful
        """
        return self.audio_recorder.export_wav(filename, normalize)

    def get_track_recording_info(self) -> dict:
        """Get information about the current track recording.

        Returns:
            Dictionary with recording info
        """
        info = self.audio_recorder.get_recording_info()
        info['track_id'] = self._recording_track_id
        return info

    def _track_recording_callback(self, audio: np.ndarray) -> None:
        """Callback for audio capture to feed audio to the recorder."""
        if self._audio_recording_active:
            self.audio_recorder.add_audio_buffer(audio)

    def _on_audio_recording_stopped(self, metadata: RecordingMetadata) -> None:
        """Callback when audio recording is stopped (e.g., max duration reached)."""
        if self._audio_recording_active:
            self._audio_recording_active = False
            self._broadcast_threadsafe({
                'type': 'track_recording_auto_stopped',
                'data': {
                    'track_id': self._recording_track_id,
                    'duration': metadata.duration,
                    'reason': 'max_duration_reached'
                }
            })

    # === Audio Export Callbacks ===

    def _on_export_progress(self, progress: ExportProgress) -> None:
        """Callback for export progress updates."""
        self._broadcast_threadsafe({
            'type': 'export_progress',
            'data': {
                'state': progress.state.value,
                'progress': progress.progress,
                'current_sample': progress.current_sample,
                'total_samples': progress.total_samples,
                'elapsed_seconds': progress.elapsed_seconds
            }
        })

    def _on_export_completed(self, filepath: str, success: bool, error: Optional[str]) -> None:
        """Callback for export completion."""
        self._broadcast_threadsafe({
            'type': 'export_completed',
            'data': {
                'filepath': filepath,
                'success': success,
                'error': error
            }
        })

    # === Audio Export Methods ===

    def prepare_export(self) -> bool:
        """
        Prepare audio export by loading all audio clips from the current project.
        Copies track settings and clip data to the exporter.

        Returns:
            True if export preparation successful
        """
        if not self._project:
            return False

        # Clear previous export data
        self.audio_exporter.clear_clips()

        # Get sample rate and timing info
        sample_rate = self.config.sample_rate
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm
        samples_per_tick = (sample_rate * 60.0) / (bpm * ppqn)

        # Load clips from all tracks
        clip_count = 0
        for track in self._project.tracks:
            # Set track settings for export
            self.audio_exporter.set_track_settings(
                track_id=track.id,
                volume=track.volume,
                pan=track.pan,
                muted=track.muted
            )

            # Load audio clips from playback system
            for clip in track.clips:
                if clip.type == 'audio' and clip.audio_data:
                    # Get clip audio data from audio playback if loaded
                    playback_clip = self.audio_playback._clips.get(clip.id)
                    if playback_clip and playback_clip.audio_data is not None:
                        # Calculate start position in samples
                        start_sample = int(clip.start_tick * samples_per_tick)

                        # Add clip to exporter
                        success = self.audio_exporter.add_clip(
                            clip_id=clip.id,
                            track_id=track.id,
                            audio_data=playback_clip.audio_data,
                            sample_rate=sample_rate,
                            start_sample=start_sample,
                            volume=clip.audio_data.get('volume', 1.0) if isinstance(clip.audio_data, dict) else 1.0,
                            pan=clip.audio_data.get('pan', 0.0) if isinstance(clip.audio_data, dict) else 0.0,
                            muted=track.muted
                        )
                        if success:
                            clip_count += 1

        # Set master volume
        self.audio_exporter.master_volume = self.audio_playback.master_volume

        return clip_count > 0

    def export_audio(self, filepath: str, start_tick: int = 0,
                     end_tick: Optional[int] = None) -> bool:
        """
        Export audio to a WAV file (blocking).

        Args:
            filepath: Output file path
            start_tick: Start position in ticks (default: 0)
            end_tick: End position in ticks (default: auto-detect)

        Returns:
            True if export completed successfully
        """
        # Prepare export with current project data
        if not self.prepare_export():
            return False

        # Convert ticks to samples
        sample_rate = self.config.sample_rate
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm
        samples_per_tick = (sample_rate * 60.0) / (bpm * ppqn)

        start_sample = int(start_tick * samples_per_tick)
        end_sample = int(end_tick * samples_per_tick) if end_tick is not None else None

        # Perform export
        return self.audio_exporter.export_to_wav(filepath, start_sample, end_sample)

    def export_audio_async(self, filepath: str, start_tick: int = 0,
                           end_tick: Optional[int] = None) -> bool:
        """
        Start async audio export to a WAV file (non-blocking).

        Args:
            filepath: Output file path
            start_tick: Start position in ticks (default: 0)
            end_tick: End position in ticks (default: auto-detect)

        Returns:
            True if export started successfully
        """
        # Prepare export with current project data
        if not self.prepare_export():
            return False

        # Convert ticks to samples
        sample_rate = self.config.sample_rate
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm
        samples_per_tick = (sample_rate * 60.0) / (bpm * ppqn)

        start_sample = int(start_tick * samples_per_tick)
        end_sample = int(end_tick * samples_per_tick) if end_tick is not None else None

        # Start async export
        self.audio_exporter.export_to_wav_async(filepath, start_sample, end_sample)
        return True

    def cancel_export(self) -> None:
        """Cancel the current export operation."""
        self.audio_exporter.cancel()

    def get_export_state(self) -> str:
        """Get the current export state.

        Returns:
            Current state as string: 'idle', 'exporting', 'completed', 'cancelled', 'error'
        """
        return self.audio_exporter.get_state().value

    def get_export_progress(self) -> dict:
        """Get the current export progress.

        Returns:
            Dictionary with progress info
        """
        progress = self.audio_exporter.get_progress()
        return {
            'state': progress.state.value,
            'progress': progress.progress,
            'current_sample': progress.current_sample,
            'total_samples': progress.total_samples,
            'elapsed_seconds': progress.elapsed_seconds,
            'error_message': progress.error_message
        }

    def get_export_info(self) -> dict:
        """Get information about the current export setup.

        Returns:
            Dictionary with export info
        """
        return self.audio_exporter.get_export_info()


class WebSocketServer:
    """WebSocket server for frontend communication."""

    def __init__(self, engine: BeatBoxDawEngine, host: str = "localhost", port: int = 8765):
        self.engine = engine
        self.host = host
        self.port = port
        self.server = None

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a connected client."""
        self.engine.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.engine.clients)}")

        try:
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'status',
                'data': self.engine.get_status()
            }))

            # Send project data
            project = self.engine.get_project()
            if project:
                await websocket.send(json.dumps({
                    'type': 'project',
                    'data': project
                }))

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handle_message(data)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'data': {'message': 'Invalid JSON'}
                    }))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.engine.clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.engine.clients)}")

    async def handle_message(self, data: dict) -> Optional[dict]:
        """Handle incoming WebSocket message."""
        msg_type = data.get('type')
        payload = data.get('data', {})

        # === BeatBox Detection Commands ===
        if msg_type == 'start':
            success = self.engine.start()
            return {'type': 'start_response', 'data': {'success': success}}

        elif msg_type == 'stop':
            self.engine.stop()
            return {'type': 'stop_response', 'data': {'success': True}}

        elif msg_type == 'pause':
            self.engine.pause()
            return {'type': 'pause_response', 'data': {'success': True}}

        elif msg_type == 'resume':
            self.engine.resume()
            return {'type': 'resume_response', 'data': {'success': True}}

        elif msg_type == 'status':
            return {'type': 'status', 'data': self.engine.get_status()}

        elif msg_type == 'start_recording':
            self.engine.start_recording()
            return {'type': 'recording_started', 'data': {'success': True}}

        elif msg_type == 'stop_recording':
            events = self.engine.stop_recording()
            return {'type': 'recording_stopped', 'data': {'events': events}}

        elif msg_type == 'export_midi':
            filename = payload.get('filename', 'output.mid')
            bpm = payload.get('bpm', 120)
            success = self.engine.export_midi(filename, bpm)
            return {'type': 'export_response', 'data': {'success': success, 'filename': filename}}

        elif msg_type == 'list_devices':
            input_devices = AudioCapture.list_devices()
            output_devices = AudioCapture.list_output_devices()
            return {'type': 'devices', 'data': {
                'devices': input_devices,
                'input_devices': input_devices,
                'output_devices': output_devices
            }}

        elif msg_type == 'set_audio_device':
            device_id = payload.get('device_id')
            device_type = payload.get('type', 'input')
            if device_id is not None:
                if device_type == 'input':
                    success = self.engine.audio_capture.set_device(device_id)
                elif device_type == 'output':
                    success = self.engine.set_audio_output_device(device_id)
                else:
                    return {'type': 'set_audio_device_response', 'data': {
                        'success': False,
                        'error': f'Invalid device type: {device_type}'
                    }}
                return {'type': 'set_audio_device_response', 'data': {
                    'success': success,
                    'device_id': device_id,
                    'type': device_type
                }}
            return {'type': 'set_audio_device_response', 'data': {
                'success': False,
                'error': 'device_id is required'
            }}

        # === DAW Transport Commands ===
        elif msg_type == 'transport_play':
            success = self.engine.transport_play()
            return {'type': 'transport_play_response', 'data': {'success': success}}

        elif msg_type == 'transport_pause':
            success = self.engine.transport_pause()
            return {'type': 'transport_pause_response', 'data': {'success': success}}

        elif msg_type == 'transport_stop':
            success = self.engine.transport_stop()
            return {'type': 'transport_stop_response', 'data': {'success': success}}

        elif msg_type == 'transport_record':
            recording = self.engine.transport_record()
            return {'type': 'transport_record_response', 'data': {'recording': recording}}

        elif msg_type == 'transport_seek':
            tick = payload.get('tick', 0)
            success = self.engine.transport_seek(tick)
            return {'type': 'transport_seek_response', 'data': {'success': success, 'tick': tick}}

        elif msg_type == 'set_bpm':
            bpm = payload.get('bpm', 120)
            self.engine.transport.bpm = bpm
            if self.engine._project:
                self.engine._project.bpm = bpm
            return {'type': 'set_bpm_response', 'data': {'bpm': bpm}}

        elif msg_type == 'set_loop':
            enabled = payload.get('enabled', False)
            start = payload.get('start_tick')
            end = payload.get('end_tick')
            self.engine.transport.set_loop(enabled, start, end)
            return {'type': 'set_loop_response', 'data': {'success': True}}

        elif msg_type == 'set_click':
            enabled = payload.get('enabled', True)
            self.engine.transport.config.click_enabled = enabled
            return {'type': 'set_click_response', 'data': {'enabled': enabled}}

        # === Project Commands ===
        elif msg_type == 'new_project':
            name = payload.get('name', 'New Project')
            project = self.engine.new_project(name)
            return {'type': 'project', 'data': project}

        elif msg_type == 'get_project':
            project = self.engine.get_project()
            return {'type': 'project', 'data': project}

        elif msg_type == 'add_track':
            track_type = payload.get('track_type', 'drum')
            name = payload.get('name')
            track = self.engine.add_track(track_type, name)
            return {'type': 'track_added', 'data': track}

        elif msg_type == 'add_beatbox_clip':
            track_id = payload.get('track_id')
            start_tick = payload.get('start_tick', 0)
            clip = self.engine.add_beatbox_clip(track_id, start_tick)
            return {'type': 'clip_added', 'data': clip}

        elif msg_type == 'set_track_armed':
            track_id = payload.get('track_id')
            armed = payload.get('armed', False)
            if not track_id:
                return {'type': 'set_track_armed_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.set_track_armed(track_id, armed)
            return {'type': 'set_track_armed_response', 'data': {
                'success': success,
                'track_id': track_id,
                'armed': armed
            }}

        # === Audio Track Recording Commands ===
        elif msg_type == 'start_track_recording':
            track_id = payload.get('track_id')
            if not track_id:
                return {'type': 'start_track_recording_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.start_track_recording(track_id)
            return {'type': 'start_track_recording_response', 'data': {
                'success': success,
                'track_id': track_id
            }}

        elif msg_type == 'stop_track_recording':
            result = self.engine.stop_track_recording()
            if result:
                return {'type': 'stop_track_recording_response', 'data': {
                    'success': True,
                    **result
                }}
            return {'type': 'stop_track_recording_response', 'data': {
                'success': False,
                'error': 'No active recording'
            }}

        elif msg_type == 'get_track_recording_info':
            info = self.engine.get_track_recording_info()
            return {'type': 'track_recording_info', 'data': info}

        # === Audio Playback Commands ===
        elif msg_type == 'load_audio_clip':
            filepath = payload.get('filepath')
            clip_id = payload.get('clip_id')
            track_id = payload.get('track_id')
            start_tick = payload.get('start_tick', 0)
            duration_ticks = payload.get('duration_ticks', 0)
            volume = payload.get('volume', 1.0)
            pan = payload.get('pan', 0.0)

            if not all([filepath, clip_id, track_id]):
                return {'type': 'load_audio_clip_response', 'data': {
                    'success': False,
                    'error': 'filepath, clip_id, and track_id are required'
                }}

            success = self.engine.load_audio_clip(
                filepath=filepath,
                clip_id=clip_id,
                track_id=track_id,
                start_tick=start_tick,
                duration_ticks=duration_ticks,
                volume=volume,
                pan=pan
            )
            return {'type': 'load_audio_clip_response', 'data': {
                'success': success,
                'clip_id': clip_id
            }}

        elif msg_type == 'unload_audio_clip':
            clip_id = payload.get('clip_id')
            if not clip_id:
                return {'type': 'unload_audio_clip_response', 'data': {
                    'success': False,
                    'error': 'clip_id is required'
                }}
            success = self.engine.unload_audio_clip(clip_id)
            return {'type': 'unload_audio_clip_response', 'data': {
                'success': success,
                'clip_id': clip_id
            }}

        elif msg_type == 'update_track_settings':
            track_id = payload.get('track_id')
            if not track_id:
                return {'type': 'update_track_settings_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.update_track_settings(
                track_id=track_id,
                volume=payload.get('volume'),
                pan=payload.get('pan'),
                muted=payload.get('muted')
            )
            return {'type': 'update_track_settings_response', 'data': {
                'success': success,
                'track_id': track_id
            }}

        elif msg_type == 'set_master_volume':
            volume = payload.get('volume', 1.0)
            success = self.engine.set_master_volume(volume)
            return {'type': 'set_master_volume_response', 'data': {
                'success': success,
                'volume': self.engine.audio_playback.master_volume
            }}

        elif msg_type == 'get_audio_playback_status':
            status = self.engine.get_audio_playback_status()
            return {'type': 'audio_playback_status', 'data': status}

        elif msg_type == 'set_audio_output_device':
            device_id = payload.get('device_id')
            if device_id is None:
                return {'type': 'set_audio_output_device_response', 'data': {
                    'success': False,
                    'error': 'device_id is required'
                }}
            success = self.engine.set_audio_output_device(device_id)
            return {'type': 'set_audio_output_device_response', 'data': {
                'success': success,
                'device_id': device_id
            }}

        elif msg_type == 'list_output_devices':
            devices = AudioPlayback.list_output_devices()
            default = AudioPlayback.get_default_output_device()
            return {'type': 'output_devices', 'data': {
                'devices': devices,
                'default': default
            }}

        # === Track Effects Commands ===
        elif msg_type == 'add_track_effect':
            track_id = payload.get('track_id')
            effect_type = payload.get('effect_type')
            if not track_id or not effect_type:
                return {'type': 'add_track_effect_response', 'data': {
                    'success': False,
                    'error': 'track_id and effect_type are required'
                }}
            effect_id = self.engine.add_track_effect(
                track_id=track_id,
                effect_type=effect_type,
                effect_id=payload.get('effect_id'),
                position=payload.get('position')
            )
            return {'type': 'add_track_effect_response', 'data': {
                'success': effect_id is not None,
                'effect_id': effect_id,
                'track_id': track_id
            }}

        elif msg_type == 'remove_track_effect':
            track_id = payload.get('track_id')
            effect_id = payload.get('effect_id')
            if not track_id or not effect_id:
                return {'type': 'remove_track_effect_response', 'data': {
                    'success': False,
                    'error': 'track_id and effect_id are required'
                }}
            success = self.engine.remove_track_effect(track_id, effect_id)
            return {'type': 'remove_track_effect_response', 'data': {
                'success': success,
                'track_id': track_id,
                'effect_id': effect_id
            }}

        elif msg_type == 'move_track_effect':
            track_id = payload.get('track_id')
            effect_id = payload.get('effect_id')
            new_position = payload.get('position')
            if not track_id or not effect_id or new_position is None:
                return {'type': 'move_track_effect_response', 'data': {
                    'success': False,
                    'error': 'track_id, effect_id, and position are required'
                }}
            success = self.engine.move_track_effect(track_id, effect_id, new_position)
            return {'type': 'move_track_effect_response', 'data': {
                'success': success,
                'track_id': track_id,
                'effect_id': effect_id
            }}

        elif msg_type == 'set_track_effect_parameter':
            track_id = payload.get('track_id')
            effect_id = payload.get('effect_id')
            param_name = payload.get('param_name')
            value = payload.get('value')
            if not all([track_id, effect_id, param_name, value is not None]):
                return {'type': 'set_track_effect_parameter_response', 'data': {
                    'success': False,
                    'error': 'track_id, effect_id, param_name, and value are required'
                }}
            success = self.engine.set_track_effect_parameter(
                track_id, effect_id, param_name, value
            )
            return {'type': 'set_track_effect_parameter_response', 'data': {
                'success': success,
                'track_id': track_id,
                'effect_id': effect_id
            }}

        elif msg_type == 'get_track_effect_parameters':
            track_id = payload.get('track_id')
            effect_id = payload.get('effect_id')
            if not track_id or not effect_id:
                return {'type': 'get_track_effect_parameters_response', 'data': {
                    'success': False,
                    'error': 'track_id and effect_id are required'
                }}
            params = self.engine.get_track_effect_parameters(track_id, effect_id)
            return {'type': 'get_track_effect_parameters_response', 'data': {
                'success': params is not None,
                'parameters': params,
                'track_id': track_id,
                'effect_id': effect_id
            }}

        elif msg_type == 'get_track_effects_chain':
            track_id = payload.get('track_id')
            if not track_id:
                return {'type': 'get_track_effects_chain_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            chain = self.engine.get_track_effects_chain(track_id)
            return {'type': 'get_track_effects_chain_response', 'data': {
                'success': True,
                'chain': chain,
                'track_id': track_id
            }}

        elif msg_type == 'bypass_track_effects':
            track_id = payload.get('track_id')
            bypassed = payload.get('bypassed', True)
            if not track_id:
                return {'type': 'bypass_track_effects_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.bypass_track_effects(track_id, bypassed)
            return {'type': 'bypass_track_effects_response', 'data': {
                'success': success,
                'track_id': track_id,
                'bypassed': bypassed
            }}

        elif msg_type == 'reset_track_effects':
            track_id = payload.get('track_id')
            if not track_id:
                return {'type': 'reset_track_effects_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.reset_track_effects(track_id)
            return {'type': 'reset_track_effects_response', 'data': {
                'success': success,
                'track_id': track_id
            }}

        elif msg_type == 'clear_track_effects':
            track_id = payload.get('track_id')
            if not track_id:
                return {'type': 'clear_track_effects_response', 'data': {
                    'success': False,
                    'error': 'track_id is required'
                }}
            success = self.engine.clear_track_effects(track_id)
            return {'type': 'clear_track_effects_response', 'data': {
                'success': success,
                'track_id': track_id
            }}

        # === Master Effects Commands ===
        elif msg_type == 'add_master_effect':
            effect_type = payload.get('effect_type')
            if not effect_type:
                return {'type': 'add_master_effect_response', 'data': {
                    'success': False,
                    'error': 'effect_type is required'
                }}
            effect_id = self.engine.add_master_effect(
                effect_type=effect_type,
                effect_id=payload.get('effect_id'),
                position=payload.get('position')
            )
            return {'type': 'add_master_effect_response', 'data': {
                'success': effect_id is not None,
                'effect_id': effect_id
            }}

        elif msg_type == 'remove_master_effect':
            effect_id = payload.get('effect_id')
            if not effect_id:
                return {'type': 'remove_master_effect_response', 'data': {
                    'success': False,
                    'error': 'effect_id is required'
                }}
            success = self.engine.remove_master_effect(effect_id)
            return {'type': 'remove_master_effect_response', 'data': {
                'success': success,
                'effect_id': effect_id
            }}

        elif msg_type == 'move_master_effect':
            effect_id = payload.get('effect_id')
            new_position = payload.get('position')
            if not effect_id or new_position is None:
                return {'type': 'move_master_effect_response', 'data': {
                    'success': False,
                    'error': 'effect_id and position are required'
                }}
            success = self.engine.move_master_effect(effect_id, new_position)
            return {'type': 'move_master_effect_response', 'data': {
                'success': success,
                'effect_id': effect_id
            }}

        elif msg_type == 'set_master_effect_parameter':
            effect_id = payload.get('effect_id')
            param_name = payload.get('param_name')
            value = payload.get('value')
            if not all([effect_id, param_name, value is not None]):
                return {'type': 'set_master_effect_parameter_response', 'data': {
                    'success': False,
                    'error': 'effect_id, param_name, and value are required'
                }}
            success = self.engine.set_master_effect_parameter(effect_id, param_name, value)
            return {'type': 'set_master_effect_parameter_response', 'data': {
                'success': success,
                'effect_id': effect_id
            }}

        elif msg_type == 'get_master_effect_parameters':
            effect_id = payload.get('effect_id')
            if not effect_id:
                return {'type': 'get_master_effect_parameters_response', 'data': {
                    'success': False,
                    'error': 'effect_id is required'
                }}
            params = self.engine.get_master_effect_parameters(effect_id)
            return {'type': 'get_master_effect_parameters_response', 'data': {
                'success': params is not None,
                'parameters': params,
                'effect_id': effect_id
            }}

        elif msg_type == 'get_master_effects_chain':
            chain = self.engine.get_master_effects_chain()
            return {'type': 'get_master_effects_chain_response', 'data': {
                'success': True,
                'chain': chain
            }}

        elif msg_type == 'bypass_master_effects':
            bypassed = payload.get('bypassed', True)
            self.engine.bypass_master_effects(bypassed)
            return {'type': 'bypass_master_effects_response', 'data': {
                'success': True,
                'bypassed': bypassed
            }}

        elif msg_type == 'reset_master_effects':
            self.engine.reset_master_effects()
            return {'type': 'reset_master_effects_response', 'data': {
                'success': True
            }}

        elif msg_type == 'clear_master_effects':
            self.engine.clear_master_effects()
            return {'type': 'clear_master_effects_response', 'data': {
                'success': True
            }}

        elif msg_type == 'get_available_effect_types':
            effect_types = self.engine.get_available_effect_types()
            return {'type': 'available_effect_types', 'data': {
                'effect_types': effect_types
            }}

        # === Unified Effect Parameter Update Command ===
        elif msg_type == 'update_effect_parameter':
            effect_id = payload.get('effect_id')
            param_name = payload.get('param_name')
            value = payload.get('value')
            track_id = payload.get('track_id')  # Optional: if provided, update track effect

            if not all([effect_id, param_name, value is not None]):
                return {'type': 'update_effect_parameter_response', 'data': {
                    'success': False,
                    'error': 'effect_id, param_name, and value are required'
                }}

            # If track_id is provided, update track effect; otherwise update master effect
            if track_id:
                success = self.engine.set_track_effect_parameter(
                    track_id, effect_id, param_name, value
                )
                return {'type': 'update_effect_parameter_response', 'data': {
                    'success': success,
                    'effect_id': effect_id,
                    'param_name': param_name,
                    'value': value,
                    'track_id': track_id
                }}
            else:
                success = self.engine.set_master_effect_parameter(effect_id, param_name, value)
                return {'type': 'update_effect_parameter_response', 'data': {
                    'success': success,
                    'effect_id': effect_id,
                    'param_name': param_name,
                    'value': value
                }}

        # === Audio Export Commands ===
        elif msg_type == 'export_audio':
            filepath = payload.get('filepath')
            if not filepath:
                return {'type': 'export_audio_response', 'data': {
                    'success': False,
                    'error': 'filepath is required'
                }}

            start_tick = payload.get('start_tick', 0)
            end_tick = payload.get('end_tick')  # None for auto-detect

            # Use async export for non-blocking operation with progress updates
            success = self.engine.export_audio_async(filepath, start_tick, end_tick)
            return {'type': 'export_audio_response', 'data': {
                'success': success,
                'filepath': filepath,
                'start_tick': start_tick,
                'end_tick': end_tick
            }}

        elif msg_type == 'cancel_export':
            self.engine.cancel_export()
            return {'type': 'cancel_export_response', 'data': {
                'success': True
            }}

        elif msg_type == 'get_export_state':
            state = self.engine.get_export_state()
            return {'type': 'export_state', 'data': {
                'state': state
            }}

        elif msg_type == 'get_export_progress':
            progress = self.engine.get_export_progress()
            return {'type': 'export_progress', 'data': progress}

        elif msg_type == 'get_export_info':
            info = self.engine.get_export_info()
            return {'type': 'export_info', 'data': info}

        elif msg_type == 'ping':
            return {'type': 'pong', 'data': {}}

        return None

    async def start(self) -> None:
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    def stop(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()


async def main():
    """Main entry point."""
    print("=" * 50)
    print("BeatBox DAW Engine")
    print("=" * 50)

    # Create engine
    config = EngineConfig()
    engine = BeatBoxDawEngine(config)

    # Store the event loop reference for thread-safe callbacks
    engine._event_loop = asyncio.get_running_loop()

    # Create WebSocket server
    server = WebSocketServer(engine, config.websocket_host, config.websocket_port)

    # Handle shutdown
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        engine.stop()
        engine.transport.stop()
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())
