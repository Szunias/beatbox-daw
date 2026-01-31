"""
BeatBox DAW Engine

Main entry point for the Python audio processing engine.
Provides WebSocket server for communication with Tauri frontend.
Supports both BeatBox-to-MIDI conversion and full DAW functionality.
"""

import asyncio
import json
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

from audio_capture import AudioCapture, AudioConfig
from audio_recorder import AudioRecorder, RecordingConfig, RecordingMetadata
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
            TransportState.PLAYING: [TransportState.PAUSED, TransportState.STOPPED, TransportState.RECORDING],
            TransportState.PAUSED: [TransportState.PLAYING, TransportState.STOPPED],
            TransportState.RECORDING: [TransportState.STOPPED, TransportState.PAUSED],
            TransportState.PRE_ROLL: [TransportState.STOPPED, TransportState.RECORDING],
        }

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
        """Process incoming audio buffer for beatbox detection."""
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

            asyncio.create_task(self.broadcast({
                'type': 'transport_position',
                'data': {
                    'tick': tick,
                    'bar': bar,
                    'beat': beat,
                    'state': self.transport.state.value,
                    'bpm': self.transport.bpm,
                    'timestamp': current_time  # High-precision timestamp for sync
                }
            }))

    def _on_transport_state_change(self, state: TransportState):
        """Broadcast transport state change to all clients.

        Includes full position and timing data for accurate frontend synchronization.
        """
        current_tick = self.transport.current_tick
        bar, beat = self.transport._ticks_to_bar_beat(max(0, current_tick))

        asyncio.create_task(self.broadcast({
            'type': 'transport_state',
            'data': {
                'state': state.value,
                'tick': current_tick,
                'bar': bar,
                'beat': beat,
                'bpm': self.transport.bpm,
                'timestamp': time.perf_counter()
            }
        }))

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
        """Handle metronome click."""
        if self.transport.config.click_enabled:
            velocity = 100 if beat == 1 else 70
            note = 76 if beat == 1 else 77  # Woodblock sounds
            self.midi_output.send_note(note, velocity, duration=0.05, channel=9)

            asyncio.create_task(self.broadcast({
                'type': 'click',
                'data': {'bar': bar, 'beat': beat}
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
        Returns True if recording started, False otherwise.
        """
        current_state = self.transport.state
        # If already recording or in pre-roll, we're stopping - always valid
        if current_state in (TransportState.RECORDING, TransportState.PRE_ROLL):
            return self.transport.record()

        # Starting recording - validate transition
        if not self._validate_state_transition(TransportState.RECORDING) and \
           not self._validate_state_transition(TransportState.PRE_ROLL):
            return False

        return self.transport.record()

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

        # Ensure audio capture is running and feeding to recorder
        if not self.audio_capture.is_running:
            self.audio_capture.add_callback(self._track_recording_callback)
            self.audio_capture.start()
        else:
            # Add recording callback if not already added
            if self._track_recording_callback not in self.audio_capture.callbacks:
                self.audio_capture.add_callback(self._track_recording_callback)

        asyncio.create_task(self.broadcast({
            'type': 'track_recording_started',
            'data': {
                'track_id': track_id,
                'start_time': start_time
            }
        }))

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

        # Remove recording callback
        if self._track_recording_callback in self.audio_capture.callbacks:
            self.audio_capture.remove_callback(self._track_recording_callback)

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

        asyncio.create_task(self.broadcast({
            'type': 'track_recording_stopped',
            'data': result
        }))

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
            asyncio.create_task(self.broadcast({
                'type': 'track_recording_auto_stopped',
                'data': {
                    'track_id': self._recording_track_id,
                    'duration': metadata.duration,
                    'reason': 'max_duration_reached'
                }
            }))


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
            if device_type == 'input' and device_id is not None:
                success = self.engine.audio_capture.set_device(device_id)
                return {'type': 'set_audio_device_response', 'data': {
                    'success': success,
                    'device_id': device_id,
                    'type': device_type
                }}
            return {'type': 'set_audio_device_response', 'data': {
                'success': False,
                'error': 'Invalid device configuration'
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
