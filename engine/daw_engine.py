"""
DAW Engine Module
Main DAW engine integrating transport, project, scheduler, and audio.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Set, Dict, Any
from pathlib import Path

from transport import Transport, TransportState, TransportConfig
from project import Project, ProjectManager, Track, Clip, MidiNote, MidiClipData
from scheduler import MidiScheduler, ScheduledEvent
from midi_output import MidiOutput


@dataclass
class DawEngineConfig:
    """DAW engine configuration."""
    sample_rate: int = 44100
    buffer_size: int = 512
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    projects_dir: str = None


class DawEngine:
    """
    Main DAW engine coordinating all components.
    Provides high-level API for the frontend.
    """

    def __init__(self, config: Optional[DawEngineConfig] = None):
        self.config = config or DawEngineConfig()

        # Initialize components
        self.transport = Transport()
        self.project_manager = ProjectManager(self.config.projects_dir)
        self.scheduler = MidiScheduler(self.transport)
        self.midi_output = MidiOutput()

        # Current project
        self._project: Optional[Project] = None

        # WebSocket clients for broadcasting
        self.clients: Set = set()

        # Audio levels per track
        self._track_levels: Dict[str, float] = {}
        self._master_level: float = 0.0

        # Setup MIDI callbacks
        self.scheduler.set_note_on_callback(self._on_note_on)
        self.scheduler.set_note_off_callback(self._on_note_off)
        self.scheduler.set_click_callback(self._on_click)

        # Setup transport callbacks
        self.transport.add_state_callback(self._on_transport_state_change)

        # Create default project
        self.new_project()

    @property
    def project(self) -> Optional[Project]:
        return self._project

    def new_project(self, name: str = "New Project") -> Project:
        """Create a new project."""
        self._project = self.project_manager.new_project(name)
        self.scheduler.load_project(self._project)
        return self._project

    def load_project(self, filepath: str) -> Project:
        """Load a project from file."""
        self._project = self.project_manager.load_project(filepath)
        self.scheduler.load_project(self._project)
        self.transport.bpm = self._project.bpm
        return self._project

    def save_project(self, filepath: str = None) -> str:
        """Save current project."""
        return self.project_manager.save_project(filepath)

    # === Transport Controls ===

    def play(self) -> bool:
        """Start playback."""
        if self._project:
            self.scheduler.load_project(self._project)
        return self.transport.play()

    def pause(self):
        """Pause playback."""
        self.transport.pause()

    def stop(self):
        """Stop playback."""
        self.transport.stop()

    def record(self) -> bool:
        """Toggle recording."""
        return self.transport.record()

    def seek(self, tick: int):
        """Seek to position."""
        self.transport.seek(tick)

    def set_bpm(self, bpm: float):
        """Set BPM."""
        self.transport.bpm = bpm
        if self._project:
            self._project.bpm = bpm

    def set_loop(self, enabled: bool, start_tick: int = None, end_tick: int = None):
        """Configure loop."""
        self.transport.set_loop(enabled, start_tick, end_tick)

    def set_click(self, enabled: bool):
        """Enable/disable metronome."""
        self.transport.config.click_enabled = enabled

    # === Track Management ===

    def add_track(self, track_type: str = "midi", name: str = None) -> Track:
        """Add a track to the project."""
        track = self.project_manager.add_track(track_type, name)
        return track

    def remove_track(self, track_id: str) -> bool:
        """Remove a track."""
        return self.project_manager.remove_track(track_id)

    def set_track_volume(self, track_id: str, volume: float):
        """Set track volume."""
        track = self.project_manager.get_track(track_id)
        if track:
            track.volume = max(0.0, min(1.5, volume))

    def set_track_pan(self, track_id: str, pan: float):
        """Set track pan."""
        track = self.project_manager.get_track(track_id)
        if track:
            track.pan = max(-1.0, min(1.0, pan))

    def set_track_mute(self, track_id: str, muted: bool):
        """Set track mute state."""
        track = self.project_manager.get_track(track_id)
        if track:
            track.muted = muted
            # Reload scheduler to apply mute
            if self._project:
                self.scheduler.load_track(track)

    def set_track_solo(self, track_id: str, solo: bool):
        """Set track solo state."""
        track = self.project_manager.get_track(track_id)
        if track:
            track.solo = solo

    def set_track_armed(self, track_id: str, armed: bool):
        """Set track record arm state."""
        track = self.project_manager.get_track(track_id)
        if track:
            track.armed = armed

    # === Clip Management ===

    def add_clip(self, track_id: str, name: str, start_tick: int, duration: int,
                 notes: list = None) -> Optional[Clip]:
        """Add a MIDI clip to a track."""
        track = self.project_manager.get_track(track_id)
        if not track:
            return None

        clip = Clip.create_midi(name, start_tick, duration, color=track.color)
        if notes:
            clip.data.notes = [MidiNote(**n) if isinstance(n, dict) else n for n in notes]

        track.clips.append(clip)
        self.scheduler.load_track(track)
        return clip

    def remove_clip(self, track_id: str, clip_id: str) -> bool:
        """Remove a clip from a track."""
        track = self.project_manager.get_track(track_id)
        if not track:
            return False

        for i, clip in enumerate(track.clips):
            if clip.id == clip_id:
                track.clips.pop(i)
                self.scheduler.load_track(track)
                return True
        return False

    def add_beatbox_clip(self, track_id: str, events: list, start_tick: int) -> Optional[Clip]:
        """Add a clip from beatbox events."""
        track = self.project_manager.get_track(track_id)
        if not track:
            return None

        # Convert beatbox events to MIDI notes
        notes = []
        ppqn = self.transport.ticks_per_beat
        bpm = self.transport.bpm

        for event in events:
            # Convert timestamp to ticks
            tick = int(event.get('timestamp', 0) * (bpm / 60.0) * ppqn)
            notes.append(MidiNote.create(
                pitch=event.get('midi_note', 36),
                velocity=event.get('velocity', 100),
                start_tick=tick,
                duration=ppqn // 4  # 16th note
            ))

        # Calculate clip duration
        if notes:
            max_tick = max(n.start_tick + n.duration for n in notes)
            duration = ((max_tick // (ppqn * 4)) + 1) * ppqn * 4  # Round up to bar
        else:
            duration = ppqn * 4  # 1 bar

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

        print(f"DawEngine: Added beatbox clip with {len(notes)} notes")
        return clip

    # === MIDI Callbacks ===

    def _on_note_on(self, channel: int, note: int, velocity: int, track_id: str):
        """Handle note on from scheduler."""
        # Get track for volume/pan
        track = self.project_manager.get_track(track_id)
        if track and not track.muted:
            adjusted_velocity = int(velocity * track.volume)
            self.midi_output.send_note(note, adjusted_velocity, channel=channel)

            # Update track level
            self._track_levels[track_id] = velocity / 127.0

    def _on_note_off(self, channel: int, note: int, track_id: str):
        """Handle note off from scheduler."""
        self.midi_output.send_note_off(note, channel=channel)

    def _on_click(self, bar: int, beat: int):
        """Handle metronome click."""
        # Play click sound (using MIDI for now)
        velocity = 100 if beat == 1 else 70
        note = 76 if beat == 1 else 77  # Woodblock sounds
        self.midi_output.send_note(note, velocity, channel=10)

        # Broadcast click event
        asyncio.create_task(self.broadcast({
            'type': 'click',
            'data': {'bar': bar, 'beat': beat}
        }))

    def _on_transport_state_change(self, state: TransportState):
        """Handle transport state change."""
        asyncio.create_task(self.broadcast({
            'type': 'transport_state',
            'data': {'state': state.value}
        }))

    # === Status ===

    def get_status(self) -> dict:
        """Get full engine status."""
        transport_status = self.transport.get_status()

        project_data = None
        if self._project:
            project_data = {
                'id': self._project.id,
                'name': self._project.name,
                'tracks': len(self._project.tracks),
            }

        return {
            'transport': transport_status,
            'project': project_data,
            'midi_connected': self.midi_output.is_connected,
            'track_levels': self._track_levels,
            'master_level': self._master_level,
        }

    def get_project_data(self) -> Optional[dict]:
        """Get full project data for frontend."""
        if not self._project:
            return None
        return self._project.to_dict()

    # === WebSocket Broadcasting ===

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        message_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_json) for client in self.clients],
            return_exceptions=True
        )

    def connect(self) -> bool:
        """Connect MIDI output."""
        return self.midi_output.connect()

    def disconnect(self):
        """Disconnect and cleanup."""
        self.transport.stop()
        self.midi_output.disconnect()


# Export for other modules
__all__ = ['DawEngine', 'DawEngineConfig']
