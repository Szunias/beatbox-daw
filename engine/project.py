"""
Project Management Module
Handles DAW project state, tracks, clips, and persistence.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID."""
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


@dataclass
class MidiNote:
    """A single MIDI note."""
    id: str
    pitch: int  # 0-127
    velocity: int  # 0-127
    start_tick: int
    duration: int

    @classmethod
    def create(cls, pitch: int, velocity: int, start_tick: int, duration: int) -> 'MidiNote':
        return cls(
            id=generate_id("note"),
            pitch=pitch,
            velocity=velocity,
            start_tick=start_tick,
            duration=duration
        )


@dataclass
class MidiClipData:
    """MIDI clip data containing notes."""
    notes: List[MidiNote] = field(default_factory=list)
    duration: int = 1920  # Default 1 bar at 480 PPQN


@dataclass
class Clip:
    """A clip on a track timeline."""
    id: str
    name: str
    type: str  # 'midi', 'audio', 'beatbox'
    start_tick: int
    duration: int
    color: str
    muted: bool = False
    data: Optional[MidiClipData] = None

    @classmethod
    def create_midi(cls, name: str, start_tick: int, duration: int,
                   notes: List[MidiNote] = None, color: str = "#4ade80") -> 'Clip':
        return cls(
            id=generate_id("clip"),
            name=name,
            type="midi",
            start_tick=start_tick,
            duration=duration,
            color=color,
            data=MidiClipData(notes=notes or [], duration=duration)
        )


@dataclass
class Track:
    """A track in the project."""
    id: str
    name: str
    type: str  # 'midi', 'audio', 'drum', 'master'
    color: str
    volume: float = 0.8
    pan: float = 0.0
    muted: bool = False
    solo: bool = False
    armed: bool = False
    clips: List[Clip] = field(default_factory=list)
    instrument_id: Optional[str] = None
    midi_channel: int = 1

    @classmethod
    def create_drum(cls, name: str = "Drums", color: str = "#e94560") -> 'Track':
        return cls(
            id=generate_id("track"),
            name=name,
            type="drum",
            color=color,
            midi_channel=10  # GM drum channel
        )

    @classmethod
    def create_midi(cls, name: str, color: str = "#4ade80", channel: int = 1) -> 'Track':
        return cls(
            id=generate_id("track"),
            name=name,
            type="midi",
            color=color,
            midi_channel=channel
        )

    @classmethod
    def create_master(cls) -> 'Track':
        return cls(
            id=generate_id("master"),
            name="Master",
            type="master",
            color="#fbbf24",
            volume=1.0
        )


@dataclass
class Project:
    """A DAW project containing all tracks and settings."""
    id: str
    name: str
    created_at: float
    modified_at: float
    bpm: float = 120.0
    time_sig_numerator: int = 4
    time_sig_denominator: int = 4
    tracks: List[Track] = field(default_factory=list)
    master_track: Track = field(default_factory=Track.create_master)

    @classmethod
    def create(cls, name: str = "Untitled Project") -> 'Project':
        return cls(
            id=generate_id("project"),
            name=name,
            created_at=time.time(),
            modified_at=time.time(),
            tracks=[Track.create_drum("BeatBox Drums")],
            master_track=Track.create_master()
        )

    def to_dict(self) -> dict:
        """Convert project to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'bpm': self.bpm,
            'time_sig_numerator': self.time_sig_numerator,
            'time_sig_denominator': self.time_sig_denominator,
            'tracks': [self._track_to_dict(t) for t in self.tracks],
            'master_track': self._track_to_dict(self.master_track)
        }

    def _track_to_dict(self, track: Track) -> dict:
        """Convert track to dictionary."""
        return {
            'id': track.id,
            'name': track.name,
            'type': track.type,
            'color': track.color,
            'volume': track.volume,
            'pan': track.pan,
            'muted': track.muted,
            'solo': track.solo,
            'armed': track.armed,
            'clips': [self._clip_to_dict(c) for c in track.clips],
            'instrument_id': track.instrument_id,
            'midi_channel': track.midi_channel
        }

    def _clip_to_dict(self, clip: Clip) -> dict:
        """Convert clip to dictionary."""
        result = {
            'id': clip.id,
            'name': clip.name,
            'type': clip.type,
            'start_tick': clip.start_tick,
            'duration': clip.duration,
            'color': clip.color,
            'muted': clip.muted
        }
        if clip.data:
            result['data'] = {
                'notes': [asdict(n) for n in clip.data.notes],
                'duration': clip.data.duration
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'Project':
        """Create project from dictionary."""
        project = cls(
            id=data['id'],
            name=data['name'],
            created_at=data['created_at'],
            modified_at=data['modified_at'],
            bpm=data.get('bpm', 120.0),
            time_sig_numerator=data.get('time_sig_numerator', 4),
            time_sig_denominator=data.get('time_sig_denominator', 4)
        )

        # Load tracks
        for track_data in data.get('tracks', []):
            project.tracks.append(cls._track_from_dict(track_data))

        # Load master track
        if 'master_track' in data:
            project.master_track = cls._track_from_dict(data['master_track'])

        return project

    @classmethod
    def _track_from_dict(cls, data: dict) -> Track:
        """Create track from dictionary."""
        track = Track(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            color=data['color'],
            volume=data.get('volume', 0.8),
            pan=data.get('pan', 0.0),
            muted=data.get('muted', False),
            solo=data.get('solo', False),
            armed=data.get('armed', False),
            instrument_id=data.get('instrument_id'),
            midi_channel=data.get('midi_channel', 1)
        )

        # Load clips
        for clip_data in data.get('clips', []):
            track.clips.append(cls._clip_from_dict(clip_data))

        return track

    @classmethod
    def _clip_from_dict(cls, data: dict) -> Clip:
        """Create clip from dictionary."""
        clip = Clip(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            start_tick=data['start_tick'],
            duration=data['duration'],
            color=data['color'],
            muted=data.get('muted', False)
        )

        # Load MIDI data
        if 'data' in data and data['data']:
            notes = [MidiNote(**n) for n in data['data'].get('notes', [])]
            clip.data = MidiClipData(
                notes=notes,
                duration=data['data'].get('duration', clip.duration)
            )

        return clip


class ProjectManager:
    """Manages project state and persistence."""

    def __init__(self, projects_dir: str = None):
        self.projects_dir = Path(projects_dir) if projects_dir else Path.home() / "BeatBoxDAW" / "Projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.current_project: Optional[Project] = None

    def new_project(self, name: str = "Untitled Project") -> Project:
        """Create a new project."""
        self.current_project = Project.create(name)
        return self.current_project

    def save_project(self, filepath: str = None) -> str:
        """Save current project to file."""
        if not self.current_project:
            raise ValueError("No project to save")

        if filepath is None:
            filepath = self.projects_dir / f"{self.current_project.name}.bbdaw"

        filepath = Path(filepath)
        self.current_project.modified_at = time.time()

        with open(filepath, 'w') as f:
            json.dump(self.current_project.to_dict(), f, indent=2)

        print(f"Project saved to {filepath}")
        return str(filepath)

    def load_project(self, filepath: str) -> Project:
        """Load project from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.current_project = Project.from_dict(data)
        print(f"Project loaded: {self.current_project.name}")
        return self.current_project

    def add_track(self, track_type: str = "midi", name: str = None) -> Track:
        """Add a track to the current project."""
        if not self.current_project:
            raise ValueError("No project loaded")

        if track_type == "drum":
            track = Track.create_drum(name or f"Drums {len(self.current_project.tracks) + 1}")
        else:
            track = Track.create_midi(name or f"Track {len(self.current_project.tracks) + 1}")

        self.current_project.tracks.append(track)
        return track

    def remove_track(self, track_id: str) -> bool:
        """Remove a track from the current project."""
        if not self.current_project:
            return False

        for i, track in enumerate(self.current_project.tracks):
            if track.id == track_id:
                self.current_project.tracks.pop(i)
                return True
        return False

    def add_clip(self, track_id: str, clip: Clip) -> bool:
        """Add a clip to a track."""
        if not self.current_project:
            return False

        for track in self.current_project.tracks:
            if track.id == track_id:
                track.clips.append(clip)
                return True
        return False

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by ID."""
        if not self.current_project:
            return None

        for track in self.current_project.tracks:
            if track.id == track_id:
                return track
        return None

    def list_projects(self) -> List[str]:
        """List all saved projects."""
        return [f.stem for f in self.projects_dir.glob("*.bbdaw")]


# Standalone test
if __name__ == "__main__":
    manager = ProjectManager()

    # Create new project
    project = manager.new_project("Test Project")
    print(f"Created project: {project.name}")

    # Add some tracks
    track1 = manager.add_track("drum", "Beat Track")
    track2 = manager.add_track("midi", "Melody")

    # Add a clip with notes
    clip = Clip.create_midi("Pattern 1", start_tick=0, duration=1920)
    clip.data.notes = [
        MidiNote.create(36, 100, 0, 240),     # Kick
        MidiNote.create(38, 100, 480, 240),   # Snare
        MidiNote.create(42, 80, 240, 120),    # HiHat
        MidiNote.create(42, 80, 720, 120),    # HiHat
    ]
    manager.add_clip(track1.id, clip)

    # Save project
    filepath = manager.save_project()

    # Load it back
    loaded = manager.load_project(filepath)
    print(f"Loaded project: {loaded.name}")
    print(f"Tracks: {[t.name for t in loaded.tracks]}")

    # Print clip info
    for track in loaded.tracks:
        for clip in track.clips:
            print(f"  Clip: {clip.name} ({len(clip.data.notes) if clip.data else 0} notes)")
