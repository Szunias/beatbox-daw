"""
MIDI and Audio Event Scheduler Module
Handles scheduling and playback of MIDI and audio clip events with precise timing.
"""

import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple
from queue import PriorityQueue
import heapq

from transport import Transport, TransportState
from project import Project, Track, Clip, MidiNote


@dataclass
class ScheduledEvent:
    """A MIDI event scheduled for playback."""
    tick: int
    event_type: str  # 'note_on', 'note_off', 'cc', 'program'
    channel: int
    note: int
    velocity: int
    track_id: str

    def __lt__(self, other):
        return self.tick < other.tick


@dataclass
class ScheduledAudioClipEvent:
    """An audio clip event scheduled for playback."""
    tick: int
    event_type: str  # 'clip_start', 'clip_stop'
    clip_id: str
    track_id: str
    file_path: Optional[str] = None  # Path to audio file (for clip_start)
    duration_ticks: int = 0  # Duration in ticks
    volume: float = 1.0
    pan: float = 0.0

    def __lt__(self, other):
        return self.tick < other.tick


class MidiScheduler:
    """
    Schedules and triggers MIDI and audio clip events during playback.
    Works with Transport to handle timing and playback position.
    """

    def __init__(self, transport: Transport):
        self.transport = transport
        self._events: List[ScheduledEvent] = []
        self._event_index = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # MIDI Callbacks
        self._note_on_callback: Optional[Callable[[int, int, int, str], None]] = None
        self._note_off_callback: Optional[Callable[[int, int, str], None]] = None
        self._click_callback: Optional[Callable[[int, int], None]] = None

        # Audio clip events and storage
        self._audio_events: List[ScheduledAudioClipEvent] = []
        self._audio_event_index = 0
        self._audio_clips: Dict[str, Dict] = {}  # clip_id -> clip info

        # Audio clip callbacks
        self._audio_clip_start_callback: Optional[Callable[[str, str, str, int, float, float], None]] = None
        self._audio_clip_stop_callback: Optional[Callable[[str, str], None]] = None

        # Lookahead buffer (events to play soon)
        self._lookahead_ms = 50  # 50ms lookahead
        self._last_processed_tick = -1

        # Register transport callbacks
        transport.add_state_callback(self._on_transport_state_change)
        transport.add_beat_callback(self._on_beat)

    def set_note_on_callback(self, callback: Callable[[int, int, int, str], None]):
        """Set callback for note on events (channel, note, velocity, track_id)."""
        self._note_on_callback = callback

    def set_note_off_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for note off events (channel, note, track_id)."""
        self._note_off_callback = callback

    def set_click_callback(self, callback: Callable[[int, int], None]):
        """Set callback for metronome clicks (bar, beat)."""
        self._click_callback = callback

    def set_audio_clip_start_callback(self, callback: Callable[[str, str, str, int, float, float], None]):
        """Set callback for audio clip start events (clip_id, track_id, file_path, duration_ticks, volume, pan)."""
        self._audio_clip_start_callback = callback

    def set_audio_clip_stop_callback(self, callback: Callable[[str, str], None]):
        """Set callback for audio clip stop events (clip_id, track_id)."""
        self._audio_clip_stop_callback = callback

    def load_project(self, project: Project):
        """Load all MIDI and audio events from project into scheduler."""
        self._events.clear()
        self._audio_events.clear()
        self._audio_clips.clear()

        midi_event_count = 0
        audio_clip_count = 0

        for track in project.tracks:
            if track.muted:
                continue

            for clip in track.clips:
                if clip.muted:
                    continue

                # Handle audio clips
                if clip.type == 'audio':
                    # Get file path from clip data or attributes
                    file_path = None
                    if hasattr(clip, 'file_path'):
                        file_path = clip.file_path
                    elif clip.data and hasattr(clip.data, 'file_path'):
                        file_path = clip.data.file_path

                    if file_path:
                        self.load_audio_clip(
                            clip_id=clip.id,
                            track_id=track.id,
                            file_path=file_path,
                            start_tick=clip.start_tick,
                            duration_ticks=clip.duration,
                            volume=track.volume,
                            pan=track.pan
                        )
                        audio_clip_count += 1
                    continue

                # Handle MIDI clips
                if not clip.data:
                    continue

                # Schedule all notes in clip
                for note in clip.data.notes:
                    # Note on
                    self._events.append(ScheduledEvent(
                        tick=clip.start_tick + note.start_tick,
                        event_type='note_on',
                        channel=track.midi_channel,
                        note=note.pitch,
                        velocity=note.velocity,
                        track_id=track.id
                    ))
                    midi_event_count += 1

                    # Note off
                    self._events.append(ScheduledEvent(
                        tick=clip.start_tick + note.start_tick + note.duration,
                        event_type='note_off',
                        channel=track.midi_channel,
                        note=note.pitch,
                        velocity=0,
                        track_id=track.id
                    ))
                    midi_event_count += 1

        # Sort by tick
        self._events.sort()
        self._event_index = 0
        self._last_processed_tick = -1

        # Audio events are already sorted by load_audio_clip
        self._reset_audio_index()

        print(f"Scheduler: Loaded {midi_event_count} MIDI events and {audio_clip_count} audio clips from {len(project.tracks)} tracks")

    def load_track(self, track: Track, append: bool = False):
        """Load events from a single track."""
        if not append:
            # Remove existing events from this track
            self._events = [e for e in self._events if e.track_id != track.id]

        if track.muted:
            return

        for clip in track.clips:
            if clip.muted or not clip.data:
                continue

            for note in clip.data.notes:
                # Note on
                self._events.append(ScheduledEvent(
                    tick=clip.start_tick + note.start_tick,
                    event_type='note_on',
                    channel=track.midi_channel,
                    note=note.pitch,
                    velocity=note.velocity,
                    track_id=track.id
                ))

                # Note off
                self._events.append(ScheduledEvent(
                    tick=clip.start_tick + note.start_tick + note.duration,
                    event_type='note_off',
                    channel=track.midi_channel,
                    note=note.pitch,
                    velocity=0,
                    track_id=track.id
                ))

        # Re-sort
        self._events.sort()
        self._reset_index()

    def load_audio_clip(self, clip_id: str, track_id: str, file_path: str,
                        start_tick: int, duration_ticks: int,
                        volume: float = 1.0, pan: float = 0.0) -> bool:
        """
        Load an audio clip for scheduled playback.

        Args:
            clip_id: Unique ID for this clip
            track_id: ID of the track this clip belongs to
            file_path: Path to audio file
            start_tick: Timeline position where clip starts
            duration_ticks: Duration in ticks
            volume: Clip volume (0.0-1.0)
            pan: Stereo pan (-1.0 left, 0.0 center, 1.0 right)

        Returns:
            True if loaded successfully
        """
        try:
            # Store clip info
            self._audio_clips[clip_id] = {
                'clip_id': clip_id,
                'track_id': track_id,
                'file_path': file_path,
                'start_tick': start_tick,
                'duration_ticks': duration_ticks,
                'volume': volume,
                'pan': pan
            }

            # Schedule start event
            self._audio_events.append(ScheduledAudioClipEvent(
                tick=start_tick,
                event_type='clip_start',
                clip_id=clip_id,
                track_id=track_id,
                file_path=file_path,
                duration_ticks=duration_ticks,
                volume=volume,
                pan=pan
            ))

            # Schedule stop event
            self._audio_events.append(ScheduledAudioClipEvent(
                tick=start_tick + duration_ticks,
                event_type='clip_stop',
                clip_id=clip_id,
                track_id=track_id
            ))

            # Re-sort audio events
            self._audio_events.sort()
            self._reset_audio_index()

            return True

        except Exception as e:
            print(f"Failed to load audio clip {clip_id}: {e}")
            return False

    def unload_audio_clip(self, clip_id: str) -> bool:
        """Remove a loaded audio clip."""
        if clip_id in self._audio_clips:
            del self._audio_clips[clip_id]
            # Remove events for this clip
            self._audio_events = [e for e in self._audio_events if e.clip_id != clip_id]
            self._reset_audio_index()
            return True
        return False

    def load_audio_track(self, track: Track, append: bool = False):
        """Load audio clip events from a track."""
        if not append:
            # Remove existing audio events from this track
            self._audio_events = [e for e in self._audio_events if e.track_id != track.id]
            # Remove clip info for this track
            self._audio_clips = {k: v for k, v in self._audio_clips.items()
                                 if v.get('track_id') != track.id}

        if track.muted:
            return

        for clip in track.clips:
            if clip.muted or clip.type != 'audio':
                continue

            # Get file path from clip data or metadata
            file_path = getattr(clip, 'file_path', None) or getattr(clip.data, 'file_path', None) if clip.data else None

            if file_path:
                self.load_audio_clip(
                    clip_id=clip.id,
                    track_id=track.id,
                    file_path=file_path,
                    start_tick=clip.start_tick,
                    duration_ticks=clip.duration,
                    volume=track.volume,
                    pan=track.pan
                )

        # Re-sort
        self._audio_events.sort()
        self._reset_audio_index()

    def _reset_audio_index(self):
        """Reset audio event index to match current transport position."""
        current_tick = self.transport.current_tick
        self._audio_event_index = 0

        for i, event in enumerate(self._audio_events):
            if event.tick >= current_tick:
                self._audio_event_index = i
                break
        else:
            self._audio_event_index = len(self._audio_events)

    def get_audio_clips_in_range(self, start_tick: int, end_tick: int) -> List[Dict]:
        """Get all audio clips in a tick range (for visualization)."""
        clips = []
        for clip_id, info in self._audio_clips.items():
            clip_start = info['start_tick']
            clip_end = clip_start + info['duration_ticks']
            if clip_start < end_tick and clip_end > start_tick:
                clips.append(info.copy())
        return clips

    def clear(self):
        """Clear all scheduled events (MIDI and audio)."""
        self._events.clear()
        self._event_index = 0
        self._last_processed_tick = -1
        # Clear audio events
        self._audio_events.clear()
        self._audio_event_index = 0
        self._audio_clips.clear()

    def _reset_index(self):
        """Reset event index to match current transport position."""
        current_tick = self.transport.current_tick
        self._event_index = 0

        for i, event in enumerate(self._events):
            if event.tick >= current_tick:
                self._event_index = i
                break
        else:
            self._event_index = len(self._events)

        self._last_processed_tick = current_tick - 1

        # Also reset audio index
        self._reset_audio_index()

    def _on_transport_state_change(self, state: TransportState):
        """Handle transport state changes."""
        if state == TransportState.PLAYING:
            self._reset_index()
            self._start_scheduler()
        elif state in (TransportState.STOPPED, TransportState.PAUSED):
            self._stop_scheduler()
            # Send all notes off
            self._all_notes_off()
            # Stop all audio clips
            self._stop_all_audio_clips()

    def _on_beat(self, bar: int, beat: int):
        """Handle beat events for metronome."""
        if self.transport.config.click_enabled and self._click_callback:
            try:
                self._click_callback(bar, beat)
            except Exception as e:
                print(f"Click callback error: {e}")

    def _start_scheduler(self):
        """Start the scheduler thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def _stop_scheduler(self):
        """Stop the scheduler thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.1)
            self._thread = None

    def _scheduler_loop(self):
        """Main scheduler loop - processes MIDI and audio events based on transport position."""
        while self._running:
            current_tick = self.transport.current_tick

            # Process MIDI events up to current position
            while (self._event_index < len(self._events) and
                   self._events[self._event_index].tick <= current_tick):

                event = self._events[self._event_index]
                self._trigger_event(event)
                self._event_index += 1

            # Process audio clip events up to current position
            while (self._audio_event_index < len(self._audio_events) and
                   self._audio_events[self._audio_event_index].tick <= current_tick):

                audio_event = self._audio_events[self._audio_event_index]
                self._trigger_audio_event(audio_event)
                self._audio_event_index += 1

            # Sleep briefly
            time.sleep(0.001)  # 1ms resolution

    def _trigger_event(self, event: ScheduledEvent):
        """Trigger a MIDI event."""
        try:
            if event.event_type == 'note_on':
                if self._note_on_callback:
                    self._note_on_callback(event.channel, event.note, event.velocity, event.track_id)
            elif event.event_type == 'note_off':
                if self._note_off_callback:
                    self._note_off_callback(event.channel, event.note, event.track_id)
        except Exception as e:
            print(f"Event trigger error: {e}")

    def _trigger_audio_event(self, event: ScheduledAudioClipEvent):
        """Trigger an audio clip event."""
        try:
            if event.event_type == 'clip_start':
                if self._audio_clip_start_callback:
                    self._audio_clip_start_callback(
                        event.clip_id,
                        event.track_id,
                        event.file_path,
                        event.duration_ticks,
                        event.volume,
                        event.pan
                    )
            elif event.event_type == 'clip_stop':
                if self._audio_clip_stop_callback:
                    self._audio_clip_stop_callback(event.clip_id, event.track_id)
        except Exception as e:
            print(f"Audio event trigger error: {e}")

    def _all_notes_off(self):
        """Send note off for all active notes."""
        if not self._note_off_callback:
            return

        # Track which notes are currently on
        active_notes: Dict[Tuple[int, int, str], bool] = {}

        for i in range(self._event_index):
            event = self._events[i]
            key = (event.channel, event.note, event.track_id)
            if event.event_type == 'note_on':
                active_notes[key] = True
            elif event.event_type == 'note_off':
                active_notes.pop(key, None)

        # Send note off for all active notes
        for (channel, note, track_id) in active_notes.keys():
            try:
                self._note_off_callback(channel, note, track_id)
            except Exception as e:
                print(f"Note off error: {e}")

    def _stop_all_audio_clips(self):
        """Stop all currently playing audio clips."""
        if not self._audio_clip_stop_callback:
            return

        # Track which clips are currently playing
        active_clips: Dict[str, str] = {}  # clip_id -> track_id

        for i in range(self._audio_event_index):
            event = self._audio_events[i]
            if event.event_type == 'clip_start':
                active_clips[event.clip_id] = event.track_id
            elif event.event_type == 'clip_stop':
                active_clips.pop(event.clip_id, None)

        # Send stop for all active clips
        for clip_id, track_id in active_clips.items():
            try:
                self._audio_clip_stop_callback(clip_id, track_id)
            except Exception as e:
                print(f"Audio clip stop error: {e}")

    def get_events_in_range(self, start_tick: int, end_tick: int) -> List[ScheduledEvent]:
        """Get all MIDI events in a tick range (for visualization)."""
        return [e for e in self._events if start_tick <= e.tick < end_tick]

    def get_audio_events_in_range(self, start_tick: int, end_tick: int) -> List[ScheduledAudioClipEvent]:
        """Get all audio clip events in a tick range (for visualization)."""
        return [e for e in self._audio_events if start_tick <= e.tick < end_tick]

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            'midi_events': len(self._events),
            'audio_events': len(self._audio_events),
            'audio_clips': len(self._audio_clips),
            'midi_event_index': self._event_index,
            'audio_event_index': self._audio_event_index
        }


# Standalone test
if __name__ == "__main__":
    from project import Project, Track, Clip, MidiNote, MidiClipData

    # Create transport
    transport = Transport()

    # Create scheduler
    scheduler = MidiScheduler(transport)

    # Set up MIDI callbacks
    def on_note_on(channel, note, velocity, track_id):
        print(f"MIDI NOTE ON: ch={channel} note={note} vel={velocity}")

    def on_note_off(channel, note, track_id):
        print(f"MIDI NOTE OFF: ch={channel} note={note}")

    def on_click(bar, beat):
        accent = "*" if beat == 1 else ""
        print(f"CLICK: {bar}.{beat} {accent}")

    # Set up audio clip callbacks
    def on_audio_clip_start(clip_id, track_id, file_path, duration_ticks, volume, pan):
        print(f"AUDIO CLIP START: {clip_id} on track {track_id}, file={file_path}, duration={duration_ticks} ticks")

    def on_audio_clip_stop(clip_id, track_id):
        print(f"AUDIO CLIP STOP: {clip_id} on track {track_id}")

    scheduler.set_note_on_callback(on_note_on)
    scheduler.set_note_off_callback(on_note_off)
    scheduler.set_click_callback(on_click)
    scheduler.set_audio_clip_start_callback(on_audio_clip_start)
    scheduler.set_audio_clip_stop_callback(on_audio_clip_stop)

    # Create a simple project
    project = Project.create("Test")

    # Add a drum track with a pattern
    drum_track = Track.create_drum("Drums")

    # Create a clip with a basic beat
    clip = Clip.create_midi("Beat", 0, 1920)
    clip.data = MidiClipData(
        notes=[
            MidiNote.create(36, 100, 0, 120),      # Kick on 1
            MidiNote.create(42, 80, 0, 60),        # HiHat
            MidiNote.create(42, 80, 240, 60),      # HiHat
            MidiNote.create(38, 100, 480, 120),    # Snare on 2
            MidiNote.create(42, 80, 480, 60),      # HiHat
            MidiNote.create(42, 80, 720, 60),      # HiHat
            MidiNote.create(36, 100, 960, 120),    # Kick on 3
            MidiNote.create(42, 80, 960, 60),      # HiHat
            MidiNote.create(42, 80, 1200, 60),     # HiHat
            MidiNote.create(38, 100, 1440, 120),   # Snare on 4
            MidiNote.create(42, 80, 1440, 60),     # HiHat
            MidiNote.create(42, 80, 1680, 60),     # HiHat
        ],
        duration=1920
    )
    drum_track.clips.append(clip)
    project.tracks.append(drum_track)

    # Load project into scheduler
    scheduler.load_project(project)

    # Test audio clip scheduling (simulated - no actual audio file)
    print("\nTesting audio clip scheduling...")
    scheduler.load_audio_clip(
        clip_id="audio_test_1",
        track_id="test_track",
        file_path="/path/to/test.wav",
        start_tick=480,  # Start at beat 2
        duration_ticks=960,  # 2 beats duration
        volume=0.8,
        pan=0.0
    )

    # Get scheduler stats
    stats = scheduler.get_stats()
    print(f"Scheduler stats: {stats}")

    # Play
    print("\nStarting playback...")
    transport.bpm = 120
    transport.play()

    try:
        time.sleep(4)  # Play for 4 seconds (2 bars at 120 BPM)
    except KeyboardInterrupt:
        pass

    transport.stop()
    print("Done")
