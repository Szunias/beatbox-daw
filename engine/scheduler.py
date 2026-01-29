"""
MIDI Event Scheduler Module
Handles scheduling and playback of MIDI events with precise timing.
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


class MidiScheduler:
    """
    Schedules and triggers MIDI events during playback.
    Works with Transport to handle timing and playback position.
    """

    def __init__(self, transport: Transport):
        self.transport = transport
        self._events: List[ScheduledEvent] = []
        self._event_index = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._note_on_callback: Optional[Callable[[int, int, int, str], None]] = None
        self._note_off_callback: Optional[Callable[[int, int, str], None]] = None
        self._click_callback: Optional[Callable[[int, int], None]] = None

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

    def load_project(self, project: Project):
        """Load all MIDI events from project into scheduler."""
        self._events.clear()

        for track in project.tracks:
            if track.muted:
                continue

            for clip in track.clips:
                if clip.muted or not clip.data:
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

                    # Note off
                    self._events.append(ScheduledEvent(
                        tick=clip.start_tick + note.start_tick + note.duration,
                        event_type='note_off',
                        channel=track.midi_channel,
                        note=note.pitch,
                        velocity=0,
                        track_id=track.id
                    ))

        # Sort by tick
        self._events.sort()
        self._event_index = 0
        self._last_processed_tick = -1

        print(f"Scheduler: Loaded {len(self._events)} events from {len(project.tracks)} tracks")

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

    def clear(self):
        """Clear all scheduled events."""
        self._events.clear()
        self._event_index = 0
        self._last_processed_tick = -1

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

    def _on_transport_state_change(self, state: TransportState):
        """Handle transport state changes."""
        if state == TransportState.PLAYING:
            self._reset_index()
            self._start_scheduler()
        elif state in (TransportState.STOPPED, TransportState.PAUSED):
            self._stop_scheduler()
            # Send all notes off
            self._all_notes_off()

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
        """Main scheduler loop - processes events based on transport position."""
        while self._running:
            current_tick = self.transport.current_tick

            # Process events up to current position
            while (self._event_index < len(self._events) and
                   self._events[self._event_index].tick <= current_tick):

                event = self._events[self._event_index]
                self._trigger_event(event)
                self._event_index += 1

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

    def get_events_in_range(self, start_tick: int, end_tick: int) -> List[ScheduledEvent]:
        """Get all events in a tick range (for visualization)."""
        return [e for e in self._events if start_tick <= e.tick < end_tick]


# Standalone test
if __name__ == "__main__":
    from project import Project, Track, Clip, MidiNote, MidiClipData

    # Create transport
    transport = Transport()

    # Create scheduler
    scheduler = MidiScheduler(transport)

    # Set up callbacks
    def on_note_on(channel, note, velocity, track_id):
        print(f"NOTE ON: ch={channel} note={note} vel={velocity}")

    def on_note_off(channel, note, track_id):
        print(f"NOTE OFF: ch={channel} note={note}")

    def on_click(bar, beat):
        accent = "*" if beat == 1 else ""
        print(f"CLICK: {bar}.{beat} {accent}")

    scheduler.set_note_on_callback(on_note_on)
    scheduler.set_note_off_callback(on_note_off)
    scheduler.set_click_callback(on_click)

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

    # Play
    print("Starting playback...")
    transport.bpm = 120
    transport.play()

    try:
        time.sleep(4)  # Play for 4 seconds (2 bars at 120 BPM)
    except KeyboardInterrupt:
        pass

    transport.stop()
    print("Done")
