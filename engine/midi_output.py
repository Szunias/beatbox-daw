"""MIDI output module for sending drum events."""

import mido
from mido import Message, MidiFile, MidiTrack
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
import time


@dataclass
class MidiEvent:
    """Represents a MIDI drum event."""
    note: int  # MIDI note number (GM drum map)
    velocity: int  # 0-127
    timestamp: float  # Time in seconds
    duration: float = 0.1  # Note duration in seconds
    channel: int = 9  # MIDI channel 10 (0-indexed = 9) for drums


class MidiOutput:
    """Handles MIDI output to virtual port or file."""

    # General MIDI Drum Map (common percussion)
    GM_DRUM_MAP = {
        'kick': 36,       # Bass Drum 1
        'snare': 38,      # Acoustic Snare
        'hihat': 42,      # Closed Hi-Hat
        'hihat_open': 46, # Open Hi-Hat
        'clap': 39,       # Hand Clap
        'tom_low': 45,    # Low Tom
        'tom_mid': 47,    # Low-Mid Tom
        'tom_high': 50,   # High Tom
        'crash': 49,      # Crash Cymbal 1
        'ride': 51,       # Ride Cymbal 1
        'rim': 37,        # Side Stick
    }

    def __init__(self, port_name: str = "BeatBox MIDI"):
        self.port_name = port_name
        self.port: Optional[mido.ports.BaseOutput] = None
        self.is_connected = False

        # Event queue for async sending
        self.event_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Recording for export
        self.recording: List[MidiEvent] = []
        self.is_recording = False
        self.recording_start_time: Optional[float] = None

    def connect(self) -> bool:
        """Connect to MIDI output port."""
        try:
            # Try to open a virtual port (works on Linux/Mac with rtmidi)
            # On Windows, may need loopMIDI or similar
            available_ports = mido.get_output_names()
            print(f"Available MIDI outputs: {available_ports}")

            # Try to find existing port with our name
            for port in available_ports:
                if self.port_name.lower() in port.lower():
                    self.port = mido.open_output(port)
                    self.is_connected = True
                    print(f"Connected to existing port: {port}")
                    return True

            # Try to create virtual port
            try:
                self.port = mido.open_output(self.port_name, virtual=True)
                self.is_connected = True
                print(f"Created virtual MIDI port: {self.port_name}")
                return True
            except Exception as e:
                print(f"Could not create virtual port: {e}")

                # Fall back to first available port
                if available_ports:
                    self.port = mido.open_output(available_ports[0])
                    self.is_connected = True
                    print(f"Connected to fallback port: {available_ports[0]}")
                    return True

        except Exception as e:
            print(f"MIDI connection error: {e}")

        return False

    def disconnect(self) -> None:
        """Disconnect from MIDI port."""
        if self.port:
            self.port.close()
            self.port = None
        self.is_connected = False
        print("MIDI disconnected")

    def send_note(self, note: int, velocity: int = 100,
                  duration: float = 0.1, channel: int = 9) -> None:
        """Send a MIDI note on/off pair."""
        if not self.is_connected or not self.port:
            return

        try:
            # Note on
            msg_on = Message('note_on', note=note, velocity=velocity, channel=channel)
            self.port.send(msg_on)

            # Schedule note off
            def send_note_off_delayed():
                time.sleep(duration)
                if self.port and self.is_connected:
                    msg_off = Message('note_off', note=note, velocity=0, channel=channel)
                    self.port.send(msg_off)

            threading.Thread(target=send_note_off_delayed, daemon=True).start()

        except Exception as e:
            print(f"Error sending MIDI note: {e}")

    def send_note_off(self, note: int, channel: int = 9) -> None:
        """Send a MIDI note off immediately."""
        if not self.is_connected or not self.port:
            return

        try:
            msg_off = Message('note_off', note=note, velocity=0, channel=channel)
            self.port.send(msg_off)
        except Exception as e:
            print(f"Error sending MIDI note off: {e}")

    def send_drum_hit(self, drum_class: str, velocity: int = 100) -> None:
        """Send a drum hit by class name."""
        note = self.GM_DRUM_MAP.get(drum_class, 36)  # Default to kick
        self.send_note(note, velocity, channel=9)

        # Record if active
        if self.is_recording and self.recording_start_time:
            event = MidiEvent(
                note=note,
                velocity=velocity,
                timestamp=time.time() - self.recording_start_time
            )
            self.recording.append(event)

    def start_recording(self) -> None:
        """Start recording MIDI events."""
        self.recording = []
        self.recording_start_time = time.time()
        self.is_recording = True
        print("MIDI recording started")

    def stop_recording(self) -> List[MidiEvent]:
        """Stop recording and return events."""
        self.is_recording = False
        events = self.recording.copy()
        print(f"MIDI recording stopped: {len(events)} events")
        return events

    def export_midi_file(self, filename: str, events: Optional[List[MidiEvent]] = None,
                         bpm: int = 120) -> bool:
        """Export recorded events to MIDI file."""
        if events is None:
            events = self.recording

        if not events:
            print("No events to export")
            return False

        try:
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)

            # Set tempo
            tempo = mido.bpm2tempo(bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo))

            # Sort events by timestamp
            events = sorted(events, key=lambda e: e.timestamp)

            # Convert to MIDI messages with delta times
            ticks_per_beat = mid.ticks_per_beat
            last_tick = 0

            for event in events:
                # Convert time to ticks
                beat_time = event.timestamp * bpm / 60
                current_tick = int(beat_time * ticks_per_beat)
                delta = max(0, current_tick - last_tick)

                # Note on
                track.append(Message('note_on', note=event.note,
                                    velocity=event.velocity,
                                    channel=event.channel,
                                    time=delta))

                # Note off (short duration for drums)
                note_off_delta = int(event.duration * bpm / 60 * ticks_per_beat)
                track.append(Message('note_off', note=event.note,
                                    velocity=0,
                                    channel=event.channel,
                                    time=note_off_delta))

                last_tick = current_tick + note_off_delta

            # Save file
            mid.save(filename)
            print(f"Exported MIDI file: {filename}")
            return True

        except Exception as e:
            print(f"Error exporting MIDI: {e}")
            return False

    @staticmethod
    def list_ports() -> dict:
        """List available MIDI ports."""
        return {
            'inputs': mido.get_input_names(),
            'outputs': mido.get_output_names()
        }


if __name__ == "__main__":
    # Test MIDI output
    print("Available MIDI ports:")
    ports = MidiOutput.list_ports()
    print(f"  Inputs: {ports['inputs']}")
    print(f"  Outputs: {ports['outputs']}")

    # Test connection
    midi = MidiOutput()
    if midi.connect():
        print("\nSending test drum pattern...")

        # Simple pattern: kick-hihat-snare-hihat
        pattern = [
            ('kick', 100),
            ('hihat', 80),
            ('snare', 100),
            ('hihat', 80),
        ]

        midi.start_recording()

        for drum, vel in pattern:
            midi.send_drum_hit(drum, vel)
            print(f"  Sent: {drum} (vel={vel})")
            time.sleep(0.25)

        events = midi.stop_recording()

        # Export to file
        midi.export_midi_file("test_output.mid", events, bpm=120)

        midi.disconnect()
    else:
        print("Could not connect to MIDI output")
