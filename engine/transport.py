"""
Transport Control Module
Handles DAW playback, recording, and timeline position management.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, List
import threading


class TransportState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    RECORDING = "recording"
    PRE_ROLL = "pre_roll"  # Counting in before recording


@dataclass
class LoopRegion:
    """Loop region configuration."""
    enabled: bool = False
    start_tick: int = 0
    end_tick: int = 7680  # 4 bars at 480 PPQN


@dataclass
class TransportConfig:
    """Transport configuration."""
    bpm: float = 120.0
    time_sig_numerator: int = 4
    time_sig_denominator: int = 4
    ppqn: int = 480  # Pulses per quarter note (standard MIDI resolution)
    click_enabled: bool = True
    click_volume: float = 0.7
    pre_roll_bars: int = 1


class Transport:
    """
    Transport controller for DAW playback and recording.
    Manages timeline position, BPM, and synchronization.
    """

    def __init__(self, config: Optional[TransportConfig] = None):
        self.config = config or TransportConfig()
        self._state = TransportState.STOPPED
        self._current_tick = 0
        self._start_time: Optional[float] = None
        self._start_tick: int = 0
        self._loop = LoopRegion()

        # Callbacks
        self._position_callbacks: List[Callable[[int], None]] = []
        self._state_callbacks: List[Callable[[TransportState], None]] = []
        self._beat_callbacks: List[Callable[[int, int], None]] = []  # bar, beat
        self._pre_roll_callbacks: List[Callable[[int, int], None]] = []  # bar, beat during pre-roll

        # Internal timing - use perf_counter for higher precision
        self._last_beat = -1
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

        # Pre-roll state
        self._pre_roll_start_time: Optional[float] = None
        self._pre_roll_ticks: int = 0
        self._record_after_pre_roll: bool = False

    @property
    def state(self) -> TransportState:
        return self._state

    @property
    def current_tick(self) -> int:
        """Get current timeline position in ticks."""
        if self._state == TransportState.PRE_ROLL:
            # During pre-roll, return negative ticks (counting down to zero)
            elapsed = time.perf_counter() - self._pre_roll_start_time
            ticks_elapsed = self._seconds_to_ticks(elapsed)
            remaining = int(self._pre_roll_ticks - ticks_elapsed)
            return -remaining if remaining > 0 else 0

        if self._state in (TransportState.PLAYING, TransportState.RECORDING):
            # Use perf_counter for higher precision timing
            elapsed = time.perf_counter() - self._start_time
            ticks = self._start_tick + self._seconds_to_ticks(elapsed)

            # Handle loop
            if self._loop.enabled and ticks >= self._loop.end_tick:
                loop_length = self._loop.end_tick - self._loop.start_tick
                ticks = self._loop.start_tick + ((ticks - self._loop.start_tick) % loop_length)

            return int(ticks)
        return self._current_tick

    @property
    def bpm(self) -> float:
        return self.config.bpm

    @bpm.setter
    def bpm(self, value: float):
        """Set BPM (20-300 range). Recalibrates timing if playback is active."""
        new_bpm = max(20.0, min(300.0, value))

        # If playing or recording, recalibrate timing to prevent position jumps
        if self._state in (TransportState.PLAYING, TransportState.RECORDING) and self._start_time is not None:
            # Capture current position using OLD BPM before changing
            current_tick = self.current_tick

            # Update reference point to current position
            self._start_tick = current_tick
            self._start_time = time.perf_counter()

            print(f"Transport: BPM changed from {self.config.bpm} to {new_bpm} at tick {current_tick}")

        self.config.bpm = new_bpm

    @property
    def ticks_per_beat(self) -> int:
        return self.config.ppqn

    @property
    def ticks_per_bar(self) -> int:
        return self.config.ppqn * self.config.time_sig_numerator

    def _seconds_to_ticks(self, seconds: float) -> float:
        """Convert seconds to ticks based on current BPM."""
        return seconds * (self.config.bpm / 60.0) * self.config.ppqn

    def _ticks_to_seconds(self, ticks: int) -> float:
        """Convert ticks to seconds based on current BPM."""
        return (ticks / self.config.ppqn) * (60.0 / self.config.bpm)

    def _ticks_to_bar_beat(self, ticks: int) -> tuple:
        """Convert ticks to bar and beat numbers."""
        bar = ticks // self.ticks_per_bar + 1
        beat = (ticks % self.ticks_per_bar) // self.config.ppqn + 1
        return bar, beat

    def play(self) -> bool:
        """Start playback."""
        if self._state == TransportState.PLAYING:
            return True

        # Use perf_counter for higher precision timing
        self._start_time = time.perf_counter()
        self._start_tick = self._current_tick
        self._state = TransportState.PLAYING
        self._running = True

        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

        self._notify_state_change()
        print(f"Transport: Playing from tick {self._current_tick}")
        return True

    def pause(self):
        """Pause playback."""
        if self._state not in (TransportState.PLAYING, TransportState.RECORDING):
            return

        self._current_tick = self.current_tick
        self._state = TransportState.PAUSED
        self._running = False
        self._notify_state_change()
        print(f"Transport: Paused at tick {self._current_tick}")

    def stop(self):
        """Stop playback and return to start."""
        was_active = self._state in (TransportState.PLAYING, TransportState.RECORDING, TransportState.PRE_ROLL)
        self._state = TransportState.STOPPED
        self._running = False
        self._current_tick = 0
        self._last_beat = -1
        self._record_after_pre_roll = False

        if was_active:
            self._notify_state_change()
        print("Transport: Stopped")

    def record(self, use_pre_roll: bool = True) -> bool:
        """Start recording, optionally with pre-roll count-in."""
        if self._state == TransportState.RECORDING:
            # Stop recording
            self._current_tick = self.current_tick
            self._state = TransportState.STOPPED
            self._running = False
            self._record_after_pre_roll = False
            self._notify_state_change()
            print("Transport: Recording stopped")
            return False
        elif self._state == TransportState.PRE_ROLL:
            # Cancel pre-roll
            self._state = TransportState.STOPPED
            self._running = False
            self._record_after_pre_roll = False
            self._notify_state_change()
            print("Transport: Pre-roll cancelled")
            return False
        else:
            # Start recording (with or without pre-roll)
            if use_pre_roll and self.config.pre_roll_bars > 0:
                return self._start_pre_roll()
            else:
                return self._start_recording_immediate()

    def _start_pre_roll(self) -> bool:
        """Start pre-roll count-in before recording."""
        self._pre_roll_ticks = self.config.pre_roll_bars * self.ticks_per_bar
        self._pre_roll_start_time = time.perf_counter()
        self._record_after_pre_roll = True
        self._state = TransportState.PRE_ROLL
        self._running = True

        # Start update thread for pre-roll
        self._update_thread = threading.Thread(target=self._pre_roll_loop, daemon=True)
        self._update_thread.start()

        self._notify_state_change()
        print(f"Transport: Pre-roll started ({self.config.pre_roll_bars} bars)")
        return True

    def _start_recording_immediate(self) -> bool:
        """Start recording immediately without pre-roll."""
        # Use perf_counter for higher precision timing
        self._start_time = time.perf_counter()
        self._start_tick = self._current_tick
        self._state = TransportState.RECORDING
        self._running = True

        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

        self._notify_state_change()
        print(f"Transport: Recording from tick {self._current_tick}")
        return True

    def _pre_roll_loop(self):
        """Background thread for pre-roll countdown."""
        pre_roll_beat = -1

        while self._running and self._state == TransportState.PRE_ROLL:
            elapsed = time.perf_counter() - self._pre_roll_start_time
            ticks_elapsed = self._seconds_to_ticks(elapsed)

            # Check if pre-roll is complete
            if ticks_elapsed >= self._pre_roll_ticks:
                # Transition to recording
                if self._record_after_pre_roll:
                    self._running = False  # Stop pre-roll loop
                    self._start_recording_immediate()
                return

            # Calculate current beat in pre-roll (for click track)
            current_beat = int(ticks_elapsed // self.config.ppqn)
            if current_beat != pre_roll_beat:
                pre_roll_beat = current_beat
                bar = current_beat // self.config.time_sig_numerator + 1
                beat = current_beat % self.config.time_sig_numerator + 1

                # Notify pre-roll callbacks (for metronome)
                for callback in self._pre_roll_callbacks:
                    try:
                        callback(bar, beat)
                    except Exception as e:
                        print(f"Pre-roll callback error: {e}")

                # Also notify beat callbacks during pre-roll
                for callback in self._beat_callbacks:
                    try:
                        callback(-bar, beat)  # Negative bar indicates pre-roll
                    except Exception as e:
                        print(f"Beat callback error: {e}")

            # High precision sleep - 1ms for accurate timing
            time.sleep(0.001)

    def seek(self, tick: int):
        """Seek to specific tick position."""
        was_playing = self._state == TransportState.PLAYING
        if was_playing:
            self.pause()

        self._current_tick = max(0, tick)
        self._notify_position_change()

        if was_playing:
            self.play()

        print(f"Transport: Seeked to tick {self._current_tick}")

    def seek_to_bar(self, bar: int):
        """Seek to start of specific bar."""
        tick = (bar - 1) * self.ticks_per_bar
        self.seek(tick)

    def set_loop(self, enabled: bool, start_tick: int = None, end_tick: int = None):
        """Configure loop region."""
        self._loop.enabled = enabled
        if start_tick is not None:
            self._loop.start_tick = max(0, start_tick)
        if end_tick is not None:
            self._loop.end_tick = max(self._loop.start_tick + self.config.ppqn, end_tick)
        print(f"Transport: Loop {'enabled' if enabled else 'disabled'} "
              f"({self._loop.start_tick} - {self._loop.end_tick})")

    def set_time_signature(self, numerator: int, denominator: int):
        """Set time signature."""
        self.config.time_sig_numerator = max(1, min(16, numerator))
        self.config.time_sig_denominator = max(1, min(16, denominator))
        print(f"Transport: Time signature set to {numerator}/{denominator}")

    def _update_loop(self):
        """Background thread for position updates and beat callbacks."""
        while self._running:
            current = self.current_tick
            self._notify_position_change()

            # Check for beat boundaries
            current_beat = current // self.config.ppqn
            if current_beat != self._last_beat:
                self._last_beat = current_beat
                bar, beat = self._ticks_to_bar_beat(current)
                for callback in self._beat_callbacks:
                    try:
                        callback(bar, beat)
                    except Exception as e:
                        print(f"Beat callback error: {e}")

            # Sleep for ~1ms for higher precision timing
            # This provides better timing accuracy while keeping CPU usage reasonable
            time.sleep(0.001)

    def _notify_position_change(self):
        """Notify position change callbacks."""
        current = self.current_tick
        for callback in self._position_callbacks:
            try:
                callback(current)
            except Exception as e:
                print(f"Position callback error: {e}")

    def _notify_state_change(self):
        """Notify state change callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(self._state)
            except Exception as e:
                print(f"State callback error: {e}")

    def add_position_callback(self, callback: Callable[[int], None]):
        """Register callback for position changes."""
        self._position_callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[TransportState], None]):
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def add_beat_callback(self, callback: Callable[[int, int], None]):
        """Register callback for beat events (bar, beat)."""
        self._beat_callbacks.append(callback)

    def add_pre_roll_callback(self, callback: Callable[[int, int], None]):
        """Register callback for pre-roll beat events (bar, beat)."""
        self._pre_roll_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get current transport status."""
        current_tick = self.current_tick
        bar, beat = self._ticks_to_bar_beat(max(0, current_tick))

        # Handle pre-roll status
        pre_roll_remaining = 0
        if self._state == TransportState.PRE_ROLL:
            pre_roll_remaining = -current_tick if current_tick < 0 else 0

        return {
            'state': self._state.value,
            'current_tick': current_tick,
            'bar': bar,
            'beat': beat,
            'bpm': self.config.bpm,
            'time_sig': f"{self.config.time_sig_numerator}/{self.config.time_sig_denominator}",
            'loop_enabled': self._loop.enabled,
            'loop_start': self._loop.start_tick,
            'loop_end': self._loop.end_tick,
            'click_enabled': self.config.click_enabled,
            'pre_roll_bars': self.config.pre_roll_bars,
            'pre_roll_remaining': pre_roll_remaining,
        }

    def set_pre_roll_bars(self, bars: int):
        """Set the number of pre-roll bars before recording."""
        self.config.pre_roll_bars = max(0, min(8, bars))
        print(f"Transport: Pre-roll set to {self.config.pre_roll_bars} bars")


# Standalone test
if __name__ == "__main__":
    transport = Transport()

    def on_beat(bar, beat):
        print(f"Beat: {bar}.{beat}")

    transport.add_beat_callback(on_beat)

    print("Starting playback...")
    transport.play()

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass

    transport.stop()
    print("Done")
