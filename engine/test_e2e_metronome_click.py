"""
End-to-end verification test for metronome click during playback.

Tests the complete flow:
1. Verify click samples are generated
2. Enable click in transport
3. Start playback
4. Verify click callback is invoked on each beat
5. Verify audio synthesis fallback works when MIDI unavailable
"""

import asyncio
import numpy as np
import time
import sys
import threading

# Add engine directory to path
sys.path.insert(0, '.')

from main import BeatBoxDawEngine, EngineConfig
from transport import TransportState


def test_metronome_click_flow():
    """Test the complete metronome click E2E flow."""
    print("=" * 60)
    print("E2E Metronome Click Verification")
    print("=" * 60)

    results = []

    # Create engine with click enabled
    config = EngineConfig()
    engine = BeatBoxDawEngine(config)

    print("\n[1/6] Verifying click samples are generated...")
    # Check that click samples exist and have correct structure
    has_click_samples = hasattr(engine, '_click_samples') and engine._click_samples is not None
    has_downbeat = has_click_samples and 'downbeat' in engine._click_samples
    has_beat = has_click_samples and 'beat' in engine._click_samples

    if has_downbeat:
        downbeat_len = len(engine._click_samples['downbeat'])
        print(f"  Downbeat samples: {downbeat_len} samples")
    if has_beat:
        beat_len = len(engine._click_samples['beat'])
        print(f"  Regular beat samples: {beat_len} samples")

    samples_valid = (has_downbeat and has_beat and
                     len(engine._click_samples['downbeat']) > 0 and
                     len(engine._click_samples['beat']) > 0)

    results.append(("Click samples generated",
                   samples_valid,
                   f"Downbeat: {has_downbeat}, Beat: {has_beat}"))

    print("\n[2/6] Verifying click callback is registered...")
    # Check scheduler has click callback set
    callback_set = engine.scheduler._click_callback is not None
    print(f"  Click callback set: {callback_set}")
    results.append(("Click callback registered",
                   callback_set,
                   f"Callback: {engine.scheduler._click_callback}"))

    print("\n[3/6] Enabling click in transport...")
    # Enable click (it's enabled by default, but verify explicitly)
    engine.transport.config.click_enabled = True
    click_enabled = engine.transport.config.click_enabled
    print(f"  Click enabled: {click_enabled}")
    results.append(("Click enabled",
                   click_enabled,
                   "Transport click_enabled = True"))

    print("\n[4/6] Starting playback and tracking click events...")
    # Track click events
    click_events = []
    original_on_click = engine._on_click

    def track_click(bar, beat):
        click_events.append({'bar': bar, 'beat': beat, 'time': time.time()})
        original_on_click(bar, beat)

    engine.scheduler._click_callback = track_click

    # Start transport playback
    playback_started = engine.transport.play()
    print(f"  Playback started: {playback_started}")

    # Wait for some clicks to occur (at least 2 bars at 120 BPM = ~4 seconds)
    # At 120 BPM, 1 beat = 0.5 seconds, so 4 beats/bar, 2 bars = 8 beats = 4 seconds
    # Let's wait 2.5 seconds for at least 4-5 beats
    wait_time = 2.5
    print(f"  Waiting {wait_time}s for click events...")

    # The transport runs its own thread for beat callbacks
    # Just wait and let the transport fire callbacks
    time.sleep(wait_time)

    # Stop transport
    engine.transport.stop()
    print(f"  Transport stopped")

    # Verify we got click events
    num_clicks = len(click_events)
    print(f"  Click events captured: {num_clicks}")

    if num_clicks > 0:
        print(f"  First click: bar={click_events[0]['bar']}, beat={click_events[0]['beat']}")
        print(f"  Last click: bar={click_events[-1]['bar']}, beat={click_events[-1]['beat']}")

    # At 120 BPM, in 2.5 seconds we should get ~5 beats
    expected_min_clicks = 3
    clicks_valid = num_clicks >= expected_min_clicks

    results.append(("Click events during playback",
                   clicks_valid,
                   f"Captured {num_clicks} clicks (expected >= {expected_min_clicks})"))

    print("\n[5/6] Testing audio synthesis fallback...")
    # Verify _play_click_audio works (MIDI will likely not be connected in test)

    # First, verify MIDI is not connected (expected in test environment)
    midi_connected = engine.midi_output.is_connected if hasattr(engine, 'midi_output') else False
    print(f"  MIDI connected: {midi_connected}")

    # Test audio playback function directly (non-blocking)
    audio_play_success = False
    try:
        # This should not throw even if audio device isn't perfect
        engine._play_click_audio(True)  # Downbeat
        engine._play_click_audio(False)  # Regular beat
        audio_play_success = True
        print(f"  Audio synthesis test: Success (no errors thrown)")
    except Exception as e:
        print(f"  Audio synthesis test: Failed ({e})")

    results.append(("Audio synthesis fallback",
                   audio_play_success,
                   f"MIDI: {midi_connected}, Audio fallback: {audio_play_success}"))

    print("\n[6/6] Verifying click plays on correct beats...")
    # Check that clicks happened on expected beat pattern (beats 1,2,3,4 in 4/4 time)
    beats_pattern_valid = True
    if num_clicks > 0:
        for event in click_events:
            beat = event['beat']
            if beat < 1 or beat > 4:  # Standard 4/4 time
                beats_pattern_valid = False
                print(f"  Invalid beat number: {beat}")
                break

        # Check we got a downbeat (beat 1)
        has_downbeat_event = any(e['beat'] == 1 for e in click_events)
        print(f"  Has downbeat (beat 1): {has_downbeat_event}")

        # Check beat progression
        if len(click_events) >= 2:
            print(f"  Beat sequence: {[e['beat'] for e in click_events[:8]]}")

    results.append(("Beat pattern valid",
                   beats_pattern_valid and num_clicks > 0,
                   f"Beats follow 4/4 pattern, {num_clicks} events"))

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed, details in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}: {details}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up...")
    try:
        engine.transport.stop()
    except Exception as cleanup_err:
        print(f"  Transport stop: {cleanup_err}")

    try:
        if hasattr(engine, 'audio_capture'):
            engine.audio_capture.stop()
    except Exception as cleanup_err:
        print(f"  Audio capture stop: {cleanup_err}")

    print("Cleanup complete.")

    return all_passed


if __name__ == "__main__":
    success = test_metronome_click_flow()
    sys.exit(0 if success else 1)
