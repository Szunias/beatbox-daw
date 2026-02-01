"""
End-to-end verification test for loop region playback.

Tests the complete flow:
1. Add some content to timeline
2. Configure loop region (set start and end ticks)
3. Enable loop mode
4. Start playback
5. Verify playback loops within region (position wraps back to start)
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
from project import Clip, MidiNote


def test_loop_region_playback():
    """Test the complete loop region playback E2E flow."""
    print("=" * 60)
    print("E2E Loop Region Playback Verification")
    print("=" * 60)

    results = []

    # Create engine
    config = EngineConfig()
    engine = BeatBoxDawEngine(config)

    # Use 120 BPM for predictable timing
    engine.transport.bpm = 120.0
    ppqn = engine.transport.ticks_per_beat  # 480 ticks per beat
    ticks_per_bar = engine.transport.ticks_per_bar  # 1920 ticks per bar at 4/4

    print("\n[1/7] Adding content to timeline...")
    # Add a drum track with a clip (content on timeline)
    track_data = engine.add_track('drum', 'Loop Test Track')
    track_id = track_data['id']
    print(f"  Created track: {track_id}")

    # Create MIDI notes for the clip (simple 4-beat pattern)
    notes = [
        MidiNote.create(pitch=36, velocity=100, start_tick=0, duration=ppqn // 4),  # Kick on beat 1
        MidiNote.create(pitch=38, velocity=100, start_tick=ppqn, duration=ppqn // 4),  # Snare on beat 2
        MidiNote.create(pitch=36, velocity=100, start_tick=ppqn * 2, duration=ppqn // 4),  # Kick on beat 3
        MidiNote.create(pitch=38, velocity=100, start_tick=ppqn * 3, duration=ppqn // 4),  # Snare on beat 4
    ]

    # Create a MIDI clip and add it directly to the track
    test_clip = Clip.create_midi(
        name="Test Loop Clip",
        start_tick=0,
        duration=ticks_per_bar,  # 1 bar
        notes=notes,
        color="#4ade80"
    )

    # Get the track and add the clip
    track = engine.project_manager.get_track(track_id)
    if track:
        track.clips.append(test_clip)
        print(f"  Added clip to track: {test_clip.id}")
    else:
        print(f"  WARNING: Could not find track {track_id}")

    # Verify track and clip exist
    track = engine.project_manager.get_track(track_id)
    has_content = track is not None and len(track.clips) > 0

    results.append(("Content added to timeline",
                   has_content,
                   f"Track: {track_id}, Clips: {len(track.clips) if track else 0}"))

    print("\n[2/7] Configuring loop region...")
    # Define loop region: bar 2 to bar 4 (1920 to 5760 ticks)
    # This is a 2-bar loop region (bars 2-3)
    loop_start = ticks_per_bar  # Start at bar 2 (tick 1920)
    loop_end = ticks_per_bar * 3  # End at bar 4 (tick 5760)
    loop_length = loop_end - loop_start

    print(f"  Loop region: {loop_start} - {loop_end} ticks")
    print(f"  Loop length: {loop_length} ticks ({loop_length / ppqn} beats)")

    # Set the loop region (without enabling yet)
    engine.transport.set_loop(enabled=False, start_tick=loop_start, end_tick=loop_end)

    # Verify loop region was set
    loop = engine.transport._loop
    region_set = loop.start_tick == loop_start and loop.end_tick == loop_end

    results.append(("Loop region configured",
                   region_set,
                   f"Start: {loop.start_tick}, End: {loop.end_tick}"))

    print("\n[3/7] Enabling loop mode...")
    # Enable loop
    engine.transport.set_loop(enabled=True, start_tick=loop_start, end_tick=loop_end)

    loop_enabled = engine.transport._loop.enabled
    print(f"  Loop enabled: {loop_enabled}")

    results.append(("Loop mode enabled",
                   loop_enabled,
                   f"Loop enabled: {loop_enabled}"))

    print("\n[4/7] Seeking to start of loop region...")
    # Seek to start of loop region to ensure consistent starting position
    engine.transport.seek(loop_start)
    time.sleep(0.1)  # Give time for seek to complete

    start_position = engine.transport.current_tick
    print(f"  Starting position: {start_position} ticks")

    # Verify we're at or near the loop start
    at_loop_start = abs(start_position - loop_start) < ppqn  # Within 1 beat

    results.append(("Seeked to loop start",
                   at_loop_start,
                   f"Position: {start_position}, Target: {loop_start}"))

    print("\n[5/7] Starting playback and tracking position...")
    # Track position changes to verify looping
    position_history = []

    def track_position(tick):
        position_history.append({'tick': tick, 'time': time.time()})

    engine.transport.add_position_callback(track_position)

    # Start playback
    playback_started = engine.transport.play()
    print(f"  Playback started: {playback_started}")

    # Calculate how long to wait for the loop to occur
    # At 120 BPM, we have 480 ticks/beat = 960 ticks/second
    # Loop length is 2 bars = 8 beats = 3840 ticks
    # That's 4 seconds per loop
    # Wait 5 seconds to capture at least one full loop cycle

    ticks_per_second = (engine.transport.bpm / 60.0) * ppqn
    seconds_per_loop = loop_length / ticks_per_second
    wait_time = seconds_per_loop + 1.5  # Wait for loop + margin

    print(f"  At 120 BPM: {ticks_per_second:.1f} ticks/second")
    print(f"  Seconds per loop: {seconds_per_loop:.2f}s")
    print(f"  Waiting {wait_time:.1f}s for loop to occur...")

    time.sleep(wait_time)

    # Stop transport
    engine.transport.stop()
    print(f"  Transport stopped")

    # Analyze position history
    num_positions = len(position_history)
    print(f"  Position updates captured: {num_positions}")

    results.append(("Position tracking during playback",
                   num_positions > 0,
                   f"Captured {num_positions} position updates"))

    print("\n[6/7] Verifying loop behavior...")
    # Verify playback stayed within loop region and wrapped around
    loop_verified = False
    loop_wrap_detected = False
    positions_in_region = 0
    positions_outside = 0

    if num_positions > 0:
        # Check positions are within loop region
        for pos in position_history:
            tick = pos['tick']
            if loop_start <= tick <= loop_end:
                positions_in_region += 1
            else:
                positions_outside += 1

        # Check for loop wrap (position jump back to start)
        prev_tick = None
        for pos in position_history:
            tick = pos['tick']
            if prev_tick is not None:
                # Detect wrap: previous tick was near end, current tick is near start
                if prev_tick > (loop_end - ppqn * 2) and tick < (loop_start + ppqn * 2):
                    loop_wrap_detected = True
                    print(f"  Loop wrap detected: {prev_tick} -> {tick}")
                    break
            prev_tick = tick

        # Calculate percentage in region (allow some tolerance for timing)
        total_checked = positions_in_region + positions_outside
        if total_checked > 0:
            in_region_pct = (positions_in_region / total_checked) * 100
            print(f"  Positions in loop region: {positions_in_region}/{total_checked} ({in_region_pct:.1f}%)")

        # Loop is verified if most positions are in region or we detected a wrap
        loop_verified = (positions_in_region > positions_outside) or loop_wrap_detected

        # Show first and last few positions for debugging
        print(f"  First 5 positions: {[p['tick'] for p in position_history[:5]]}")
        print(f"  Last 5 positions: {[p['tick'] for p in position_history[-5:]]}")

    results.append(("Loop wrap detected",
                   loop_wrap_detected,
                   f"Wrap: {loop_wrap_detected}, In region: {positions_in_region}, Outside: {positions_outside}"))

    print("\n[7/7] Verifying loop constraints...")
    # Additional verification: test that transport reports correct loop status
    transport_status = engine.transport.get_status()
    status_loop_enabled = transport_status['loop_enabled']
    status_loop_start = transport_status['loop_start']
    status_loop_end = transport_status['loop_end']

    print(f"  Transport status loop_enabled: {status_loop_enabled}")
    print(f"  Transport status loop_start: {status_loop_start}")
    print(f"  Transport status loop_end: {status_loop_end}")

    status_matches = (status_loop_enabled and
                      status_loop_start == loop_start and
                      status_loop_end == loop_end)

    results.append(("Transport status reports loop correctly",
                   status_matches,
                   f"Enabled: {status_loop_enabled}, Start: {status_loop_start}, End: {status_loop_end}"))

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
    success = test_loop_region_playback()
    sys.exit(0 if success else 1)
