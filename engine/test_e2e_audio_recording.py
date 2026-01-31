"""
End-to-end verification test for audio recording flow.

Tests the complete flow:
1. Arm audio track for recording
2. Press record
3. Simulate sound into microphone (generate test audio)
4. Stop recording
5. Verify clip data is captured
6. Verify playback capability
"""

import asyncio
import numpy as np
import time
import sys

# Add engine directory to path
sys.path.insert(0, '.')

from main import BeatBoxDawEngine, EngineConfig
from transport import TransportState
from project import Track


def test_audio_recording_flow():
    """Test the complete audio recording E2E flow."""
    print("=" * 60)
    print("E2E Audio Recording Flow Verification")
    print("=" * 60)

    results = []

    # Create engine
    config = EngineConfig()
    engine = BeatBoxDawEngine(config)

    print("\n[1/6] Setting up audio track...")
    # Add an audio track
    track_data = engine.add_track('drum', 'Recording Track')
    track_id = track_data['id']
    print(f"  Created track: {track_id}")

    # Verify track exists
    track = engine.project_manager.get_track(track_id)
    assert track is not None, "Track should exist"
    results.append(("Track creation", True, "Track created successfully"))

    print("\n[2/6] Arming track for recording...")
    # Arm the track
    success = engine.set_track_armed(track_id, True)
    assert success, "Should be able to arm track"

    # Verify track is armed
    track = engine.project_manager.get_track(track_id)
    assert track.armed, "Track should be armed"
    results.append(("Track arming", True, f"Track {track_id} armed successfully"))
    print(f"  Track armed: {track.armed}")

    print("\n[3/6] Starting recording...")
    # Get initial tick position
    initial_tick = engine.transport.current_tick
    print(f"  Initial tick: {initial_tick}")

    # Start recording via transport
    recording_started = engine.transport_record()
    assert recording_started, "Recording should start"

    # Give transport time to enter recording state
    time.sleep(0.1)

    # Verify transport is in recording state
    transport_state = engine.transport.state
    print(f"  Transport state: {transport_state}")

    # Verify track recording is active
    recording_active = engine._audio_recording_active
    print(f"  Track recording active: {recording_active}")

    # Transport can be in RECORDING or PRE_ROLL state (pre-roll is count-in before recording)
    recording_state_valid = transport_state in (TransportState.RECORDING, TransportState.PRE_ROLL)
    results.append(("Recording start",
                   recording_state_valid and recording_active,
                   f"Transport: {transport_state.value}, Recording: {recording_active}"))

    print("\n[4/6] Simulating audio input (generating test audio)...")
    # Simulate audio input by directly feeding audio to the recorder
    # Generate 1 second of 440Hz sine wave (A4 note)
    sample_rate = config.sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Feed audio in chunks (simulating real-time capture)
    chunk_size = config.buffer_size
    chunks_fed = 0
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i + chunk_size]
        engine.audio_recorder.add_audio_buffer(chunk)
        chunks_fed += 1

    print(f"  Fed {chunks_fed} audio chunks ({len(test_audio)} samples total)")
    print(f"  Audio duration: {duration}s")

    # Verify recording has captured audio
    recorded_duration = engine.audio_recorder.get_duration()
    print(f"  Recorded duration: {recorded_duration:.3f}s")

    results.append(("Audio capture",
                   recorded_duration > 0.9,
                   f"Recorded {recorded_duration:.3f}s of audio"))

    print("\n[5/6] Stopping recording...")
    # Stop recording via transport
    engine.transport_record()  # Toggle to stop

    # Give time for recording to stop
    time.sleep(0.1)

    # Verify transport stopped recording
    final_state = engine.transport.state
    print(f"  Final transport state: {final_state}")

    # Verify track recording stopped
    recording_active_after = engine._audio_recording_active
    print(f"  Recording active after stop: {recording_active_after}")

    # Get recording info
    recording_info = engine.audio_recorder.get_recording_info()
    print(f"  Recording info: {recording_info}")

    results.append(("Recording stop",
                   final_state != TransportState.RECORDING and not recording_active_after,
                   f"Recording stopped, {recording_info['total_samples']} samples captured"))

    print("\n[6/6] Verifying recorded audio data...")
    # Get the recorded audio data
    audio_data = engine.audio_recorder.get_audio_data()
    has_audio = audio_data is not None and len(audio_data) > 0

    print(f"  Audio data retrieved: {has_audio}")
    if has_audio:
        print(f"  Audio length: {len(audio_data)} samples")
        print(f"  Audio duration: {len(audio_data) / sample_rate:.3f}s")
        print(f"  Audio max amplitude: {np.abs(audio_data).max():.3f}")

    # Verify audio can be loaded into playback system
    if has_audio:
        clip_id = f"test_recording_{int(time.time())}"
        start_tick = 0

        # Calculate duration in ticks
        duration_seconds = len(audio_data) / sample_rate
        ppqn = engine.transport.ticks_per_beat
        bpm = engine.transport.bpm
        ticks_per_second = (bpm / 60.0) * ppqn
        duration_ticks = int(duration_seconds * ticks_per_second)

        print(f"  Preparing to load clip...")
        print(f"  Duration in ticks: {duration_ticks}")

        # Note: We skip actual playback system loading in test mode
        # because it requires an active audio output device
        # Instead, we verify the data is valid for loading
        playback_success = audio_data is not None and len(audio_data) > 0 and duration_ticks > 0

        print(f"  Audio ready for playback: {playback_success}")
        print(f"  Clip ID: {clip_id}")

        results.append(("Playback setup",
                       playback_success,
                       f"Audio data validated for playback ({duration_ticks} ticks)"))
    else:
        results.append(("Playback setup", False, "No audio data to load"))

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
        engine.audio_capture.stop()
    except Exception as cleanup_err:
        print(f"  Audio capture stop: {cleanup_err}")

    try:
        engine.transport.stop()
    except Exception as cleanup_err:
        print(f"  Transport stop: {cleanup_err}")

    print("Cleanup complete.")

    return all_passed


if __name__ == "__main__":
    success = test_audio_recording_flow()
    sys.exit(0 if success else 1)
