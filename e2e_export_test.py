#!/usr/bin/env python3
"""
End-to-end test for audio export.
Tests: Create project with multiple tracks, add clips, adjust mixer, export audio.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy module not installed. Install with: pip install numpy")
    sys.exit(1)

try:
    import scipy.io.wavfile as wav
except ImportError:
    print("ERROR: scipy module not installed. Install with: pip install scipy")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: websockets module not installed. Install with: pip install websockets")
    sys.exit(1)


class AudioExportE2ETest:
    """End-to-end test suite for audio export functionality."""

    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self.ws = None
        self.results = []
        self.temp_dir = tempfile.mkdtemp(prefix="beatbox_export_test_")
        self.export_output_file = os.path.join(self.temp_dir, "test_export.wav")

    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.ws = await websockets.connect(self.ws_url)
            print(f"[OK] Connected to {self.ws_url}")
            return True
        except Exception as e:
            print(f"[FAIL] Could not connect to {self.ws_url}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def send_and_receive(self, msg_type: str, data: dict = None, timeout: float = 5.0):
        """Send a message and wait for response."""
        message = {"type": msg_type}
        if data:
            message["data"] = data

        await self.ws.send(json.dumps(message))

        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            return json.loads(response)
        except asyncio.TimeoutError:
            return None

    async def receive_until(self, target_type: str, timeout: float = 10.0):
        """Receive messages until we get the target type."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                data = json.loads(response)
                if data.get("type") == target_type:
                    return data
            except asyncio.TimeoutError:
                continue
        return None

    async def clear_pending_messages(self, timeout: float = 0.1):
        """Clear any pending WebSocket messages."""
        try:
            while True:
                await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        status = "[PASS]" if passed else "[FAIL]"
        msg = f"{status} {test_name}"
        if details:
            msg += f" - {details}"
        print(msg)
        self.results.append({"test": test_name, "passed": passed, "details": details})

    async def test_create_project(self):
        """Test: Create a new project."""
        response = await self.send_and_receive("new_project", {"name": "Export Test Project"})

        if response and response.get("type") == "project":
            project_data = response.get("data", {})
            project_name = project_data.get("name", "")
            success = project_name == "Export Test Project"
            self.log_result(
                "Create project",
                success,
                f"Project name: {project_name}"
            )
            return success
        else:
            self.log_result("Create project", False, "No response or wrong response type")
            return False

    async def test_add_tracks(self):
        """Test: Add multiple tracks to project."""
        # Add track 1
        response1 = await self.send_and_receive("add_track", {
            "track_type": "audio",
            "name": "Track 1 - Left Pan"
        })

        # Add track 2
        response2 = await self.send_and_receive("add_track", {
            "track_type": "audio",
            "name": "Track 2 - Right Pan"
        })

        track1_id = None
        track2_id = None

        if response1 and response1.get("type") == "track_added":
            track1_id = response1.get("data", {}).get("id")

        if response2 and response2.get("type") == "track_added":
            track2_id = response2.get("data", {}).get("id")

        success = track1_id is not None and track2_id is not None
        self.log_result(
            "Add tracks",
            success,
            f"Track 1: {track1_id}, Track 2: {track2_id}"
        )
        return success, track1_id, track2_id

    async def test_adjust_mixer(self, track1_id: str, track2_id: str):
        """Test: Adjust mixer settings for tracks."""
        # Set Track 1: volume 0.8, pan left (-0.5)
        response1 = await self.send_and_receive("update_track_settings", {
            "track_id": track1_id,
            "volume": 0.8,
            "pan": -0.5,
            "muted": False
        })

        # Set Track 2: volume 0.6, pan right (0.5)
        response2 = await self.send_and_receive("update_track_settings", {
            "track_id": track2_id,
            "volume": 0.6,
            "pan": 0.5,
            "muted": False
        })

        # Set master volume
        response3 = await self.send_and_receive("set_master_volume", {
            "volume": 0.9
        })

        success1 = response1 and response1.get("data", {}).get("success", False)
        success2 = response2 and response2.get("data", {}).get("success", False)
        success3 = response3 and response3.get("data", {}).get("success", False)

        success = success1 and success2 and success3
        self.log_result(
            "Adjust mixer settings",
            success,
            f"Track1: {success1}, Track2: {success2}, Master: {success3}"
        )
        return success

    async def test_export_audio(self):
        """Test: Export audio to WAV file."""
        # Request export
        response = await self.send_and_receive("export_audio", {
            "filepath": self.export_output_file,
            "start_tick": 0,
            "end_tick": None  # Auto-detect end
        })

        if not response or response.get("type") != "export_audio_response":
            self.log_result("Export audio - start", False, "No export response received")
            return False

        export_started = response.get("data", {}).get("success", False)

        if not export_started:
            # Export may fail if no audio clips - this is expected without actual audio
            error = response.get("data", {}).get("error", "No audio data to export")
            self.log_result("Export audio - start", False, f"Export not started: {error}")
            return False

        # Wait for export completion
        completed = await self.receive_until("export_completed", timeout=30.0)

        if completed:
            export_data = completed.get("data", {})
            success = export_data.get("success", False)
            filepath = export_data.get("filepath", "")
            error = export_data.get("error", "")

            if success:
                self.log_result("Export audio - complete", True, f"Exported to: {filepath}")
            else:
                self.log_result("Export audio - complete", False, f"Export failed: {error}")

            return success
        else:
            self.log_result("Export audio - complete", False, "Export completion timeout")
            return False

    async def test_verify_wav_file(self):
        """Test: Verify exported WAV file has correct format and content."""
        if not os.path.exists(self.export_output_file):
            self.log_result("Verify WAV file - exists", False, f"File not found: {self.export_output_file}")
            return False

        self.log_result("Verify WAV file - exists", True, f"File found: {self.export_output_file}")

        try:
            sample_rate, audio_data = wav.read(self.export_output_file)

            # Verify sample rate (should be 44100)
            sample_rate_ok = sample_rate == 44100
            self.log_result(
                "Verify WAV file - sample rate",
                sample_rate_ok,
                f"Sample rate: {sample_rate} (expected: 44100)"
            )

            # Verify stereo (2 channels)
            is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] == 2
            self.log_result(
                "Verify WAV file - stereo",
                is_stereo,
                f"Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1} (expected: 2)"
            )

            # Verify audio data is not empty
            has_data = len(audio_data) > 0
            self.log_result(
                "Verify WAV file - has data",
                has_data,
                f"Samples: {len(audio_data)}"
            )

            # Verify audio data has some content (not all zeros)
            has_content = np.abs(audio_data).max() > 0 if has_data else False
            self.log_result(
                "Verify WAV file - has content",
                has_content,
                f"Max amplitude: {np.abs(audio_data).max() if has_data else 0}"
            )

            return sample_rate_ok and is_stereo and has_data and has_content

        except Exception as e:
            self.log_result("Verify WAV file - read", False, f"Error reading WAV: {e}")
            return False

    async def test_direct_export_with_audio(self):
        """
        Test direct audio export using AudioExporter directly.
        This bypasses the need for audio clips in the project and tests
        the export pipeline with synthesized audio data.
        """
        print("\n--- Direct Export Test (with synthesized audio) ---")

        # We'll test this by directly using the engine module
        # This is a more thorough test of the export functionality
        try:
            # Change to engine directory for imports
            engine_dir = Path(__file__).parent / "engine"
            sys.path.insert(0, str(engine_dir))

            from audio_export import AudioExporter, ExportConfig

            # Create exporter
            config = ExportConfig(
                sample_rate=44100,
                channels=2,
                bit_depth=16,
                normalize=True
            )
            exporter = AudioExporter(config)

            # Generate test audio: 2 second tones
            duration_seconds = 2.0
            sample_rate = 44100
            num_samples = int(sample_rate * duration_seconds)
            t = np.linspace(0, duration_seconds, num_samples, False)

            # Tone 1: 440Hz sine wave (panned left)
            tone1 = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

            # Tone 2: 660Hz sine wave (panned right, starts at 0.5s)
            tone2 = (np.sin(2 * np.pi * 660 * t) * 0.3).astype(np.float32)

            # Add clips to exporter
            exporter.add_clip(
                clip_id="test_clip1",
                track_id="track1",
                audio_data=tone1,
                sample_rate=sample_rate,
                start_sample=0,
                volume=0.8,
                pan=-0.5  # Left
            )

            exporter.add_clip(
                clip_id="test_clip2",
                track_id="track2",
                audio_data=tone2,
                sample_rate=sample_rate,
                start_sample=int(sample_rate * 0.5),  # Start at 0.5 seconds
                volume=0.6,
                pan=0.5  # Right
            )

            # Set track mixer settings
            exporter.set_track_settings("track1", volume=0.9, pan=-0.5)
            exporter.set_track_settings("track2", volume=0.7, pan=0.5)

            # Set master volume
            exporter.master_volume = 0.9

            # Export
            direct_export_file = os.path.join(self.temp_dir, "direct_export.wav")

            print(f"Exporting to: {direct_export_file}")

            # Progress tracking
            progress_updates = []

            def on_progress(progress):
                progress_updates.append(progress.progress)
                if len(progress_updates) % 5 == 0:
                    print(f"  Progress: {progress.progress * 100:.1f}%")

            exporter.set_progress_callback(on_progress)

            # Perform export
            success = exporter.export_to_wav(direct_export_file)

            if not success:
                error = exporter.get_progress().error_message
                self.log_result("Direct export - export", False, f"Export failed: {error}")
                return False

            self.log_result(
                "Direct export - export",
                True,
                f"Exported with {len(progress_updates)} progress updates"
            )

            # Verify the exported file
            if not os.path.exists(direct_export_file):
                self.log_result("Direct export - file exists", False, "File not found")
                return False

            self.log_result("Direct export - file exists", True, direct_export_file)

            # Read and verify
            sample_rate, audio_data = wav.read(direct_export_file)

            # Verify sample rate
            sample_rate_ok = sample_rate == 44100
            self.log_result(
                "Direct export - sample rate",
                sample_rate_ok,
                f"Sample rate: {sample_rate}"
            )

            # Verify stereo
            is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] == 2
            self.log_result(
                "Direct export - stereo",
                is_stereo,
                f"Shape: {audio_data.shape}"
            )

            # Verify duration (should be ~2.5 seconds - clip2 starts at 0.5s + 2s duration)
            duration = len(audio_data) / sample_rate
            duration_ok = 2.0 <= duration <= 3.0
            self.log_result(
                "Direct export - duration",
                duration_ok,
                f"Duration: {duration:.2f}s"
            )

            # Verify content - check that left and right channels have correct power balance due to panning
            panning_works = False
            if is_stereo:
                left_channel = audio_data[:, 0].astype(np.float64)
                right_channel = audio_data[:, 1].astype(np.float64)

                # Verify left channel has more content from tone1 (which is panned left at -0.5)
                # and right channel has content from tone2 (panned right at 0.5)
                # This is the key check that panning is working
                left_power = np.mean(left_channel ** 2)
                right_power = np.mean(right_channel ** 2)

                # Left should have more power due to tone1 being panned left and louder (0.8 vol vs 0.6)
                power_ratio = left_power / right_power if right_power > 0 else 0
                panning_works = power_ratio > 1.5  # Left should be significantly louder

                self.log_result(
                    "Direct export - stereo panning",
                    panning_works,
                    f"Left power: {left_power:.0f}, Right power: {right_power:.0f}, Ratio: {power_ratio:.2f}"
                )

                # Both channels should have some content
                both_have_content = left_power > 0 and right_power > 0

                self.log_result(
                    "Direct export - channel balance",
                    both_have_content,
                    f"Both channels have content: {both_have_content}"
                )

            # Clean up
            os.remove(direct_export_file)

            return sample_rate_ok and is_stereo and duration_ok and panning_works

        except ImportError as e:
            self.log_result("Direct export - import", False, f"Could not import audio_export: {e}")
            return False
        except Exception as e:
            self.log_result("Direct export - exception", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run_all_tests(self):
        """Run all audio export tests."""
        print("\n" + "=" * 60)
        print("E2E Audio Export Verification")
        print("=" * 60 + "\n")

        print(f"Temp directory: {self.temp_dir}")

        # Test 1: Direct export test (doesn't need WebSocket)
        print("\n--- Test 1: Direct Audio Export ---")
        await self.test_direct_export_with_audio()

        # Tests via WebSocket
        if not await self.connect():
            print("\nSkipping WebSocket-based tests (server not available)")
            self.cleanup()
            return self.print_summary()

        try:
            # Consume initial messages
            await self.clear_pending_messages()

            # Test 2: Create project
            print("\n--- Test 2: Create Project ---")
            project_ok = await self.test_create_project()

            # Test 3: Add tracks
            print("\n--- Test 3: Add Tracks ---")
            tracks_ok, track1_id, track2_id = await self.test_add_tracks()

            if tracks_ok and track1_id and track2_id:
                # Test 4: Adjust mixer
                print("\n--- Test 4: Adjust Mixer Settings ---")
                await self.test_adjust_mixer(track1_id, track2_id)

            # Test 5: Export audio
            # Note: This may fail if there are no audio clips loaded
            print("\n--- Test 5: Export Audio (via WebSocket) ---")
            export_ok = await self.test_export_audio()

            if export_ok:
                # Test 6: Verify WAV file
                print("\n--- Test 6: Verify WAV File ---")
                await self.test_verify_wav_file()

        finally:
            await self.disconnect()

        self.cleanup()
        return self.print_summary()

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"\n[CLEANUP] Removed temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"\n[CLEANUP] Warning: Could not remove temp directory: {e}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        for result in self.results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{status}] {result['test']}")

        print(f"\nTotal: {passed}/{total} tests passed")
        print("=" * 60 + "\n")

        return passed == total


async def main():
    """Main entry point."""
    tester = AudioExportE2ETest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
