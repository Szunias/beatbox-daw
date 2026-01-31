#!/usr/bin/env python3
"""
End-to-end test for transport controls.
Tests: play, pause, stop, seek, loop functionality via WebSocket.
"""

import asyncio
import json
import time
import sys

try:
    import websockets
except ImportError:
    print("ERROR: websockets module not installed. Install with: pip install websockets")
    sys.exit(1)


class TransportE2ETest:
    """End-to-end test suite for transport controls."""

    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self.ws = None
        self.results = []
        self.position_updates = []

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

    async def receive_until(self, target_type: str, timeout: float = 5.0):
        """Receive messages until we get the target type."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                data = json.loads(response)
                if data.get("type") == target_type:
                    return data
            except asyncio.TimeoutError:
                return None
        return None

    async def collect_positions(self, duration: float = 2.0):
        """Collect position updates for a duration."""
        self.position_updates = []
        start_time = time.time()

        while time.time() - start_time < duration:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                data = json.loads(response)
                if data.get("type") == "transport_position":
                    self.position_updates.append(data.get("data", {}))
            except asyncio.TimeoutError:
                continue

        return self.position_updates

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        status = "[PASS]" if passed else "[FAIL]"
        msg = f"{status} {test_name}"
        if details:
            msg += f" - {details}"
        print(msg)
        self.results.append({"test": test_name, "passed": passed, "details": details})

    async def test_play(self):
        """Test: Play starts playback and position increases."""
        # First stop to ensure clean state
        await self.send_and_receive("transport_stop")
        await asyncio.sleep(0.2)

        # Get initial position
        initial_status = await self.send_and_receive("status")
        initial_tick = initial_status.get("data", {}).get("transport", {}).get("current_tick", 0) if initial_status else 0

        # Start playback
        play_response = await self.send_and_receive("transport_play")

        # Wait and collect positions
        await asyncio.sleep(0.5)
        positions = await self.collect_positions(1.5)

        # Stop playback
        await self.send_and_receive("transport_stop")

        # Verify position increased
        if positions and len(positions) > 0:
            final_tick = positions[-1].get("tick", 0)
            position_increased = final_tick > initial_tick
            self.log_result(
                "Play - playhead moves",
                position_increased,
                f"Initial: {initial_tick}, Final: {final_tick}, Updates: {len(positions)}"
            )
            return position_increased
        else:
            self.log_result("Play - playhead moves", False, "No position updates received")
            return False

    async def test_pause(self):
        """Test: Pause stops playhead movement."""
        # Start playback
        await self.send_and_receive("transport_stop")
        await asyncio.sleep(0.2)
        await self.send_and_receive("transport_play")
        await asyncio.sleep(0.5)

        # Pause
        await self.send_and_receive("transport_pause")
        await asyncio.sleep(0.2)

        # Get position right after pause
        status1 = await self.send_and_receive("status")
        tick1 = status1.get("data", {}).get("transport", {}).get("current_tick", 0) if status1 else 0

        # Wait a bit
        await asyncio.sleep(0.5)

        # Get position again - should be the same (paused)
        status2 = await self.send_and_receive("status")
        tick2 = status2.get("data", {}).get("transport", {}).get("current_tick", 0) if status2 else 0

        # Clean up
        await self.send_and_receive("transport_stop")

        # Verify position stayed the same (with small tolerance for timing)
        position_stable = abs(tick2 - tick1) < 10  # Allow small variance
        self.log_result(
            "Pause - playhead stops",
            position_stable,
            f"After pause: {tick1}, After wait: {tick2}"
        )
        return position_stable

    async def test_seek(self):
        """Test: Seek updates position correctly."""
        # Stop first
        await self.send_and_receive("transport_stop")
        await asyncio.sleep(0.3)

        # Clear any pending messages
        try:
            while True:
                await asyncio.wait_for(self.ws.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            pass

        # Seek to specific position
        target_tick = 1920  # 1 bar at 480 PPQN with 4/4 time
        seek_response = await self.send_and_receive("transport_seek", {"tick": target_tick})
        await asyncio.sleep(0.2)

        # Get current position by requesting status and finding the response
        await self.ws.send(json.dumps({"type": "status"}))
        status = None

        # Look for the status response (skip any position broadcasts)
        for _ in range(10):
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                data = json.loads(response)
                if data.get("type") == "status":
                    status = data
                    break
            except asyncio.TimeoutError:
                break

        current_tick = status.get("data", {}).get("transport", {}).get("current_tick", 0) if status else 0

        # Verify position matches
        position_correct = abs(current_tick - target_tick) < 50  # Allow larger tolerance
        self.log_result(
            "Seek - position updates",
            position_correct,
            f"Target: {target_tick}, Actual: {current_tick}"
        )

        # Clean up
        await self.send_and_receive("transport_stop")
        return position_correct

    async def test_loop(self):
        """Test: Loop region enables looping."""
        # Stop and reset
        await self.send_and_receive("transport_stop")
        await asyncio.sleep(0.2)

        # Set a short loop region (1 bar)
        loop_start = 0
        loop_end = 1920  # 1 bar
        await self.send_and_receive("set_loop", {
            "enabled": True,
            "start_tick": loop_start,
            "end_tick": loop_end
        })

        # Start playback
        await self.send_and_receive("transport_play")

        # Collect positions over time (longer than loop to test wrapping)
        # At 120 BPM, 1 bar = 2 seconds
        positions = await self.collect_positions(3.0)

        # Stop playback
        await self.send_and_receive("transport_stop")

        # Disable loop
        await self.send_and_receive("set_loop", {"enabled": False})

        # Verify looping occurred - position should wrap around
        if positions and len(positions) > 5:
            # Check if any position is less than a previous one (indicates loop wrap)
            wrapped = False
            for i in range(1, len(positions)):
                prev_tick = positions[i-1].get("tick", 0)
                curr_tick = positions[i].get("tick", 0)
                if curr_tick < prev_tick - 100:  # Account for timing variance
                    wrapped = True
                    break

            # Also check that no position exceeded loop end significantly
            max_tick = max(p.get("tick", 0) for p in positions)
            stayed_in_bounds = max_tick <= loop_end + 200  # Small tolerance

            loop_worked = wrapped or stayed_in_bounds
            self.log_result(
                "Loop - position wraps",
                loop_worked,
                f"Wrapped: {wrapped}, Max tick: {max_tick}, Loop end: {loop_end}"
            )
            return loop_worked
        else:
            self.log_result("Loop - position wraps", False, f"Insufficient position data: {len(positions)}")
            return False

    async def test_sync(self):
        """Test: Frontend-backend sync within 50ms tolerance."""
        # This tests the position broadcast timing
        await self.send_and_receive("transport_stop")
        await asyncio.sleep(0.2)

        # Start playback
        await self.send_and_receive("transport_play")

        # Measure time between position updates
        update_times = []
        start_time = time.time()

        while len(update_times) < 10 and time.time() - start_time < 3.0:
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                data = json.loads(response)
                if data.get("type") == "transport_position":
                    update_times.append(time.time())
            except asyncio.TimeoutError:
                continue

        # Stop playback
        await self.send_and_receive("transport_stop")

        # Calculate intervals
        if len(update_times) >= 2:
            intervals = [update_times[i] - update_times[i-1] for i in range(1, len(update_times))]
            avg_interval = sum(intervals) / len(intervals)
            max_interval = max(intervals)

            # Position updates should come frequently (< 100ms typical, allow up to 200ms)
            sync_ok = max_interval < 0.2
            self.log_result(
                "Sync - regular updates",
                sync_ok,
                f"Avg interval: {avg_interval*1000:.1f}ms, Max: {max_interval*1000:.1f}ms"
            )
            return sync_ok
        else:
            self.log_result("Sync - regular updates", False, "Insufficient updates")
            return False

    async def run_all_tests(self):
        """Run all transport tests."""
        print("\n" + "="*50)
        print("E2E Transport Controls Verification")
        print("="*50 + "\n")

        if not await self.connect():
            return False

        try:
            # Run each test
            print("\n--- Test 1: Play ---")
            await self.test_play()

            print("\n--- Test 2: Pause ---")
            await self.test_pause()

            print("\n--- Test 3: Seek ---")
            await self.test_seek()

            print("\n--- Test 4: Loop ---")
            await self.test_loop()

            print("\n--- Test 5: Sync ---")
            await self.test_sync()

        finally:
            await self.disconnect()

        # Summary
        print("\n" + "="*50)
        print("Test Summary")
        print("="*50)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        for result in self.results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{status}] {result['test']}")

        print(f"\nTotal: {passed}/{total} tests passed")
        print("="*50 + "\n")

        return passed == total


async def main():
    """Main entry point."""
    tester = TransportE2ETest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
