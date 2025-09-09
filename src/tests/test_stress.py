"""Stress tests to verify system stability under resource pressure.

These tests verify that starlings behaves safely under various stress conditions
without overwhelming the system or causing crashes.
"""

import concurrent.futures
import gc
import logging
import os
import time

import psutil
import pytest
import starlings as sl

try:
    from .test_safety import MemoryBalloon
except ImportError:
    from test_safety import MemoryBalloon


@pytest.mark.stress
class TestStressConditions:
    """Test system behaviour under various stress conditions."""

    def setup_method(self):
        """Set conservative safety for stress tests."""
        os.environ["STARLINGS_SAFETY_LEVEL"] = "conservative"
        gc.collect()

    def teardown_method(self):
        """Clean up after stress tests."""
        os.environ.pop("STARLINGS_SAFETY_LEVEL", None)
        gc.collect()

    def test_rapid_successive_operations(self):
        """Test rapid successive operations don't overwhelm system."""
        operations_count = 0
        start_time = time.time()
        max_duration = 10  # seconds

        try:
            while time.time() - start_time < max_duration:
                edges = [(i, i + 1, 0.5) for i in range(1000)]
                try:
                    collection = sl.Collection.from_edges(edges)
                    if collection:
                        operations_count += 1
                except (MemoryError, RuntimeError):
                    # Expected when system under pressure
                    break

                # Small delay to prevent tight loop
                time.sleep(0.01)

                # Check system responsiveness
                if operations_count % 10 == 0:
                    cpu_percent = psutil.cpu_percent(interval=0.01)
                    if cpu_percent > 95:
                        logging.warning(f"CPU too high ({cpu_percent}%), backing off")
                        time.sleep(0.1)

        except KeyboardInterrupt:
            pytest.fail("Test interrupted - system may be unresponsive")

        end_time = time.time()

        # Should have completed at least some operations
        assert operations_count > 5, f"Too few operations completed: {operations_count}"

        # System should remain responsive
        assert end_time - start_time < max_duration + 2, "System became unresponsive"

        logging.info(
            f"Completed {operations_count} operations in {end_time - start_time:.2f}s"
        )

    def test_concurrent_operations_with_memory_pressure(self):
        """Test concurrent operations under artificial memory pressure."""
        balloon = MemoryBalloon()

        try:
            # Create moderate memory pressure
            balloon.inflate_to_percent(65, max_chunks=15)

            def worker_operation(worker_id: int) -> dict:
                """Worker function that performs operations."""
                results = {"worker_id": worker_id, "success": 0, "failed": 0}

                for _i in range(5):  # 5 operations per worker
                    try:
                        start_id = worker_id * 1000
                        end_id = (worker_id + 1) * 1000
                        edges = [(j, j + 1, 0.5) for j in range(start_id, end_id)]
                        collection = sl.Collection.from_edges(edges)
                        if collection:
                            results["success"] += 1
                    except (MemoryError, RuntimeError):
                        results["failed"] += 1

                    # Small delay between operations
                    time.sleep(0.05)

                return results

            # Run concurrent workers
            num_workers = 4
            start_time = time.time()

            executor_kwargs = {"max_workers": num_workers}
            with concurrent.futures.ThreadPoolExecutor(**executor_kwargs) as executor:
                futures = [
                    executor.submit(worker_operation, i) for i in range(num_workers)
                ]

                # Wait for completion with timeout
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        pytest.fail("Workers timed out - system unresponsive")

            end_time = time.time()

            # Analyse results
            total_success = sum(r["success"] for r in results)
            total_failed = sum(r["failed"] for r in results)
            total_ops = total_success + total_failed

            assert len(results) == num_workers, "Not all workers completed"
            assert total_ops > 0, "No operations attempted"
            assert end_time - start_time < 35, "Operations took too long"

            # At least some operations should succeed even under pressure
            success_rate = total_success / total_ops if total_ops > 0 else 0
            assert success_rate > 0.2, f"Success rate too low: {success_rate:.2%}"

            logging.info(
                f"Concurrent stress test: {total_success}/{total_ops} operations "
                f"succeeded ({success_rate:.1%}) in {end_time - start_time:.2f}s"
            )

        finally:
            balloon.deflate()

    def test_memory_exhaustion_recovery(self):
        """Test system recovery after memory exhaustion."""
        balloon = MemoryBalloon()

        try:
            # Phase 1: Create high memory pressure
            balloon.inflate_to_percent(80, max_chunks=25)

            # Operations should fail gracefully under pressure
            failed_operations = 0
            for _i in range(5):
                try:
                    edges = [(j, j + 1, 0.5) for j in range(1000)]
                    collection = sl.Collection.from_edges(edges)
                    if collection is None:
                        failed_operations += 1
                except (MemoryError, RuntimeError):
                    failed_operations += 1

            fail_msg = "Expected some operations to fail under pressure"
            assert failed_operations > 0, fail_msg

            # Phase 2: Release memory and verify recovery
            balloon.deflate()
            time.sleep(0.5)  # Allow system to recover

            # Operations should work again after recovery
            recovery_success = 0
            for _i in range(3):
                try:
                    edges = [(j, j + 1, 0.5) for j in range(1000)]
                    collection = sl.Collection.from_edges(edges)
                    if collection:
                        recovery_success += 1
                except (MemoryError, RuntimeError):
                    pass

                time.sleep(0.1)

            recovery_msg = (
                f"System did not recover properly: {recovery_success}/3 "
                f"operations succeeded"
            )
            assert recovery_success >= 2, recovery_msg

            logging.info(
                f"Recovery test: {failed_operations} ops failed under pressure, "
                f"{recovery_success} ops succeeded after recovery"
            )

        finally:
            balloon.deflate()

    def test_gradual_memory_increase(self):
        """Test system behaviour as memory usage gradually increases."""
        balloon = MemoryBalloon()

        try:
            operation_times = []
            memory_levels = [40, 50, 60, 70, 75]  # Gradual increase

            for target_percent in memory_levels:
                balloon.deflate()
                balloon.inflate_to_percent(target_percent, max_chunks=20)

                current_percent = balloon.get_current_memory_percent()
                if current_percent < target_percent - 5:
                    skip_msg = (
                        f"Skipping {target_percent}% test - only achieved "
                        f"{current_percent}%"
                    )
                    logging.info(skip_msg)
                    continue

                # Time a standard operation
                start = time.perf_counter()
                try:
                    edges = [(i, i + 1, 0.5) for i in range(2000)]
                    collection = sl.Collection.from_edges(edges)
                    elapsed = time.perf_counter() - start
                    success = collection is not None
                except (MemoryError, RuntimeError):
                    elapsed = time.perf_counter() - start
                    success = False

                operation_times.append({
                    "memory_percent": current_percent,
                    "elapsed": elapsed,
                    "success": success,
                })

                time.sleep(0.1)  # Brief pause between tests

            # Analyse progression
            successful_ops = [op for op in operation_times if op["success"]]

            if len(successful_ops) >= 2:
                # Operations should generally get slower with higher memory pressure
                # (though we allow some variation due to system noise)
                times_by_memory = sorted(
                    successful_ops, key=lambda x: x["memory_percent"]
                )

                logging.info("Memory progression results:")
                for op in times_by_memory:
                    status = "✓" if op["success"] else "✗"
                    logging.info(
                        f"  {op['memory_percent']:.1f}% memory: "
                        f"{op['elapsed']:.3f}s {status}"
                    )

                # Should have at least some successful operations
                fail_msg = "Too many operations failed"
                assert len(successful_ops) >= len(memory_levels) // 2, fail_msg

        finally:
            balloon.deflate()

    @pytest.mark.slow
    def test_sustained_operation_under_pressure(self):
        """Test sustained operations under moderate memory pressure."""
        balloon = MemoryBalloon()

        try:
            # Create moderate sustained pressure
            balloon.inflate_to_percent(60, max_chunks=15)

            operations_completed = 0
            operations_failed = 0
            start_time = time.time()
            test_duration = 20  # seconds

            while time.time() - start_time < test_duration:
                try:
                    edges = [(i, i + 1, 0.5) for i in range(1500)]
                    collection = sl.Collection.from_edges(edges)
                    if collection:
                        operations_completed += 1
                    else:
                        operations_failed += 1
                except (MemoryError, RuntimeError):
                    operations_failed += 1

                # Regular delay to prevent system overwhelm
                time.sleep(0.2)

                # Check system health periodically
                if (operations_completed + operations_failed) % 10 == 0:
                    cpu_percent = psutil.cpu_percent(interval=0.01)

                    if cpu_percent > 98:
                        logging.warning(
                            f"CPU too high ({cpu_percent}%), extending delay"
                        )
                        time.sleep(0.5)

            end_time = time.time()
            total_operations = operations_completed + operations_failed

            # Should complete reasonable number of operations
            assert total_operations > 10, f"Too few operations: {total_operations}"

            # Should maintain reasonable success rate under sustained pressure
            success_rate = (
                operations_completed / total_operations if total_operations > 0 else 0
            )
            assert success_rate > 0.3, f"Success rate too low: {success_rate:.2%}"

            # Should not take too long (system responsive)
            assert end_time - start_time < test_duration + 5, "Test took too long"

            logging.info(
                f"Sustained pressure test: {operations_completed}/{total_operations} "
                f"operations succeeded ({success_rate:.1%}) over "
                f"{end_time - start_time:.1f}s"
            )

        finally:
            balloon.deflate()


@pytest.mark.integration
class TestSafetyIntegration:
    """Integration tests for safety mechanisms."""

    def test_safety_mechanisms_work_together(self):
        """Test that all safety mechanisms work together correctly."""
        # Test progression from normal -> throttled -> circuit breaker
        balloon = MemoryBalloon()

        try:
            # Phase 1: Normal operation
            edges = [(i, i + 1, 0.5) for i in range(1000)]
            collection = sl.Collection.from_edges(edges)
            assert collection is not None
            logging.info("✓ Normal operation works")

            # Phase 2: Light pressure - should throttle but work
            balloon.inflate_to_percent(55, max_chunks=10)

            start = time.perf_counter()
            edges = [(i, i + 1, 0.5) for i in range(1000)]
            collection = sl.Collection.from_edges(edges)
            throttled_time = time.perf_counter() - start

            # Should still work but may be slower
            if collection is not None:
                logging.info(f"✓ Throttled operation works ({throttled_time:.3f}s)")

            # Phase 3: High pressure - may trip circuit breaker
            balloon.inflate_to_percent(75, max_chunks=20)

            high_pressure_success = False
            try:
                edges = [(i, i + 1, 0.5) for i in range(1000)]
                collection = sl.Collection.from_edges(edges)
                high_pressure_success = collection is not None
            except (MemoryError, RuntimeError) as e:
                logging.info(f"✓ Circuit breaker activated: {e}")

            # Either should work (throttled) or fail gracefully (circuit breaker)
            if high_pressure_success:
                logging.info("✓ High pressure operation succeeded with throttling")
            else:
                logging.info("✓ High pressure operation blocked by circuit breaker")

            # Phase 4: Recovery
            balloon.deflate()
            time.sleep(0.5)

            edges = [(i, i + 1, 0.5) for i in range(1000)]
            collection = sl.Collection.from_edges(edges)
            assert collection is not None
            logging.info("✓ Recovery after pressure release works")

        finally:
            balloon.deflate()

    def test_different_operation_sizes_handled_appropriately(self):
        """Test that different operation sizes are handled with appropriate safety."""
        # Small operations should always work
        small_edges = [(i, i + 1, 0.5) for i in range(100)]
        small_collection = sl.Collection.from_edges(small_edges)
        assert small_collection is not None

        # Medium operations should work under normal conditions
        medium_edges = [(i, i + 1, 0.5) for i in range(5000)]
        medium_collection = sl.Collection.from_edges(medium_edges)
        assert medium_collection is not None

        # Very large operations should be rejected based on available memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        # Only test large operations if we have substantial memory
        if available_gb > 4:
            # Calculate an operation that would use significant memory
            # Scale with available memory
            large_entities = min(100000, int(available_gb * 50000))

            try:
                large_edges = sl.generate_entity_resolution_edges(large_entities)
                large_collection = sl.Collection.from_edges(large_edges)

                # If it succeeds, it should work correctly
                if large_collection:
                    logging.info(
                        f"✓ Large operation ({large_entities} entities) succeeded"
                    )

            except (MemoryError, RuntimeError, ValueError) as e:
                # If it fails, should be due to safety mechanisms
                error_msg = str(e).lower()
                expected_words = ["safety", "limit", "memory", "too large"]
                assert any(word in error_msg for word in expected_words)
                logging.info(
                    f"✓ Large operation appropriately rejected: {type(e).__name__}"
                )

        else:
            logging.info("Skipping large operation test due to limited memory")
