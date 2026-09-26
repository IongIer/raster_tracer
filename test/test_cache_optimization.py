"""Cache prefetch preserves numerical work, allocation bounds and snap ties."""

import tempfile
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from osgeo import gdal

from .. import pointtool_raster as raster
from ..exceptions import InvalidRasterError, ResourceLimitError, TraceCancelled
from ..pointtool_session import RasterSourceSpec, TraceResult, WorkerRequest
from ..pointtool_tasks import FindPathTask, execute_request
from .tracing_fixture import ControlledTask, TraceFixture
from .utilities import get_qgis_app, write_raster


class CacheOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        alignment = patch.object(raster, "CACHE_ALIGNMENT", 32)
        alignment.start()
        self.addCleanup(alignment.stop)

    def source(self, values, nodata=None):
        path = Path(self.directory.name) / "cache.tif"
        height, width = values.shape
        write_raster(path, (values,) * 3, gdal.GDT_Float64, nodata=nodata)
        return RasterSourceSpec(str(path), height, width, 1)

    def work(self, source, bounds=(33, 53, 35, 58), color=(0, 0, 0)):
        top, bottom, left, right = bounds
        return WorkerRequest(
            1,
            source,
            bounds,
            (top + 2, left + 2),
            (bottom - 3, right - 3),
            color,
            False,
        )

    def exact(self, work):
        with patch.object(raster, "cache_bounds", return_value=work.bounds):
            return execute_request(work)

    def test_prefetch_and_warm_views_preserve_exact_paths_and_boundaries(self):
        values = np.full((96, 128), 10.0)
        values[32] = 0  # A tempting detour just outside the requested graph.
        source = self.source(values)
        work = self.work(source)
        cold = execute_request(work)
        self.assertEqual(cold.snapshot.bounds, (32, 64, 32, 64))
        exact = self.exact(work)
        self.assertEqual(
            (cold.status, cold.cost, cold.path), (exact.status, exact.cost, exact.path)
        )
        nearby = replace(work, bounds=(35, 55, 37, 60))
        with patch.object(
            raster, "read_rgb", side_effect=AssertionError("unexpected read")
        ):
            snapshot, graph, valid, stats = raster.prepare_snapshot(
                nearby, cold.snapshot, lambda: False
            )
            warm = execute_request(nearby, snapshot)
        self.assertIs(snapshot, cold.snapshot)
        self.assertEqual(graph.shape, (20, 23))
        self.assertEqual(valid.shape, graph.shape)
        self.assertTrue(np.shares_memory(graph, snapshot.cost))
        self.assertTrue(stats["rgb_cache_hit"] and stats["cost_cache_hit"])
        exact = self.exact(nearby)
        self.assertEqual(
            (warm.cost, warm.path, warm.origin), (exact.cost, exact.path, exact.origin)
        )
        outside = self.work(source, (50, 70, 40, 63))
        with patch.object(raster, "read_rgb", wraps=raster.read_rgb) as read:
            moved = execute_request(outside, snapshot)
        self.assertEqual(read.call_count, 1)
        exact = self.exact(outside)
        self.assertEqual((moved.cost, moved.path), (exact.cost, exact.path))

    def test_failed_prefetch_read_retries_only_the_requested_window(self):
        source = self.source(np.ones((96, 128)))
        work = self.work(source)
        read_rgb = raster.read_rgb

        for error_type in (InvalidRasterError, RuntimeError):
            # GDAL uses RuntimeError when its exception mode is enabled.
            def read(dataset, source, bounds, cancel):
                if bounds != work.bounds:
                    raise error_type("Damaged tile outside the requested graph")
                return read_rgb(dataset, source, bounds, cancel)

            with (
                self.subTest(error_type=error_type),
                patch.object(raster, "read_rgb", side_effect=read) as reader,
            ):
                result = execute_request(work)
            self.assertEqual(result.status, "success")
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(result.snapshot.bounds, work.bounds)

    def test_color_changes_reuse_rgb_and_recompute_exact_costs(self):
        source = self.source(np.arange(96 * 128, dtype=float).reshape(96, 128) % 251)
        work = self.work(source)
        first, cost_a, _, _ = raster.prepare_snapshot(work, None, lambda: False)
        with patch.object(
            raster, "read_rgb", side_effect=AssertionError("unexpected read")
        ):
            second, cost_b, _, stats_b = raster.prepare_snapshot(
                replace(work, color=(70, 80, 90)), first, lambda: False
            )
            third, cost_a_again, _, stats_a = raster.prepare_snapshot(
                work, second, lambda: False
            )
            auto, _, _, _ = raster.prepare_snapshot(
                replace(work, color=None), third, lambda: False
            )
        np.testing.assert_array_equal(cost_a_again, cost_a)
        self.assertFalse(np.array_equal(cost_b, cost_a))
        self.assertFalse(stats_a["cost_cache_hit"] or stats_b["cost_cache_hit"])
        self.assertIs(first.bands[0], second.bands[0])
        self.assertIs(second.bands[0], third.bands[0])
        self.assertFalse(cost_a.flags.writeable)
        goal = (work.goal[0] - auto.bounds[0], work.goal[1] - auto.bounds[2])
        self.assertEqual(auto.cost_key[1], tuple(float(b[goal]) for b in auto.bands))
        with patch.object(raster, "read_rgb", wraps=raster.read_rgb) as read:
            raster.prepare_snapshot(
                replace(work, source=replace(source, revision=2)), third, lambda: False
            )
        self.assertEqual(read.call_count, 1)

    def test_prefetch_masks_and_outside_cost_overflow_do_not_change_graph(self):
        values = np.ones((96, 128))
        values[40, 40], values[41, 41], values[42, 42] = np.nan, np.inf, -999
        values[32, 32] = 2.0**40  # Outside the requested window, inside prefetch.
        source = self.source(values, nodata=-999)
        work = self.work(source)
        empty = np.empty
        cost_allocations = []

        def tracked_empty(shape, *args, **kwargs):
            array = empty(shape, *args, **kwargs)
            if array.dtype == np.dtype("int64"):
                if cost_allocations:
                    self.assertIsNone(
                        cost_allocations[0](), "failed prefetch cost remains live"
                    )
                cost_allocations.append(weakref.ref(array))
            return array

        with patch.object(raster.np, "empty", side_effect=tracked_empty):
            snapshot, graph, valid, _ = raster.prepare_snapshot(
                work, None, lambda: False
            )
        self.assertEqual(len(cost_allocations), 2)
        self.assertEqual(snapshot.bounds, work.bounds)
        self.assertEqual(snapshot.nbytes, 32 * 32 * 25 + 20 * 23 * 8)
        self.assertEqual(np.count_nonzero(~valid), 3)
        exact = self.exact(work)
        warm = execute_request(work, snapshot)
        self.assertEqual(
            (warm.status, warm.cost, warm.path), (exact.status, exact.cost, exact.path)
        )
        np.testing.assert_array_equal(graph, exact.snapshot.cost)

    def test_alignment_falls_back_within_budget_and_accounts_borrowed_cache(self):
        source = self.source(np.ones((128, 128)))
        work = self.work(source, (33, 51, 33, 51))
        reserve = raster.SCRATCH_BYTES + raster.CURSOR_RESERVE_BYTES
        for pixels, expected in ((24 * 24, (32, 56, 32, 56)), (18 * 18, work.bounds)):
            with (
                self.subTest(pixels=pixels),
                patch.object(raster, "MAX_ARRAY_BYTES", reserve + pixels * 42),
            ):
                snapshot, _, _, _ = raster.prepare_snapshot(work, None, lambda: False)
                self.assertEqual(snapshot.bounds, expected)
                changed, _, _, stats = raster.prepare_snapshot(
                    replace(work, color=None), snapshot, lambda: False
                )
                self.assertIs(changed.bands[0], snapshot.bands[0])
                self.assertFalse(stats["cost_cache_hit"])
                self.assertLessEqual(
                    snapshot.nbytes + snapshot.valid.size * 8 + reserve,
                    raster.MAX_ARRAY_BYTES,
                )
        old, _, _, _ = raster.prepare_snapshot(work, None, lambda: False)
        moved = self.work(source, (65, 83, 65, 83))
        with (
            patch.object(raster, "MAX_ARRAY_BYTES", reserve + 18 * 18 * 42),
            patch.object(
                raster.gdal,
                "OpenEx",
                side_effect=AssertionError("must check before reading"),
            ),
        ):
            with self.assertRaises(ResourceLimitError):
                raster.prepare_snapshot(moved, old, lambda: False)

    def test_cancel_retains_only_prepared_cache_without_a_path(self):
        source = self.source(np.ones((96, 128)))
        work = self.work(source)
        snapshot, _, _, _ = raster.prepare_snapshot(work, None, lambda: False)
        with patch(
            "raster_scribe.pointtool_tasks._find_path_core", side_effect=TraceCancelled
        ):
            result = execute_request(work, snapshot)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.path, ())
        self.assertIs(result.snapshot, snapshot)
        outcomes = []
        handed_off = []

        def terminal(task, result):
            outcomes.append(result)
            handed_off.append(task._terminal_snapshot)

        task = FindPathTask(work, snapshot, terminal)
        task.outcome = TraceResult(work.request_id, "success", snapshot=snapshot)
        task.cancel()
        task.finished(True)
        self.assertEqual(outcomes[0].status, "cancelled")
        self.assertIsNone(outcomes[0].snapshot)
        self.assertIs(handed_off[0], snapshot)
        self.assertIsNone(task._terminal_snapshot)

    def test_numpy_color_snap_matches_distance_row_column_reference(self):
        rng = np.random.default_rng(582)
        context = raster.RasterTracingContext(None)
        for shape in ((1, 1), (7, 9), (31, 31), (199, 199)):
            for tied in (False, True):
                values = (
                    np.zeros(shape) if tied else rng.integers(0, 5, shape).astype(float)
                )
                valid = rng.random(shape) > 0.15
                bounds = (100, 100 + shape[0], 200, 200 + shape[1])
                bands = (values, values, values)
                context.raster_sampler = SimpleNamespace(
                    read_small=lambda *args: (bounds, bands, valid)
                )
                i, j = bounds[0] + shape[0] // 2, bounds[2] + shape[1] // 2
                costs = raster.color_cost(bands, valid, (1, 1, 1))
                points = [
                    (int(r + bounds[0]), int(c + bounds[2]))
                    for r, c in zip(*np.nonzero(valid))
                ]
                expected = (
                    min(
                        points,
                        key=lambda p: (
                            int(costs[p[0] - bounds[0], p[1] - bounds[2]]),
                            (p[0] - i) ** 2 + (p[1] - j) ** 2,
                            *p,
                        ),
                    )
                    if points
                    else (i, j)
                )
                self.assertEqual(context.snap(i, j, 99, (1, 1, 1)), expected)


class CacheCancellationTest(TraceFixture):
    def test_terminal_releases_evicted_arrays_before_replacement_submit(self):
        controller = self.plugin.task_controller
        submitted = []
        old_arrays = []

        def submit(task):
            if submitted:
                self.assertTrue(all(reference() is None for reference in old_arrays))
                self.assertIsNone(submitted[0]._terminal_snapshot)
                self.assertIsNone(submitted[0].outcome)
            submitted.append(task)

        controller._submit = submit
        self.accept(8, 2)
        request = self.tool.make_request(
            self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        )
        controller.submit(request, self.tool._on_result)
        first = submitted[0]
        first.outcome = execute_request(first.work, first.snapshot)
        old_arrays.extend(
            weakref.ref(array)
            for array in (*first.outcome.snapshot.bands, first.outcome.snapshot.cost)
        )
        original_terminal = first.callback

        def retaining_wrapper(task, result):
            # The live recorder also retains callback arguments across terminal.
            self.assertIsNone(result.snapshot)
            return original_terminal(task, result)

        first.callback = retaining_wrapper
        changed_source = replace(
            request.context.source, revision=request.context.source.revision + 1
        )
        replacement = replace(
            request,
            request_id=request.request_id + 1,
            context=replace(request.context, source=changed_source),
        )
        controller.submit(replacement, self.tool._on_result)
        first.finished(True)
        self.assertEqual(len(submitted), 2)
        controller.cancel(self.tool.owner)
        submitted[-1].finished(False)
        self.assertFalse(controller.active)

    def test_late_cache_kept_only_without_explicit_eviction(self):
        controller = self.plugin.task_controller
        controller._factory = ControlledTask
        controller._submit = lambda task: None
        for evict in (False, True):
            with self.subTest(evict=evict):
                self.tool.finish_session()
                self.accept(8, 2)
                self.accept(8, 13)
                task = controller._task
                result = execute_request(task.work, task.snapshot)
                if evict:
                    queued = self.tool.make_request(
                        self.tool.resolve_endpoint(self.tool.to_coords(8, 12))
                    )
                    controller.submit(queued, self.tool._on_result, committed=True)
                    self.assertIsNotNone(controller._queued)
                    self.tool.finish_session()
                else:
                    controller.cancel(self.tool.owner, evict=False)
                task.callback(task, result)  # Force stale success after cancellation.
                self.assertEqual(self.vector.featureCount(), 0)
                self.assertEqual(len(self.tool.anchors), 0 if evict else 1)
                self.assertIsNone(controller._queued)
                self.assertEqual(controller._snapshot is None, evict)
