"""Enhanced tracing's raster, worker, preview and commit contracts."""

import math
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from osgeo import gdal
from qgis.core import QgsPointXY

from .. import pointtool_raster as raster
from .. import pointtool_tasks as tasks
from ..astar import _find_path_core
from ..pointtool_session import RasterSourceSpec, WorkerRequest
from .tracing_fixture import TraceFixture
from .utilities import get_qgis_app, write_raster


class HFMRasterIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.color = (170, 140, 95)

    def arc_work(self, *, color=None, barrier=False, invalid_goal=False):
        """Brown arc with a short dark occlusion, generated without fixture files."""
        rows, columns = np.mgrid[:48, :64]
        center = 24 + 8 * np.sin(np.pi * (columns - 8) / 48)
        stroke = (columns >= 8) & (columns <= 56) & (abs(rows - center) <= 1)
        rgb = np.empty((48, 64, 3), dtype=np.uint8)
        rgb[:] = (225, 230, 215)
        rgb[stroke] = self.color
        rgb[:, 31:34] = (25, 25, 25)
        if barrier:
            rgb[:, 32] = 255
        if invalid_goal:
            rgb[24, 56] = 255
        path = Path(self.directory.name) / "arc.tif"
        write_raster(
            path,
            tuple(rgb[..., band] for band in range(3)),
            gdal.GDT_Byte,
            nodata=255 if barrier or invalid_goal else None,
        )
        source = RasterSourceSpec(str(path), 48, 64, 1)
        return WorkerRequest(
            1,
            source,
            (0, 48, 0, 64),
            (24, 8),
            (24, 56),
            color,
            False,
            enhanced_tracing=True,
        )

    def assert_arc(self, result):
        self.assertEqual(result.status, "success", result.diagnostics)
        self.assertIsNone(result.cost)
        self.assertGreater(result.timings["hfm_states"], 0)
        path = np.asarray(result.path, dtype=float)
        np.testing.assert_array_equal(path[[0, -1]], [[24, 8], [24, 56]])
        self.assertTrue(np.isfinite(path).all())
        self.assertGreater(len(path), 35)
        self.assertEqual(result.processed, result.path)
        # The raster is a known curve: a shortcut, ink-following detour, or
        # column/row transposition must not pass merely by reaching the goal.
        expected_rows = 24 + 8 * np.sin(np.pi * (path[:, 1] - 8) / 48)
        error = abs(path[:, 0] - expected_rows)
        visible = (path[:, 1] < 30) | (path[:, 1] > 35)
        self.assertLessEqual(float(np.max(error[visible])), 1.5)
        # The hidden centerline need not be recovered to subpixel accuracy.
        # A two-pixel bound through the ink still rules out a straight shortcut.
        self.assertLessEqual(float(np.max(error)), 2)
        self.assertLessEqual(
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).max()), 1.1
        )
        self.assertTrue(np.any((path[:, 1] >= 31) & (path[:, 1] <= 33)))

    def test_fixed_automatic_and_resampled_colors_keep_the_arc(self):
        work = self.arc_work(color=self.color)
        fixed = tasks.execute_request(work)
        self.assert_arc(fixed)
        self.assertEqual(fixed.snapshot.cost_key[1], self.color)
        with patch.object(raster, "read_rgb", side_effect=AssertionError("cached RGB")):
            automatic = tasks.execute_request(replace(work, color=None), fixed.snapshot)
            resampled_color = (171, 141, 96)
            resampled = tasks.execute_request(
                replace(work, color=resampled_color), automatic.snapshot
            )
        self.assert_arc(automatic)
        self.assert_arc(resampled)
        self.assertEqual(automatic.snapshot.cost_key[1], self.color)
        self.assertEqual(resampled.snapshot.cost_key[1], resampled_color)
        self.assertEqual(fixed.path, automatic.path)
        self.assertIs(fixed.snapshot.bands[0], resampled.snapshot.bands[0])
        self.assertFalse(resampled.snapshot.cost.flags.writeable)

    def test_nodata_barrier_and_endpoint_never_enter_refinement(self):
        for options, expected in (
            ({"barrier": True}, "no_path"),
            ({"invalid_goal": True}, "invalid_input"),
        ):
            with self.subTest(options=options):
                work = self.arc_work(color=self.color, **options)
                with patch.object(
                    tasks, "refine_path", side_effect=AssertionError("nodata")
                ) as refine:
                    result = tasks.execute_request(work)
                self.assertEqual(result.status, expected, result.diagnostics)
                self.assertEqual(result.path, ())
                self.assertEqual(result.processed, ())
                refine.assert_not_called()

    def test_cancellation_after_refinement_discards_path_and_preserves_cache(self):
        work = self.arc_work(color=self.color)
        snapshot, _, _, _ = raster.prepare_snapshot(work, None, lambda: False)
        cancelled = threading.Event()
        refine = tasks.refine_path

        def finish_then_cancel(*args, **kwargs):
            result = refine(*args, **kwargs)
            cancelled.set()
            return result

        with patch.object(tasks, "refine_path", side_effect=finish_then_cancel) as call:
            result = tasks.execute_request(work, snapshot, cancelled.is_set)
        call.assert_called_once()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.path, ())
        self.assertEqual(result.processed, ())
        self.assertIs(result.snapshot, snapshot)

    def test_backend_rejection_reports_failure_without_a_partial_path(self):
        work = self.arc_work(color=self.color)
        with patch.object(
            tasks, "refine_path", side_effect=ValueError("incomplete HFM path")
        ):
            result = tasks.execute_request(work)
        self.assertEqual(result.status, "error")
        self.assertIn("incomplete HFM path", result.diagnostics)
        self.assertEqual(result.path, ())
        self.assertEqual(result.processed, ())

    def test_snapshot_softens_dark_ink_but_keeps_background_cost(self):
        rgb = np.array(
            [[[170, 140, 95], [25, 25, 25], [225, 230, 215]]], dtype=np.uint8
        )
        path = Path(self.directory.name) / "colors.tif"
        write_raster(path, tuple(rgb[..., i] for i in range(3)), gdal.GDT_Byte)
        source = RasterSourceSpec(str(path), 1, 3, 1)
        work = WorkerRequest(
            1,
            source,
            (0, 1, 0, 3),
            (0, 0),
            (0, 2),
            self.color,
            False,
            enhanced_tracing=True,
        )
        snapshot, cost, valid, _ = raster.prepare_snapshot(work, None, lambda: False)
        raw = raster.color_cost(snapshot.bands, valid, self.color)
        self.assertEqual(int(cost[0, 0]), 0)
        self.assertEqual(int(cost[0, 1]), round(2048 + 0.1 * (int(raw[0, 1]) - 2048)))
        self.assertLess(int(cost[0, 1]), int(raw[0, 1]))
        self.assertEqual(int(cost[0, 2]), int(raw[0, 2]))

    def test_default_worker_matches_original_color_cost_and_search(self):
        enhanced = self.arc_work(color=self.color)
        # Construct without the new option: existing requests keep the original
        # scoring and search, including their integer path and total cost.
        work = WorkerRequest(
            enhanced.request_id,
            enhanced.source,
            enhanced.bounds,
            enhanced.start,
            enhanced.goal,
            enhanced.color,
            enhanced.smoothing,
        )
        self.assertFalse(work.enhanced_tracing)
        for color in (self.color, None):
            with (
                self.subTest(color=color),
                patch.object(tasks, "refine_path") as refine,
            ):
                actual = tasks.execute_request(replace(work, color=color))
                self.assertEqual(actual.status, "success", actual.diagnostics)
                target = color or tuple(
                    float(b[work.goal]) for b in actual.snapshot.bands
                )
                raw = sum(
                    (band.astype(float) - value) ** 2
                    for band, value in zip(actual.snapshot.bands, target)
                ).astype(np.int64)
                expected = _find_path_core(
                    raw, work.start, work.goal, valid=actual.snapshot.valid
                )
                np.testing.assert_array_equal(actual.snapshot.cost, raw)
                self.assertEqual(actual.path, tuple(expected.path))
                self.assertEqual(actual.cost, expected.cost)
                self.assertEqual(actual.processed, actual.path)
                self.assertNotIn("hfm_total", actual.timings)
                refine.assert_not_called()

    def test_switching_algorithm_reuses_rgb_but_never_incompatible_costs(self):
        for color in (self.color, None):
            with self.subTest(color=color):
                work = replace(self.arc_work(color=color), enhanced_tracing=False)
                original, raw, _, _ = raster.prepare_snapshot(work, None, lambda: False)
                with patch.object(
                    raster, "read_rgb", side_effect=AssertionError("cached RGB")
                ):
                    enhanced, soft, _, stats = raster.prepare_snapshot(
                        replace(work, enhanced_tracing=True), original, lambda: False
                    )
                    restored, restored_cost, _, restored_stats = (
                        raster.prepare_snapshot(work, enhanced, lambda: False)
                    )
                    same, _, _, hit_stats = raster.prepare_snapshot(
                        work, restored, lambda: False
                    )
                self.assertTrue(stats["rgb_cache_hit"])
                self.assertFalse(stats["cost_cache_hit"])
                self.assertFalse(restored_stats["cost_cache_hit"])
                self.assertTrue(hit_stats["cost_cache_hit"])
                self.assertIs(same, restored)
                self.assertIs(enhanced.bands[0], original.bands[0])
                self.assertIs(restored.valid, original.valid)
                self.assertNotEqual(enhanced.cost_key, original.cost_key)
                self.assertEqual(restored.cost_key, original.cost_key)
                self.assertLess(int(soft[24, 32]), int(raw[24, 32]))
                np.testing.assert_array_equal(restored_cost, raw)


class HFMPreviewIntegrationTest(TraceFixture):
    def assert_known_line(self, path):
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], self.expected_path[0])
        self.assertEqual(path[-1], self.expected_path[-1])
        self.assertTrue(all(math.isfinite(v) for point in path for v in point))
        self.assertTrue(all(abs(row - 8) <= 0.5 for row, _ in path))
        self.assertTrue(all(2 <= column <= 13 for _, column in path))
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        self.assertAlmostEqual(length, 11, delta=0.25)

    def receive_result(self, request, result):
        self.assertEqual(result.status, "cancelled")

    def test_scheduler_drops_changed_cost_policy_before_starting_worker(self):
        submitted = self.control_tasks()
        self.accept(8, 2)
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        base = self.tool.make_request(endpoint)
        for color in ((0, 0, 0), None):
            for enhanced in (False, True):
                with self.subTest(color=color, enhanced=enhanced):
                    before = replace(base, color=color, enhanced_tracing=enhanced)
                    after = replace(before, enhanced_tracing=not enhanced)
                    self.assertNotEqual(before.key, after.key)
                    self.assertEqual(
                        after.worker_input().enhanced_tracing, not enhanced
                    )
                    snapshot, _, _, _ = raster.prepare_snapshot(
                        before.worker_input(), None, lambda: False
                    )
                    self.scheduler._snapshot = snapshot
                    self.scheduler.submit(after, self.receive_result)
                    worker_snapshot = submitted[-1].snapshot
                    self.assertIs(worker_snapshot.bands[0], snapshot.bands[0])
                    self.assertIs(worker_snapshot.valid, snapshot.valid)
                    self.assertIsNone(worker_snapshot.cost)
                    self.assertIsNone(worker_snapshot.cost_key)
                    submitted[-1].finish("cancelled")

    def test_cached_refined_preview_commits_without_a_second_solver_run(self):
        submitted = self.control_tasks()
        self.tool.enhanced_tracing = True
        self.tool.smooth_line = True
        self.accept(8, 2)
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(8, 13))
        screen = self.tool.toCanvasCoordinates(QgsPointXY(*endpoint.xy))
        request = self.tool.make_request(endpoint, screen)
        with patch.object(tasks, "refine_path", wraps=tasks.refine_path) as refine:
            self.tool.preview_controller.queue(request)
            self.tool.preview_controller.ensure_inflight_started()
            self.assertEqual(len(submitted), 1)
            submitted[0].finish()
            cached = self.tool.preview_controller._cached_result
            self.assertEqual(cached.status, "success")
            self.assert_known_line(cached.path)
            rendered = self.tool.preview_controller._rubber_band.asGeometry()
            points = [QgsPointXY(vertex) for vertex in rendered.vertices()]
            self.tool.accept_click(endpoint.xy, screen)
            self.assertEqual(self.tool.session.geometry.asMultiPolyline()[0], points)
            self.assertTrue(self.tool.finish_session())
            committed = next(self.vector.getFeatures()).geometry().asMultiPolyline()[0]
            self.assertEqual(committed, points)
            refine.assert_called_once()
        self.assertEqual(len(submitted), 1)
        self.assertTrue(
            all(math.isfinite(p.x()) and math.isfinite(p.y()) for p in committed)
        )
