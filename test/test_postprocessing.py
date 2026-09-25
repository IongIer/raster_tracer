"""Native simplification fidelity and its worker/preview/commit contract."""

import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from qgis.core import QgsPointXY

from ..astar import FindPathCoreResult
from ..line_simplification import smooth
from ..pointtool_tasks import execute_request, postprocess_path
from .test_tracing_hardening import ControlledTask
from .tracing_fixture import TraceFixture


def distance_to_line(point, line):
    """Independent point-to-segment calculation in pixel coordinates."""
    distances = []
    for start, end in zip(line, line[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_squared = dx * dx + dy * dy
        fraction = (
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
            if length_squared
            else 0
        )
        fraction = min(1, max(0, fraction))
        distances.append(
            math.hypot(
                point[0] - start[0] - fraction * dx,
                point[1] - start[1] - fraction * dy,
            )
        )
    return min(distances)


class NativePostprocessTest(unittest.TestCase):
    def test_short_degenerate_and_closed_paths_preserve_endpoints(self):
        closed = (
            [(0, col) for col in range(5, 11)]
            + [(row, 10) for row in range(1, 11)]
            + [(10, col) for col in range(9, -1, -1)]
            + [(row, 0) for row in range(9, -1, -1)]
            + [(0, col) for col in range(1, 6)]
        )
        for path in ([], [(1, 2)], [(1, 2), (3, 4)], [(1, 2)] * 20, closed):
            with self.subTest(size=len(path)):
                original = list(path)
                result = postprocess_path(path)
                self.assertIsInstance(result, tuple)
                self.assertEqual(path, original)
                self.assertTrue(all(math.isfinite(v) for p in result for v in p))
                if path:
                    self.assertEqual((result[0], result[-1]), (path[0], path[-1]))
                if len(path) <= 2:
                    self.assertEqual(result, tuple(path))
                elif path == closed:
                    for point in smooth(path, size=5):
                        self.assertLessEqual(
                            distance_to_line(point, result), 0.25 + 1e-9
                        )
                else:
                    self.assertEqual(result, (path[0], path[-1]))

    def test_quarter_pixel_deviation_and_mean_shape_with_fewer_vertices(self):
        path = [(i, 3 * math.sin(i / 7) + 0.2 * (-1) ** i) for i in range(101)]
        averaged = smooth(path, size=5)
        result = postprocess_path(path)
        self.assertLess(len(result), len(path) // 2)
        self.assertGreater(len(result), 2)
        self.assertEqual((result[0], result[-1]), (path[0], path[-1]))
        for point in averaged:
            self.assertLessEqual(distance_to_line(point, result), 0.25 + 1e-9)


class WorkerPostprocessTest(TraceFixture):
    def make_work(self, smoothing=True):
        self.tool.smooth_line = smoothing
        self.accept(8, 2)
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(4, 8))
        return self.tool.make_request(endpoint).worker_input()

    def test_smoothing_off_and_empty_search_results(self):
        work = self.make_work(smoothing=False)
        with patch(
            "raster_scribe.pointtool_tasks.postprocess_path",
            side_effect=AssertionError("Smoothing is disabled"),
        ):
            result = execute_request(work)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.processed, result.path)
        for status in ("no_path", "cancelled", "resource_limit"):
            with self.subTest(status=status):
                empty = FindPathCoreResult(
                    status=status, profile_stats={"duration": 0.0, "nodes": 0}
                )
                with patch(
                    "raster_scribe.pointtool_tasks._find_path_core", return_value=empty
                ):
                    result = execute_request(replace(work, smoothing=True))
                self.assertEqual(result.status, status)
                self.assertEqual(result.processed, ())

    def test_cancellation_after_native_postprocessing_discards_the_result(self):
        work = self.make_work()
        cancelled = threading.Event()

        def finish_then_cancel(path):
            result = postprocess_path(path)
            cancelled.set()
            return result

        with patch(
            "raster_scribe.pointtool_tasks.postprocess_path",
            side_effect=finish_then_cancel,
        ):
            result = execute_request(work, cancel=cancelled.is_set)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.processed, ())
        self.assertEqual(self.vector.featureCount(), 0)

    def test_curved_worker_preview_and_adopted_commit_use_the_same_points(self):
        self.tool.smooth_line = True
        submitted = []
        scheduler = self.plugin.task_controller
        scheduler._factory = ControlledTask
        scheduler._submit = submitted.append
        self.accept(8, 2)
        endpoint = self.tool.resolve_endpoint(self.tool.to_coords(4, 8))
        screen = self.tool.toCanvasCoordinates(QgsPointXY(*endpoint.xy))
        request = self.tool.make_request(endpoint, screen)
        self.tool.preview_controller.queue(request)
        self.tool.preview_controller.ensure_inflight_started()
        task = submitted[-1]
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(execute_request, task.work, task.snapshot).result()
        self.assertEqual(result.status, "success")
        self.assertGreater(len(result.processed), 2)
        self.assertLess(len(result.processed), len(result.path))
        task.callback(task, result)
        rendered = self.tool.preview_controller._rubber_band.asGeometry()
        preview_points = [QgsPointXY(vertex) for vertex in rendered.vertices()]
        self.tool.accept_click(endpoint.xy, screen)
        draft = self.tool.session.geometry.asMultiPolyline()[0]
        self.assertEqual(draft, preview_points)
        self.assertEqual(self.vector.featureCount(), 0)
        self.assertTrue(self.tool.finish_session())
        committed = next(self.vector.getFeatures()).geometry().asMultiPolyline()[0]
        self.assertEqual(committed, preview_points)
        self.assertEqual((committed[-1].x(), committed[-1].y()), endpoint.xy)
        self.assertEqual(len(submitted), 1)
