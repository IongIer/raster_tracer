"""Portable HFM coordinates, budgets, cancellation and failure contracts."""

import time
import unittest
from unittest.mock import patch

import numpy as np

from .. import pointtool_hfm as hfm
from ..astar import _find_path_core
from ..exceptions import ResourceLimitError, TraceCancelled


class HfmAdapterTest(unittest.TestCase):
    def setUp(self):
        self.graph = np.full((32, 48), 100, dtype=np.int64)
        self.valid = np.ones(self.graph.shape, dtype=bool)
        self.start, self.goal = (16, 5), (16, 40)
        self.coarse = _find_path_core(
            self.graph, self.start, self.goal, valid=self.valid
        ).path

    def refine(self, **kwargs):
        return hfm.refine_path(
            self.graph, self.valid, self.start, self.goal, self.coarse, **kwargs
        )

    def test_real_solver_preserves_centers_with_large_retained_snapshot(self):
        path, timings = self.refine(live_bytes=80 * 1024 * 1024)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        self.assertGreater(timings["hfm_states"], 0)
        self.assertGreater(timings["hfm_marching"], 0)
        self.assertLess(timings["hfm_estimated_bytes"], hfm.MAX_ARRAY_BYTES)
        self.assertLess(np.max(np.abs(np.asarray(path)[:, 0] - 16)), 0.1)
        self.assertLessEqual(np.linalg.norm(np.diff(path, axis=0), axis=1).max(), 1.01)

    def test_degenerate_paths_do_not_solve_and_still_cancel(self):
        with patch.object(hfm, "solve") as solve:
            for goal in ((16, 5), (16, 6), (17, 6)):
                path, timings = hfm.refine_path(
                    self.graph, self.valid, self.start, goal, [self.start, goal]
                )
                self.assertEqual(path[0], self.start)
                self.assertEqual(path[-1], goal)
                self.assertEqual(timings["hfm_states"], 0)
            with self.assertRaises(TraceCancelled):
                hfm.refine_path(
                    self.graph, self.valid, self.start, self.start, [], lambda: True
                )
            solve.assert_not_called()

    def test_diagonal_neighbor_cannot_squeeze_past_masked_corner(self):
        self.valid[16, 6] = False
        with self.assertRaisesRegex(ValueError, "invalid raster pixel"):
            hfm.refine_path(
                self.graph, self.valid, (16, 5), (17, 6), [(16, 5), (17, 5), (17, 6)]
            )

    def test_snapshot_and_state_budgets_precede_marching(self):
        with patch.object(hfm, "solve") as solve:
            with self.assertRaisesRegex(ResourceLimitError, "retained array budget"):
                self.refine(live_bytes=hfm.MAX_ARRAY_BYTES)
            graph = np.zeros((400, 400), dtype=np.int64)
            with self.assertRaisesRegex(ResourceLimitError, "state limit"):
                hfm.refine_path(
                    graph,
                    np.ones_like(graph, dtype=bool),
                    (0, 0),
                    (399, 399),
                    [(0, 0), (399, 399)],
                )
            solve.assert_not_called()

    def test_marching_failures_are_explicit_without_coarse_fallback(self):
        for status, error in (
            ("cancelled", TraceCancelled),
            ("timeout", ResourceLimitError),
            ("resource_limit", ResourceLimitError),
            ("no_path", ValueError),
        ):
            with (
                self.subTest(status=status),
                patch.object(hfm, "solve", return_value={"status": status}),
                patch.object(hfm, "backtrace") as extract,
            ):
                with self.assertRaises(error):
                    self.refine()
                extract.assert_not_called()

    def test_backtrace_failures_are_explicit_without_coarse_fallback(self):
        for status, error in (
            ("cancelled", TraceCancelled),
            ("timeout", ResourceLimitError),
            ("resource_limit", ResourceLimitError),
            ("incomplete", ValueError),
        ):
            with (
                self.subTest(status=status),
                patch.object(hfm, "backtrace", return_value={"status": status}),
            ):
                with self.assertRaises(error):
                    self.refine()

    def test_cancel_after_marching_prevents_extraction(self):
        solve = hfm.solve
        cancelled = False

        def solve_then_cancel(*args, **kwargs):
            nonlocal cancelled
            result = solve(*args, **kwargs)
            cancelled = True
            return result

        with (
            patch.object(hfm, "solve", side_effect=solve_then_cancel),
            patch.object(hfm, "backtrace") as extract,
        ):
            with self.assertRaises(TraceCancelled):
                self.refine(cancel=lambda: cancelled)
            extract.assert_not_called()

    def test_one_shared_deadline_for_marching_and_extraction(self):
        solve, backtrace = hfm.solve, hfm.backtrace
        deadlines = []

        def call(function, *args, **kwargs):
            deadlines.append(time.perf_counter() + kwargs["timeout"])
            return function(*args, **kwargs)

        with (
            patch.object(
                hfm, "solve", side_effect=lambda *a, **kw: call(solve, *a, **kw)
            ),
            patch.object(
                hfm, "backtrace", side_effect=lambda *a, **kw: call(backtrace, *a, **kw)
            ),
        ):
            self.refine()
        self.assertEqual(len(deadlines), 2)
        self.assertLess(abs(deadlines[0] - deadlines[1]), 0.01)
        with patch.object(hfm, "TIMEOUT", 0), self.assertRaises(ResourceLimitError):
            self.refine()

    def test_narrow_mask_crossing_is_detected(self):
        valid = np.ones((3, 3), dtype=bool)
        valid[0, 1] = False
        with self.assertRaisesRegex(ValueError, "invalid raster pixel"):
            hfm._validate_path(
                np.array([[0, 0], [1.0, 1.001]]),
                valid,
                lambda: False,
                time.perf_counter() + 1,
            )

    def test_actual_mask_barrier_is_impassable(self):
        self.valid[:, 24] = False
        with self.assertRaisesRegex(ValueError, "no_path"):
            self.refine()

    def test_output_resampling_is_bounded(self):
        with (
            patch.object(hfm, "MAX_PATH_POINTS", 4),
            self.assertRaises(ResourceLimitError),
        ):
            hfm._resample(np.array([[0.0, 0.0], [0.0, 8.0]]))
