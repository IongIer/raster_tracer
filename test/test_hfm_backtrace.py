"""Portable HFM extraction safety and endpoint contracts."""

import unittest
from unittest.mock import patch

import numpy as np

from .. import hfm_backtrace as backtracking
from ..hfm_march import solve


class HfmBacktraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = solve(np.ones((19, 27)), (4, 9), (22, 9))
        assert cls.field["status"] == "success"

    def test_real_ode_matches_known_straight_geometry(self):
        result = backtracking.backtrace(self.field, fallback_discrete=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["path_xy"][0], (4, 9))
        self.assertEqual(result["path_xy"][-1], (22, 9))
        self.assertGreater(len(result["path_xy"]), 60)
        self.assertLess(np.max(np.abs(np.asarray(result["path_xy"])[:, 1] - 9)), 1e-10)
        self.assertFalse(result["metadata"]["fallback_used"])

    def test_resource_limits_return_no_partial_path(self):
        for limit, method in (
            ("MAX_FLOW_NODES", "ODE"),
            ("MAX_DISCRETE_FRONTIER", "Discrete"),
            ("MAX_PATH_POINTS", "ODE"),
        ):
            with self.subTest(limit=limit), patch.object(backtracking, limit, 1):
                result = backtracking.backtrace(
                    self.field, solver=method, fallback_discrete=False
                )
                self.assertEqual(result["status"], "resource_limit")
                self.assertEqual(result["path_xy"], [])

    def test_cancel_before_and_during_extraction_and_deadline(self):
        result = backtracking.backtrace(self.field, cancel=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3

        result = backtracking.backtrace(self.field, cancel=cancel)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["path_xy"], [])
        self.assertEqual(
            backtracking.backtrace(self.field, timeout=0)["status"], "timeout"
        )

    def test_same_and_adjacent_pixel_keep_both_endpoints(self):
        cost = np.ones((13, 17))
        for goal in ((7, 6), (8, 6), (8, 7)):
            with self.subTest(goal=goal):
                field = solve(cost, (7, 6), goal)
                result = backtracking.backtrace(field)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["path_xy"][0], (7, 6))
                self.assertEqual(result["path_xy"][-1], goal)

    def test_exact_corner_and_brief_mask_crossings_rejected(self):
        valid = np.ones((8, 8), bool)
        valid[2, 3] = False
        for path in ([(2, 2), (3, 3)], [(2, 2), (4, 2.9999)]):
            with self.subTest(path=path):
                ok, _, status = backtracking._validate(
                    path, path[0], path[-1], valid, 1.5
                )
                self.assertFalse(ok)
                self.assertEqual(status, "invalid_mask_crossing")

    def test_large_residual_is_rejected_before_endpoint_pinning(self):
        valid = np.ones((20, 20), bool)
        ok, residual, status = backtracking._validate(
            [(10, 10), (15, 15)], (2, 2), (15, 15), valid, 1.5
        )
        self.assertFalse(ok)
        self.assertGreater(residual, 10)
        self.assertEqual(status, "incomplete")
