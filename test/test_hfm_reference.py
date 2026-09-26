"""Check the Python solver against independent upstream numerical results."""

import json
import unittest
from pathlib import Path

import numpy as np

from ..hfm_backtrace import backtrace
from ..hfm_march import solve


class HFMReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(
            Path(__file__).with_name("hfm_reference.json").read_text()
        )
        source = cls.reference
        cost = np.asarray(source["cost_yx"])
        valid = np.ones(cost.shape, dtype=bool)
        for x, y in source["invalid_xy"]:
            valid[y, x] = False
        cls.field = solve(cost, source["start_xy"], source["goal_xy"], valid)

    def test_upwind_values_active_neighbors_and_stopping_match_upstream(self):
        self.assertEqual(self.field["status"], "success")
        reference = self.reference
        values = self.field["values"].ravel()
        indices = np.asarray(reference["finite_indices"])
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(values)), indices)
        np.testing.assert_allclose(
            values[indices], reference["values"], rtol=1e-11, atol=1e-10
        )
        expected_active = np.zeros(values.shape, dtype=np.uint32)
        expected_active[indices] = reference["active"]
        np.testing.assert_array_equal(self.field["active"].ravel(), expected_active)
        self.assertEqual(
            self.field["metadata"]["accepted_states"] + 1,
            reference["n_accepted_including_goal"],
        )
        self.assertFalse(self.field["accepted"][self.field["goal_state"]])

    def test_both_geodesic_extractors_match_upstream(self):
        for method, expected in self.reference["paths"].items():
            with self.subTest(method=method):
                result = backtrace(self.field, solver=method, fallback_discrete=False)
                self.assertEqual(
                    result["status"], self.reference["expected_status"][method]
                )
                np.testing.assert_allclose(
                    result["raw_path_xy"], expected, rtol=0, atol=1e-9
                )
                if result["status"] == "success":
                    self.assertEqual(
                        result["path_xy"][0], tuple(self.reference["start_xy"])
                    )
                    self.assertEqual(
                        result["path_xy"][-1], tuple(self.reference["goal_xy"])
                    )
                else:
                    # This upstream Discrete path stops too far from its seed;
                    # matching its raw coordinates must not bypass validation.
                    self.assertEqual(result["path_xy"], [])
