"""Pure numerical regressions; no QGIS, GDAL or NumPy imports."""

import random
import unittest

from ..astar import _find_path_core
from ..line_simplification import simplify, smooth


class Grid:
    def __init__(self, rows):
        self.rows = rows
        self.shape = len(rows), len(rows[0])

    def __getitem__(self, point):
        return self.rows[point[0]][point[1]]


def reference_cost(grid, start, goal):
    # Independent Bellman-Ford relaxation on tiny grids.
    height, width = grid.shape
    distance = {start: 0}
    for _ in range(height * width):
        changed = False
        for row in range(height):
            for col in range(width):
                point = row, col
                if point not in distance:
                    continue
                for next_row, next_col in (
                    (row, col + 1),
                    (row + 1, col),
                    (row, col - 1),
                    (row - 1, col),
                ):
                    neighbor = next_row, next_col
                    if 0 <= next_row < height and 0 <= next_col < width:
                        cost = distance[point] + grid[neighbor]
                        if neighbor not in distance or cost < distance[neighbor]:
                            distance[neighbor] = cost
                            changed = True
        if not changed:
            break
    return distance[goal]


class SearchTest(unittest.TestCase):
    def test_zero_cost_detour(self):
        result = _find_path_core(
            Grid([[0, 0, 0], [0, 1, 0], [9, 9, 9]]), (1, 0), (1, 2)
        )
        self.assertEqual(result.cost, 0)
        self.assertIn((0, 1), result.path)

    def test_seeded_reference_connectivity_and_determinism(self):
        rng = random.Random(1945)
        for _ in range(120):
            grid = Grid([[rng.randrange(5) for _ in range(4)] for _ in range(4)])
            start, goal = (
                (rng.randrange(4), rng.randrange(4)),
                (rng.randrange(4), rng.randrange(4)),
            )
            result = _find_path_core(grid, start, goal)
            self.assertEqual(result.cost, reference_cost(grid, start, goal))
            self.assertEqual(result.path, _find_path_core(grid, start, goal).path)
            self.assertEqual(result.path[0], start)
            self.assertEqual(result.path[-1], goal)
            self.assertEqual(sum(grid[p] for p in result.path[1:]), result.cost)
            for a, b in zip(result.path, result.path[1:]):
                self.assertEqual(sum(abs(x - y) for x, y in zip(a, b)), 1)

    def test_high_cost_uses_python_integers(self):
        result = _find_path_core(Grid([[0, 2**62, 2**62, 2**62]]), (0, 0), (0, 3))
        self.assertEqual(result.cost, 3 * 2**62)

    def test_cancellation_including_same_pixel(self):
        grid = Grid([[0, 0]])
        self.assertEqual(
            _find_path_core(grid, (0, 0), (0, 0), lambda: True).status, "cancelled"
        )
        calls = iter([False, False, True])
        self.assertEqual(
            _find_path_core(grid, (0, 0), (0, 1), lambda: next(calls)).status,
            "cancelled",
        )

    def test_masks_no_path_invalid_input_and_limits(self):
        grid = Grid([[0, 0, 0]])
        mask = Grid([[True, False, True]])
        self.assertEqual(
            _find_path_core(grid, (0, 0), (0, 2), valid=mask).status, "no_path"
        )
        self.assertEqual(
            _find_path_core(grid, (0, 1), (0, 2), valid=mask).status, "invalid_input"
        )
        self.assertEqual(_find_path_core(grid, (-1, 0), (0, 2)).status, "invalid_input")
        self.assertEqual(
            _find_path_core(grid, (0, 0), (0, 2), max_nodes=1).status, "resource_limit"
        )
        self.assertEqual(
            _find_path_core(grid, (0, 0), (0, 2), max_frontier=0).status,
            "resource_limit",
        )
        for value in (-1, float("nan"), float("inf"), 0.1):
            self.assertEqual(
                _find_path_core(Grid([[0, value]]), (0, 0), (0, 1)).status,
                "invalid_input",
            )


class PostprocessTest(unittest.TestCase):
    def test_short_inputs_and_ownership(self):
        for path in ([], [(1, 2)], [(1, 2), (3, 4)]):
            for function in (smooth, simplify):
                result = function(path)
                self.assertEqual(result, path)
                self.assertIsNot(result, path)
        path = [(0, 0), (1, 2), (2, 1)]
        self.assertEqual(smooth(path, 0), path)

    def test_u_and_reversal(self):
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
        expected = [
            (0, 0),
            (0, 1),
            (1 / 3, 5 / 3),
            (1, 2),
            (5 / 3, 5 / 3),
            (2, 1),
            (2, 0),
        ]
        for a, b in zip(smooth(path, 5), expected):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y)
        self.assertEqual(path[2], (0, 2))
        self.assertGreater(len(simplify(smooth(path, 5))), 2)
        for n in range(3, 11):
            bend = [(i, (i - n // 2) ** 2) for i in range(n)]
            result = smooth(bend, 5)
            self.assertEqual(result, list(reversed(smooth(list(reversed(bend)), 5))))
            self.assertEqual((result[0], result[-1]), (bend[0], bend[-1]))

    def test_angles_duplicates_uturns_and_wrap(self):
        for path in (
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (0, 0), (1, 0), (2, 0)],
            [(2, 0), (1, 0.001), (0, 0)],
        ):
            original = list(path)
            self.assertEqual(simplify(path), [path[0], path[-1]])
            self.assertEqual(path, original)
        path = [(0, 0), (1, 0), (0, 0)]
        self.assertEqual(simplify(path), path)
        self.assertEqual(simplify([(0, 0), (1, 0), (1, 1)]), [(0, 0), (1, 0), (1, 1)])
