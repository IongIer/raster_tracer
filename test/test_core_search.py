"""Exact search behavior across the discover-once optimization; no QGIS needed."""

import random
import unittest
from collections import Counter

from ..astar import FindPathCoreResult, _find_path_core
from .test_core import Grid


def reference_search(
    graph,
    start,
    goal,
    cancel_cb=None,
    valid=None,
    max_nodes=1_000_000,
    max_frontier=1_000_000,
):
    """Unoptimized Dijkstra oracle retaining the original relaxation/tie policy.

    A sorted list makes queue ordering independent of the production heap, and
    the distance dictionary deliberately does not assume first discovery wins.
    """
    distances, parents = {}, {}

    def finish(status, path=None, cost=None):
        return FindPathCoreResult(path, cost, {"nodes": len(distances)}, status)

    if cancel_cb and cancel_cb():
        return finish("cancelled")
    height, width = graph.shape
    for row, col in (start, goal):
        if not (0 <= row < height and 0 <= col < width):
            return finish("invalid_input")
        if valid is not None and not valid[row, col]:
            return finish("invalid_input")
        try:
            value = graph[row, col]
            if int(value) != value or int(value) < 0:
                return finish("invalid_input")
        except (ValueError, OverflowError, TypeError):
            return finish("invalid_input")
    if max_nodes < 1 or max_frontier < 1:
        return finish("resource_limit")
    frontier = [(0, 0, 0, start)]
    sequence = 1
    distances[start], parents[start] = 0, None
    while frontier:
        if cancel_cb and cancel_cb():
            return finish("cancelled")
        frontier.sort()
        cost, _, _, current = frontier.pop(0)
        if cost != distances[current]:
            continue
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return finish("success", path[::-1], cost)
        row, col = current
        for neighbor in (
            (row - 1, col),
            (row, col - 1),
            (row + 1, col),
            (row, col + 1),
        ):
            i, j = neighbor
            if not (0 <= i < height and 0 <= j < width):
                continue
            if valid is not None and not valid[neighbor]:
                continue
            value = graph[neighbor]
            try:
                step = int(value)
            except (ValueError, OverflowError, TypeError):
                return finish("invalid_input")
            if step < 0 or step != value:
                return finish("invalid_input")
            candidate = cost + step
            if neighbor in distances and candidate >= distances[neighbor]:
                continue
            if (neighbor not in distances and len(distances) >= max_nodes) or len(
                frontier
            ) >= max_frontier:
                return finish("resource_limit")
            distances[neighbor], parents[neighbor] = candidate, current
            distance = abs(goal[0] - i) + abs(goal[1] - j)
            frontier.append((candidate, distance, sequence, neighbor))
            sequence += 1
    return finish("no_path")


class CountingGrid(Grid):
    def __init__(self, rows):
        super().__init__(rows)
        self.reads = Counter()

    def __getitem__(self, point):
        self.reads[point] += 1
        return super().__getitem__(point)


class SearchOptimizationTest(unittest.TestCase):
    def assert_matches_reference(self, graph, start, goal, cancel_after=None, **kwargs):
        def observe(function):
            calls = 0

            def cancel():
                nonlocal calls
                calls += 1
                return cancel_after is not None and calls >= cancel_after

            result = function(graph, start, goal, cancel_cb=cancel, **kwargs)
            return (
                result.status,
                result.path,
                result.cost,
                result.profile_stats["nodes"],
                calls,
            )

        self.assertEqual(observe(_find_path_core), observe(reference_search))

    def test_seeded_masks_and_zero_cost_ties_preserve_exact_paths(self):
        rng = random.Random(731945)
        for index in range(160):
            height, width = rng.randrange(1, 8), rng.randrange(1, 8)
            graph = Grid(
                [
                    [0 if index % 4 == 0 else rng.randrange(7) for _ in range(width)]
                    for _ in range(height)
                ]
            )
            mask = Grid(
                [[rng.random() > 0.2 for _ in range(width)] for _ in range(height)]
            )
            start = rng.randrange(height), rng.randrange(width)
            goal = rng.randrange(height), rng.randrange(width)
            with self.subTest(index=index, shape=graph.shape, start=start, goal=goal):
                self.assert_matches_reference(graph, start, goal)
                self.assert_matches_reference(graph, start, goal, valid=mask)

    def test_limits_and_cancellation_preserve_traversal(self):
        rng = random.Random(52026)
        for index in range(30):
            graph = Grid([[rng.randrange(5) for _ in range(6)] for _ in range(5)])
            for nodes, frontier in ((0, 10), (10, 0), (1, 1), (5, 3), (20, 100)):
                with self.subTest(index=index, nodes=nodes, frontier=frontier):
                    self.assert_matches_reference(
                        graph, (0, 0), (4, 5), max_nodes=nodes, max_frontier=frontier
                    )
            for cancel_after in (1, 2, 3, 9, 25, 100):
                with self.subTest(index=index, cancel_after=cancel_after):
                    self.assert_matches_reference(
                        graph, (0, 0), (4, 5), cancel_after=cancel_after
                    )

    def test_new_pixels_still_validate_cost_before_discovery(self):
        for value in (-1, 0.1, float("nan"), float("inf"), None, "bad"):
            with self.subTest(value=value):
                graph = Grid([[0, value, 0], [9, 9, 9]])
                result = _find_path_core(graph, (0, 0), (0, 2))
                self.assertEqual(result.status, "invalid_input")
                self.assert_matches_reference(graph, (0, 0), (0, 2))
                # Masked cells must remain unread; their stored costs are unused.
                mask = Grid([[True, False, True], [True, True, True]])
                self.assert_matches_reference(graph, (0, 0), (0, 2), valid=mask)
                self.assertEqual(
                    _find_path_core(graph, (0, 0), (0, 2), valid=mask).status, "success"
                )
        # The core still validates lazily: invalid cells never reached by this
        # search are the array-preparation caller's responsibility.
        graph = Grid([[0, 0, 9, 9], [9, 9, 9, -1]])
        self.assert_matches_reference(graph, (0, 0), (0, 1))
        self.assertEqual(_find_path_core(graph, (0, 0), (0, 1)).status, "success")

    def test_endpoints_and_arbitrary_integer_costs(self):
        graph = Grid([[0, 0]])
        for start, goal in (((0, 0), (0, 0)), ((-1, 0), (0, 1)), ((0, 0), (0, 2))):
            for cancel_after in (None, 1, 2, 3):
                with self.subTest(start=start, goal=goal, cancel_after=cancel_after):
                    self.assert_matches_reference(
                        graph, start, goal, cancel_after=cancel_after
                    )
        graph = Grid([[0, 2**100, 2**100]])
        self.assert_matches_reference(graph, (0, 0), (0, 2))
        self.assertEqual(_find_path_core(graph, (0, 0), (0, 2)).cost, 2**101)

    def test_discovered_pixels_avoid_repeated_cost_and_mask_reads(self):
        graph = CountingGrid([[1] * 9 for _ in range(9)])
        valid = CountingGrid([[True] * 9 for _ in range(9)])
        start, goal = (4, 0), (4, 8)
        result = _find_path_core(graph, start, goal, valid=valid)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(graph.reads), result.profile_stats["nodes"])
        self.assertGreater(len(graph.reads), len(result.path))
        for point, count in graph.reads.items():
            # The goal is validated as an endpoint and again on first discovery.
            expected = 2 if point == goal else 1
            self.assertEqual(count, expected, point)
            self.assertEqual(valid.reads[point], expected, point)
