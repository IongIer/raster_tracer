"""QGIS-independent, exact four-neighbor minimum-color-cost search."""

import heapq
import itertools
import time
from dataclasses import dataclass

MAX_DISCOVERED_NODES = 1_000_000
MAX_FRONTIER_ENTRIES = 1_000_000


@dataclass(frozen=True)
class FindPathCoreResult:
    path: object = None
    cost: object = None
    profile_stats: object = None
    status: str = "no_path"

    @property
    def cancelled(self):
        return self.status == "cancelled"


def get_neighbors(height, width, point):
    i, j = point
    # Fixed order plus a sequence number makes equal-cost paths reproducible.
    for row, col in ((i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)):
        if 0 <= row < height and 0 <= col < width:
            yield row, col


def _find_path_core(
    graph,
    start,
    goal,
    cancel_cb=None,
    valid=None,
    max_nodes=MAX_DISCOVERED_NODES,
    max_frontier=MAX_FRONTIER_ENTRIES,
):
    """Minimize entered-pixel cost (start costs zero), within this exact graph.

    Manhattan distance breaks ties only. Negative/nonintegral costs are invalid;
    callers preparing arrays validate the entire graph before publishing it.
    Python integers avoid overflow in cumulative costs.

    Costs and validity must stay unchanged during the search. Every incoming
    edge to a pixel adds the same nonnegative cost, and predecessor distances
    are popped in increasing order. Its first discovery is therefore optimal;
    parents can also track discovered pixels without a distance dictionary.
    """
    started = time.perf_counter()
    parents = {}

    def finish(status, path=None, cost=None):
        return FindPathCoreResult(
            path,
            cost,
            {
                "duration": time.perf_counter() - started,
                "nodes": len(parents),
            },
            status,
        )

    if cancel_cb and cancel_cb():
        return finish("cancelled")
    height, width = graph.shape
    for point in (start, goal):
        if not (0 <= point[0] < height and 0 <= point[1] < width):
            return finish("invalid_input")
        if valid is not None and not valid[point]:
            return finish("invalid_input")
        try:
            value = graph[point]
            if int(value) != value or int(value) < 0:
                return finish("invalid_input")
        except (ValueError, OverflowError, TypeError):
            return finish("invalid_input")
    if max_nodes < 1 or max_frontier < 1:
        return finish("resource_limit")
    sequence = itertools.count()
    frontier = [(0, 0, next(sequence), start)]
    parents[start] = None
    while frontier:
        if cancel_cb and cancel_cb():
            return finish("cancelled")
        queued_cost, _, _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()
            return finish("success", path, queued_cost)
        for neighbor in get_neighbors(height, width, current):
            # Discovered pixels already passed validation; only new pixels
            # need their mask and cost read.
            if neighbor in parents:
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
            candidate = queued_cost + step
            if len(parents) >= max_nodes or len(frontier) >= max_frontier:
                return finish("resource_limit")
            parents[neighbor] = current
            distance = abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
            heapq.heappush(frontier, (candidate, distance, next(sequence), neighbor))
    return finish("no_path")


def FindPathFunction(graph, start, goal):
    result = _find_path_core(graph, start, goal)
    return result.path, result.cost
