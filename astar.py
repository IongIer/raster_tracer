'''
Module performs searching of the best path on 2D grid between
two given points by using famous A* method.
Code is based on example from
https://www.redblobgames.com/pathfinding/a-star/implementation.html
'''

import heapq
import io
import os
import time
import cProfile
import pstats
from collections import namedtuple

from qgis.core import QgsTask, QgsMessageLog, Qgis


PROFILE_ENABLED = os.environ.get("RASTER_TRACER_PROFILE", "0") == "1"


FindPathCoreResult = namedtuple(
    "FindPathCoreResult",
    ["path", "cost", "profile_stats", "cancelled"],
)

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def get_neighbors(size_i, size_j, ij):
    """ returns possible neighbors of a numpy cell """
    i,j = ij
    neighbors = set()
    if i>0:
        neighbors.add((i-1, j))
    if j>0:
        neighbors.add((i, j-1))
    if i<size_i-1:
        neighbors.add((i+1, j))
    if j<size_j-1:
        neighbors.add((i, j+1))
    return neighbors

def get_cost(array, current, next):
    return array[next]

def _finalize_profile(profiler, start_time, explored_nodes, cancelled):
    if profiler is None or start_time is None:
        return None
    profiler.disable()
    duration = time.perf_counter() - start_time
    if cancelled:
        profile_output = "Task cancelled before completion"
    else:
        stats_stream = io.StringIO()
        pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats('cumtime').print_stats(15)
        profile_output = stats_stream.getvalue()
    return {
        "duration": duration,
        "nodes": explored_nodes,
        "profile": profile_output,
    }


def _log_profile_stats(label, stats):
    if not stats:
        return
    QgsMessageLog.logMessage(
        f"[profiling] {label} duration={stats['duration']:.3f}s nodes={stats['nodes']}",
        "RasterTracer",
        Qgis.Info,
    )
    profile_text = stats.get("profile")
    if profile_text:
        QgsMessageLog.logMessage(
            profile_text,
            "RasterTracer",
            Qgis.Info,
        )


def _find_path_core(graph, start, goal, cancel_cb=None):
    profiler = None
    start_time = None
    if PROFILE_ENABLED:
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.perf_counter()

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    size_i, size_j = graph.shape

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in get_neighbors(size_i, size_j, current):
            if cancel_cb is not None and cancel_cb():
                profile_stats = _finalize_profile(
                    profiler,
                    start_time,
                    len(cost_so_far),
                    cancelled=True,
                )
                return FindPathCoreResult(None, None, profile_stats, True)

            new_cost = cost_so_far[current] + get_cost(graph, current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    if goal in came_from:
        path = reconstruct_path(came_from, start, goal)
        cost = cost_so_far[goal]
    else:
        path = None
        cost = None

    profile_stats = _finalize_profile(
        profiler,
        start_time,
        len(cost_so_far),
        cancelled=False,
    )

    return FindPathCoreResult(path, cost, profile_stats, False)


def FindPathFunction(graph, start, goal):
    result = _find_path_core(graph, start, goal)
    if PROFILE_ENABLED and result.profile_stats:
        _log_profile_stats("FindPathFunction", result.profile_stats)
    return result.path, result.cost


class FindPathTask(QgsTask):
    '''
    Implementation of QGIS QgsTask
    for searching of the path on the background.
    '''


    def __init__(self, graph, start, goal, callback, vlayer):
        '''
        Receives: graph - 2D grid of points
        start - coordinates of start point
        goal - coordinates of finish point
        callback - function to call after finishing tracing
        vlayer - vector layer for callback function
        '''

        super().__init__(
            'Task for finding path on 2D grid for raster_tracer',
            QgsTask.CanCancel
                )
        self.graph = graph
        self.start = start
        self.goal = goal
        self.path = None
        self.cost = None
        self.callback = callback
        self.vlayer = vlayer
        self.profile_stats = None

    def run(self):
        '''
        Actually trace over 2D grid,
        i.e. finding the best path from start to goal
        '''

        graph = self.graph
        start = self.start
        goal = self.goal

        result = _find_path_core(graph, start, goal, cancel_cb=self.isCanceled)
        self.profile_stats = result.profile_stats
        self.path = result.path
        self.cost = result.cost
        if result.cancelled:
            return False
        return True

    def finished(self, result):
        '''
        Call callback function if self.run was successful
        '''

        if result:
            self.callback(self.path, self.vlayer)

        if PROFILE_ENABLED and self.profile_stats:
            _log_profile_stats("FindPathTask", self.profile_stats)
            self.profile_stats = None


    def cancel(self):
        '''
        Executed when run catches cancel signal.
        Terminates the QgsTask.
        '''

        super().cancel()


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path
