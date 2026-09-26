"""NumPy/Python port of official first-order HFM geodesic extraction.

Adapted from HamiltonFastMarching GeodesicODESolver.hxx,
GeodesicDiscreteSolver.hxx and EulerianStencil.hxx, upstream commit
420533c742d46f1d17145fa31b8167529be7182d. Original algorithm copyright
Jean-Marie Mirebeau; GPL v3 or later. No executable or compiled extension is
used here. Fixed Elastica2 stencils are supplied by the independent marcher.

Coordinates returned are native pixel-center XY, with y increasing down.
The intermediate grid uses the upstream index-center convention (+0.5).
Only scalar image cost, first order, and all-heading point seeds are supported.
"""

import heapq
import math
import time
from collections import deque

import numpy as np

from .hfm_stencils import STENCILS

MAX_FLOW_NODES = 8192
MAX_DISCRETE_FRONTIER = 8192
MAX_PATH_POINTS = 20000
# Conservative separate reserves for flow cache/frontier and path temporaries.
CACHE_RESERVE_BYTES = 40 * 1024 * 1024
PATH_RESERVE_BYTES = 16 * 1024 * 1024


class _Stopped(Exception):
    pass


class _Flow:
    def __init__(self, result, start, cancel, deadline):
        self.values = np.asarray(result["values"])
        self.active = np.asarray(result["active"])
        self.cost = np.asarray(result["cost"])
        self.h, self.w, self.n = self.values.shape
        self.valid = np.asarray(result.get("valid", np.ones((self.h, self.w), bool)))
        self.stencils = result.get("stencils", STENCILS)
        self.cancel = cancel or (lambda: False)
        self.deadline = deadline
        self.cache = {}
        self.visits = {}
        self.target = self._target(start)
        self.calls = 0

    def check(self):
        if self.cancel():
            raise _Stopped("cancelled")
        if time.perf_counter() >= self.deadline:
            raise _Stopped("timeout")

    def _target(self, start):
        # All angles are seeded and walls depend only on XY, so the upstream
        # 26-neighbor lifted BFS reduces exactly to this eight-neighbor BFS.
        sx, sy = (int(round(v)) for v in start)
        target = {(sx, sy): 0}
        queue = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            distance = target[x, y] + 1
            if distance > 6:
                continue
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if (
                        0 <= xx < self.w
                        and 0 <= yy < self.h
                        and self.valid[yy, xx]
                        and (xx, yy) not in target
                    ):
                        target[xx, yy] = distance
                        queue.append((xx, yy))
        return target

    def at(self, x, y, k):
        key = (x, y, k)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if len(self.cache) >= MAX_FLOW_NODES:
            raise _Stopped("resource_limit")
        active = int(self.active[y, x, k])
        if not active:
            data = (float(self.values[y, x, k]), 0.0, (0.0, 0.0, 0.0), ())
            self.cache[key] = data
            return data
        multiplier = 1.0 / float(self.cost[y, x])
        multiplier *= multiplier
        neighbors = []
        minimum = math.inf
        for i, (dx, dy, dk, base) in enumerate(self.stencils[k]):
            if not (active & (1 << i)):
                continue
            xx, yy, kk = x + dx, y + dy, (k + dk) % self.n
            if not (0 <= xx < self.w and 0 <= yy < self.h):
                raise ValueError("Active HFM neighbor lies outside grid")
            value = float(self.values[yy, xx, kk])
            minimum = min(minimum, value)
            neighbors.append((dx, dy, dk, base * multiplier, value))
        a, b, c = 0.0, 0.0, -1.0
        for _, _, _, weight, value in neighbors:
            v = value - minimum
            a += weight
            b += weight * v
            c += (weight * v) * v
        if not a or not math.isfinite(minimum):
            data = (math.inf, 0.0, (0.0, 0.0, 0.0), ())
        else:
            value = minimum + (b + math.sqrt(max(0.0, b * b - a * c))) / a
            fx, fy, fk, width = 0.0, 0.0, 0.0, 0.0
            discrete = []
            for dx, dy, dk, weight, neighbor in neighbors:
                fw = weight * max(0.0, value - neighbor)
                fx += fw * dx
                fy += fw * dy
                fk += fw * dk
                width += fw
                discrete.append((dx, dy, dk, fw))
            data = (value, width / a, (fx, fy, fk), tuple(discrete))
        self.cache[key] = data
        return data

    def interpolated(self, p):
        self.calls += 1
        if self.calls % 64 == 1:
            self.check()
        ix, iy, ik = math.floor(p[0]), math.floor(p[1]), math.floor(p[2])
        wx, wy, wk = p[0] - ix - 0.5, p[1] - iy - 0.5, p[2] - ik - 0.5
        sx, sy, sk = (1 if v > 0 else -1 for v in (wx, wy, wk))
        wx, wy, wk = abs(wx), abs(wy), abs(wk)
        entries = []
        min_tol, max_stat, min_value, min_width = 32767, 0, math.inf, 0.0
        for bit in range(8):
            x = ix + (sx if bit & 1 else 0)
            y = iy + (sy if bit & 2 else 0)
            k = (ik + (sk if bit & 4 else 0)) % self.n
            if not (0 <= x < self.w and 0 <= y < self.h and self.valid[y, x]):
                continue
            min_tol = min(min_tol, self.target.get((x, y), 32767))
            key = (x, y, k)
            count = self.visits.get(key, 0)
            max_stat = max(max_stat, count)
            self.visits[key] = count + 1
            data = self.at(x, y, k)
            weight = (
                (wx if bit & 1 else 1 - wx)
                * (wy if bit & 2 else 1 - wy)
                * (wk if bit & 4 else 1 - wk)
            )
            entries.append((weight, data))
            if weight > 0 and data[0] < min_value:
                min_value, min_width = data[0], data[1]
        if min_value == math.inf:
            return (0.0, 0.0, 0.0), max_stat, min_tol
        threshold = min_value + 4.0 * min_width
        fx, fy, fk, weight_sum = 0.0, 0.0, 0.0, 0.0
        for weight, data in entries:
            if weight > 0 and data[0] <= threshold:
                flow = data[2]
                fx += weight * flow[0]
                fy += weight * flow[1]
                fk += weight * flow[2]
                weight_sum += weight
        if weight_sum:
            fx, fy, fk = fx / weight_sum, fy / weight_sum, fk / weight_sum
        return (fx, fy, fk), max_stat, min_tol


def _ode(flow, tip, max_steps):
    path = [tip]
    history = deque([32767] * math.ceil(math.sqrt(3.0) / 0.25))
    stationarity_max = int(6 * math.sqrt(3.0) / 0.25)
    for _ in range(max_steps):
        p = path[-1]
        vector, stationary, target = flow.interpolated(p)
        old_target = history.popleft()
        history.append(target)
        if old_target < target:
            return path, "moving_away_from_seed"
        if stationary > stationarity_max:
            return path, "stationary"
        norm = math.sqrt(sum(v * v for v in vector))
        if not norm:
            return path, "zero_flow"
        q = tuple(p[j] + (0.5 * 0.25 * vector[j]) / norm for j in range(3))
        vector = flow.interpolated(q)[0]
        norm = math.sqrt(sum(v * v for v in vector))
        if not norm:
            return path, "zero_halfstep_flow"
        path.append(tuple(p[j] + (0.25 * vector[j]) / norm for j in range(3)))
    raise _Stopped("resource_limit")


def _discrete(flow, tip, max_steps):
    path = [tip]
    volume_bound = 5.0 * 1.3**3
    popped = 0
    for restart in range(50):
        p = path[-1]
        idx = tuple(math.floor(v) for v in p)
        diff = tuple(p[j] - idx[j] - 0.5 for j in range(3))
        weights = {}
        queue = []
        # w, wx, wy, wk, wxx, wxy, wxk, wyy, wyk, wkk.
        sums = [0.0] * 10

        def update(weight, ind):
            x, y, k = ind
            terms = (1, x, y, k, x * x, x * y, x * k, y * y, y * k, k * k)
            for j, v in enumerate(terms):
                sums[j] += weight * v

        def add(value, ind, weight):
            key = (value, *ind)
            if key not in weights:
                if len(weights) >= MAX_DISCRETE_FRONTIER:
                    raise _Stopped("resource_limit")
                heapq.heappush(queue, tuple(-v for v in key))
                weights[key] = weight
            else:
                weights[key] += weight
            update(weight, ind)

        for bit in range(8):
            ind = tuple(
                idx[j] + ((1 if diff[j] > 0 else -1) if bit & (1 << j) else 0)
                for j in range(3)
            )
            x, y, k = ind
            if not (0 <= x < flow.w and 0 <= y < flow.h):
                continue
            weight = math.prod(
                abs(diff[j]) if bit & (1 << j) else 1 - abs(diff[j]) for j in range(3)
            )
            add(float(flow.values[y, x, k % flow.n]), ind, weight)
        if not sums[0]:
            return path, "empty_tip", restart, popped
        avg = tuple(sums[j] / sums[0] for j in (1, 2, 3))
        base, steps, step_norm = avg, [], 0.0
        needs_restart = False
        while queue and sums[0] > 1e-5:
            if len(path) >= MAX_PATH_POINTS or len(steps) >= MAX_PATH_POINTS:
                raise _Stopped("resource_limit")
            popped += 1
            if popped % 128 == 1:
                flow.check()
            if popped > max_steps * 32:
                raise _Stopped("resource_limit")
            key = tuple(-v for v in heapq.heappop(queue))
            weight = weights.pop(key)
            value, x, y, k = key
            x, y, k = int(x), int(y), int(k)
            if weight == 0:
                continue
            update(-weight, (x, y, k))
            discrete = flow.at(x, y, k % flow.n)[3]
            total = sum(item[3] for item in discrete)
            if not total:
                continue
            for dx, dy, dk, fw in discrete:
                child_weight = weight * fw / total
                ind = (x + dx, y + dy, k + dk)
                child_value = float(flow.values[ind[1], ind[0], ind[2] % flow.n])
                child_key = (child_value, *ind)
                if child_key not in weights and child_weight < sums[0] * 0.001:
                    continue
                add(child_value, ind, child_weight)
            if sums[0] <= 0:
                path.append(tuple(v + 0.5 for v in avg))
                return path, "empty_mass", restart, popped
            old_avg = avg
            avg = tuple(sums[j] / sums[0] for j in (1, 2, 3))
            step = tuple(avg[j] - old_avg[j] for j in range(3))
            steps.append((value, step))
            step_norm += math.sqrt(sum(v * v for v in step))
            if len(weights) == 1:
                path.append(tuple(v + 0.5 for v in avg))
                base, steps, step_norm = avg, [], 0.0
                continue
            if step_norm < 0.25 and len(weights) > 1:
                continue
            value_inc = steps[0][0] - value
            if value_inc <= 0:
                continue
            delta = value_inc / 3.0
            p = list(base)
            for step_value, increment in steps:
                factor = (step_value - value + delta) / (value_inc + delta)
                for j in range(3):
                    p[j] += increment[j] * factor
            path.append(tuple(v + 0.5 for v in p))
            base, steps, step_norm = avg, [], 0.0
            a, b, c, d, e, f = (sums[j] / sums[0] for j in range(4, 10))
            mx, my, mk = avg
            a, b, c = a - mx * mx + 1, b - mx * my, c - mx * mk
            d, e, f = d - my * my + 1, e - my * mk, f - mk * mk + 1
            determinant = (
                a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
            )
            if determinant > volume_bound * volume_bound:
                path.append(tuple(v + 0.5 for v in avg))
                needs_restart = True
                break
        if not needs_restart:
            return path, "mass_exhausted", restart, popped
        if restart + 1 >= 5:
            volume_bound *= 1.3
    raise _Stopped("resource_limit")


def _validate(path, start, goal, valid, tolerance, check=lambda: None):
    if len(path) < 1 or not np.all(np.isfinite(path)):
        return False, math.inf, "nonfinite_path"
    residual = max(math.dist(path[0], start), math.dist(path[-1], goal))
    if residual > tolerance:
        return False, residual, "incomplete"
    # Check every crossed cell and exact corner contact, including endpoint
    # extension. Fixed-distance samples can miss arbitrarily short crossings.
    points = np.asarray([tuple(start), *path, tuple(goal)], dtype=np.float64)
    h, w = valid.shape

    def points_valid(p):
        cells = np.floor(p + 0.5).astype(np.int64)
        x, y = cells[:, 0], cells[:, 1]
        return bool(
            np.all(x >= 0)
            and np.all(x < w)
            and np.all(y >= 0)
            and np.all(y < h)
            and np.all(valid[y, x])
        )

    check()
    if not points_valid(points):
        return False, residual, "invalid_mask_crossing"
    if np.all(valid):
        return True, residual, "success"
    for index, (p, q) in enumerate(zip(points, points[1:])):
        if index % 64 == 0:
            check()
        delta = q - p
        times = [np.array([0.0, 1.0])]
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-12:
                continue
            low, high = sorted((p[axis], q[axis]))
            crossings = np.arange(math.floor(low + 0.5), math.floor(high + 0.5)) + 0.5
            if crossings.size:
                times.append((crossings - p[axis]) / delta[axis])
        ordered = np.sort(np.concatenate(times))
        midpoints = (ordered[:-1] + ordered[1:]) * 0.5
        if not points_valid(p + midpoints[:, None] * delta):
            return False, residual, "invalid_mask_crossing"
        if len(ordered) > 2:
            contacts = p + ordered[1:-1, None] * delta
            for offset in ((-1e-9, -1e-9), (-1e-9, 1e-9), (1e-9, -1e-9), (1e-9, 1e-9)):
                if not points_valid(contacts + offset):
                    return False, residual, "invalid_mask_crossing"
    check()
    return True, residual, "success"


def backtrace(
    result,
    start_xy=None,
    goal_xy=None,
    *,
    solver="ODE",
    fallback_discrete=True,
    cancel=None,
    timeout=10.0,
    endpoint_tolerance=1.5,
    max_steps=100000,
):
    """Extract a genuine HFM geodesic from the marcher arrays.

    raw_path_xy preserves official termination; path_xy pins exact endpoints
    only after the residual and validity checks pass. Failure never substitutes
    a coarse/Dijkstra path. Discrete retry consumes the same deadline.
    """
    began = time.perf_counter()
    max_steps = min(int(max_steps), MAX_PATH_POINTS - 1)
    start = tuple(start_xy if start_xy is not None else result["start_xy"])
    goal = tuple(goal_xy if goal_xy is not None else result["goal_xy"])
    metadata = {
        "method": "official HFM first-order geodesic port",
        "solver": solver,
        "fallback_used": False,
        "max_flow_nodes": MAX_FLOW_NODES,
        "max_discrete_frontier": MAX_DISCRETE_FRONTIER,
        "max_path_points": MAX_PATH_POINTS,
        "cache_reserve_bytes": CACHE_RESERVE_BYTES,
        "path_reserve_bytes": PATH_RESERVE_BYTES,
    }
    timings = {}
    path, raw = [], []
    try:
        flow = _Flow(result, start, cancel, began + timeout)
        flow.check()
        y, x, _ = result["goal_state"]
        # HFMInterface chooses the last heading attaining the minimum, which
        # need not be the first equal-valued heading popped by the marcher.
        goal_values = flow.values[y, x]
        k = flow.n - 1 - int(np.argmin(goal_values[::-1]))
        metadata["goal_heading"] = k
        tip = (float(goal[0]) + 0.5, float(goal[1]) + 0.5, float(k) + 0.5)
        prepared = time.perf_counter()
        timings["backtrace_prepare"] = prepared - began
        for method in (
            [solver, "Discrete"] if solver == "ODE" and fallback_discrete else [solver]
        ):
            method_began = time.perf_counter()
            if method == "ODE":
                points, reason = _ode(flow, tip, max_steps)
            elif method == "Discrete":
                points, reason, restarts, popped = _discrete(flow, tip, max_steps)
                metadata.update(discrete_restarts=restarts, discrete_popped=popped)
            else:
                raise ValueError("solver must be ODE or Discrete")
            timings["backtrace_" + method.lower()] = time.perf_counter() - method_began
            raw = [(p[0] - 0.5, p[1] - 0.5) for p in reversed(points)]
            success, residual, status = _validate(
                raw, start, goal, flow.valid, endpoint_tolerance, flow.check
            )
            metadata[method.lower() + "_termination"] = reason
            metadata[method.lower() + "_endpoint_residual"] = residual
            if success:
                path = list(raw)
                if len(path) == 1 and start != goal:
                    path = [start, goal]
                else:
                    path[0], path[-1] = start, goal
                metadata["solver"] = method
                metadata["fallback_used"] = method != solver
                break
        flow.check()
        metadata.update(
            endpoint_residual=residual,
            flow_cache_states=len(flow.cache),
            interpolation_calls=flow.calls,
        )
    except _Stopped as exc:
        status = str(exc)
        path = []
    timings["backtrace_total"] = time.perf_counter() - began
    return {
        "status": status,
        "path_xy": path,
        "raw_path_xy": raw,
        "timings": timings,
        "metadata": metadata,
    }
