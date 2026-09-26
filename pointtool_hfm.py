"""Enhanced tracing with HFM24, using only NumPy and Python.

The input graph already contains the coarse search's soft dark-ink costs.
All returned coordinates are native pixel centers in (row, column) order.
"""

import math
import time

import numpy as np

from .exceptions import ResourceLimitError, TraceCancelled
from .hfm_backtrace import CACHE_RESERVE_BYTES, PATH_RESERVE_BYTES, backtrace
from .hfm_march import solve
from .pointtool_raster import CURSOR_RESERVE_BYTES, MAX_ARRAY_BYTES

HEADINGS = 16
PADDING = 20
MAX_STATES = 2_000_000
HEAP_RESERVE_BYTES = 64 * 1024 * 1024
MAX_PATH_POINTS = 20_000
IO_CHUNK = 16_384
TIMEOUT = 30.0


def _check(cancel, deadline):
    if cancel():
        raise TraceCancelled()
    if time.perf_counter() >= deadline:
        raise ResourceLimitError("HFM refinement exceeded its 30-second time limit")


def _point(point, shape):
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid HFM endpoint")
    if np.any(values != np.floor(values)):
        raise ValueError("HFM endpoints must be integer pixel centers")
    row, column = (int(value) for value in values)
    if not (0 <= row < shape[0] and 0 <= column < shape[1]):
        raise ValueError("HFM endpoint is outside the raster")
    return row, column


def _validate_path(path, valid, cancel, deadline):
    """Test every crossed pixel, including narrow crossings near grid corners."""
    if len(path) > MAX_PATH_POINTS:
        raise ResourceLimitError("HFM path contains too many points")
    if len(path) == 0 or not np.all(np.isfinite(path)):
        raise ValueError("HFM returned an empty or nonfinite path")
    height, width = valid.shape

    def check_points(points):
        cells = np.floor(points + 0.5).astype(np.int64)
        if (
            np.any(cells[:, 0] < 0)
            or np.any(cells[:, 0] >= height)
            or np.any(cells[:, 1] < 0)
            or np.any(cells[:, 1] >= width)
            or not np.all(valid[cells[:, 0], cells[:, 1]])
        ):
            raise ValueError("HFM path crosses an invalid raster pixel")

    check_points(path)
    _check(cancel, deadline)
    # A rectangle of valid pixels is convex, so in-bounds vertices suffice.
    if np.all(valid):
        return
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        if index % 64 == 0:
            _check(cancel, deadline)
        delta = end - start
        times = [np.array([0.0, 1.0])]
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-12:
                continue
            low, high = sorted((start[axis], end[axis]))
            crossings = np.arange(math.floor(low + 0.5), math.floor(high + 0.5)) + 0.5
            if crossings.size:
                times.append((crossings - start[axis]) / delta[axis])
        ordered = np.sort(np.concatenate(times))
        midpoints = (ordered[:-1] + ordered[1:]) * 0.5
        check_points(start + midpoints[:, None] * delta)
        # Include cells touched exactly at a grid corner; a diagonal must not
        # pass between two masked pixels merely because the contact is brief.
        if len(ordered) > 2:
            contacts = start + ordered[1:-1, None] * delta
            for offset in ((-1e-9, -1e-9), (-1e-9, 1e-9), (1e-9, -1e-9), (1e-9, 1e-9)):
                check_points(contacts + offset)
    _check(cancel, deadline)


def _resample(path):
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-12]
    path, lengths = path[keep], lengths[keep]
    if lengths[-1] <= 1e-12:
        return path[:1]
    count = max(2, int(math.ceil(lengths[-1])) + 1)
    if count > MAX_PATH_POINTS:
        raise ResourceLimitError("HFM path exceeds the output length limit")
    samples = np.linspace(0.0, lengths[-1], count)
    result = np.column_stack(
        [np.interp(samples, lengths, path[:, axis]) for axis in (0, 1)]
    )
    result[0], result[-1] = path[0], path[-1]
    return result


def _require_success(result, phase):
    status = result["status"]
    if status == "cancelled":
        raise TraceCancelled()
    if status in ("timeout", "resource_limit"):
        raise ResourceLimitError(f"HFM {phase} stopped: {status}")
    if status != "success":
        raise ValueError(f"HFM {phase} failed: {status}")


def refine_path(
    graph,
    valid,
    start_rc,
    goal_rc,
    coarse_path_rc,
    cancel=lambda: False,
    *,
    live_bytes=0,
):
    """Refine a successful coarse trace; failures are explicit, never a fallback.

    The conservative allocation estimate includes the retained caller snapshot,
    the larger of marching or backtracking allocations, plus scalar buffers. It is
    an allocation budget, not a guarantee about whole-process RSS or OS caches.
    """
    began = time.perf_counter()
    deadline = time.perf_counter() + TIMEOUT
    _check(cancel, deadline)
    graph, valid = np.asarray(graph), np.asarray(valid)
    if graph.ndim != 2 or valid.shape != graph.shape or valid.dtype != np.bool_:
        raise ValueError(
            "HFM requires a two-dimensional graph and Boolean validity mask"
        )
    start, goal = _point(start_rc, graph.shape), _point(goal_rc, graph.shape)
    if not valid[start] or not valid[goal]:
        raise ValueError("HFM endpoint is an invalid raster pixel")
    timings = {
        "hfm_total": 0.0,
        "hfm_preparation": 0.0,
        "hfm_marching": 0.0,
        "hfm_backtrace": 0.0,
        "hfm_states": 0,
        "hfm_attempts": 0,
        "hfm_backtracking": "direct",
    }
    # The official backtracer has no interior samples for neighboring centers.
    if max(abs(start[0] - goal[0]), abs(start[1] - goal[1])) <= 1:
        direct = np.asarray([start] if start == goal else [start, goal], dtype=float)
        _validate_path(direct, valid, cancel, deadline)
        timings["hfm_total"] = time.perf_counter() - began
        return tuple(map(tuple, direct)), timings
    if len(coarse_path_rc) == 0:
        raise ValueError("HFM refinement requires a successful coarse path")
    top = bottom = start[0]
    left = right = start[1]
    for offset in range(0, len(coarse_path_rc), 4096):
        _check(cancel, deadline)
        points = np.asarray(coarse_path_rc[offset : offset + 4096], dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
            or np.any(points != np.floor(points))
            or np.any(points < 0)
            or np.any(points[:, 0] >= graph.shape[0])
            or np.any(points[:, 1] >= graph.shape[1])
        ):
            raise ValueError("Invalid HFM coarse path coordinates")
        low, high = points.min(axis=0), points.max(axis=0)
        top, bottom = min(top, int(low[0])), max(bottom, int(high[0]))
        left, right = min(left, int(low[1])), max(right, int(high[1]))
    top, bottom = (
        max(0, min(top, goal[0]) - PADDING),
        min(graph.shape[0], max(bottom, goal[0]) + PADDING + 1),
    )
    left, right = (
        max(0, min(left, goal[1]) - PADDING),
        min(graph.shape[1], max(right, goal[1]) + PADDING + 1),
    )
    pixels = (bottom - top) * (right - left)
    states = pixels * HEADINGS
    if states > MAX_STATES:
        raise ResourceLimitError("HFM local window exceeds the 2-million-state limit")
    if live_bytes < 0 or not math.isfinite(live_bytes):
        raise ValueError("Invalid HFM retained allocation estimate")
    # March coefficients and its heap die before extraction builds flow caches.
    marching_peak = states * 45 + HEAP_RESERVE_BYTES
    backtrace_peak = states * 13 + CACHE_RESERVE_BYTES + PATH_RESERVE_BYTES
    estimated = (
        live_bytes
        + max(marching_peak, backtrace_peak)
        + pixels * 48
        + CURSOR_RESERVE_BYTES
    )
    if estimated > MAX_ARRAY_BYTES:
        raise ResourceLimitError("HFM refinement exceeds the retained array budget")
    timings["hfm_states"] = states
    timings["hfm_estimated_bytes"] = estimated
    local_valid = valid[top:bottom, left:right]
    cost = np.array(graph[top:bottom, left:right], dtype=np.float64, order="C")
    flat = cost.ravel()
    for offset in range(0, flat.size, IO_CHUNK):
        _check(cancel, deadline)
        chunk = flat[offset : offset + IO_CHUNK]
        if np.any(~np.isfinite(chunk)) or np.any(chunk < 0):
            raise ValueError("HFM requires finite nonnegative appearance costs")
        chunk += 1.0
        chunk /= 2048.0
    local_start = np.array([start[1] - left, start[0] - top], dtype=float)
    local_goal = np.array([goal[1] - left, goal[0] - top], dtype=float)
    solved = solve(
        cost,
        local_start,
        local_goal,
        local_valid,
        cancel=cancel,
        timeout=max(0.0, deadline - time.perf_counter()),
    )
    _check(cancel, deadline)
    _require_success(solved, "marching")
    timings["hfm_preparation"] = solved["timings"]["preparation"]
    timings["hfm_marching"] = solved["timings"]["marching"]
    timings["hfm_accepted"] = solved["metadata"]["accepted_states"]
    timings["hfm_updates"] = solved["metadata"]["updates"]
    traced = backtrace(
        solved,
        cancel=cancel,
        timeout=max(0.0, deadline - time.perf_counter()),
        endpoint_tolerance=1.5,
        max_steps=MAX_PATH_POINTS,
    )
    _check(cancel, deadline)
    _require_success(traced, "backtracking")
    timings["hfm_backtrace"] = traced["timings"]["backtrace_total"]
    timings["hfm_attempts"] = 2 if traced["metadata"]["fallback_used"] else 1
    timings["hfm_backtracking"] = traced["metadata"]["solver"]
    timings["hfm_endpoint_residual"] = traced["metadata"]["endpoint_residual"]
    # The backtracer has already checked residuals and exact endpoint connectors.
    # Validate again after conversion and one-pixel arclength resampling.
    path = np.asarray(traced["path_xy"], dtype=np.float64)[:, ::-1].copy()
    if len(path) > MAX_PATH_POINTS:
        raise ResourceLimitError("HFM path exceeds the output point limit")
    if not np.array_equal(path[0], local_start[::-1]) or not np.array_equal(
        path[-1], local_goal[::-1]
    ):
        raise ValueError("HFM backtracking did not preserve exact endpoints")
    _validate_path(path, local_valid, cancel, deadline)
    path = _resample(path)
    _validate_path(path, local_valid, cancel, deadline)
    path += np.array([top, left])
    _check(cancel, deadline)
    timings["hfm_total"] = time.perf_counter() - began
    return tuple(map(tuple, path)), timings
