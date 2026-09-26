"""Pure Python first-order Hamiltonian Fast Marching for fixed Elastica2.

Port of HamiltonFastMarching.hxx, EulerianStencil.hxx, HFM_StencilShare.hxx
and WallObstructionTest.h from Jean-Marie Mirebeau's HamiltonFastMarching,
commit 420533c742d46f1d17145fa31b8167529be7182d (GPL-3.0-or-later).
This is the published quadratic upwind scheme, not graph edge turn penalties.

The checked-in numerical stencils fix xi=24, eps=.1, 16 headings, gridScale=1.
Only Python's standard library and NumPy execute at runtime. See
NOTICE.txt and LICENSE for upstream attribution and licensing.
"""

import heapq
import math
import time

import numpy as np

from .hfm_stencils import STENCILS

HEADINGS = 16
MAX_STATES = 2_000_000
MAX_HEAP = 1_000_000


def _walk(dx, dy, dt):
    """Native obstacle walk's intermediate XY offsets for an XYZ stencil."""
    vector = (dx, dy, dt)
    product = math.prod(abs(value) for value in vector if value)
    maximum = max(map(abs, vector))
    sign = tuple(1 if value > 0 else -1 if value < 0 else 0 for value in vector)
    next_crossing = [product // abs(value) if value else 2**60 for value in vector]
    increments = [2 * value for value in next_crossing]
    position = [0, 0, 0]
    points = []
    for index in range(1, maximum):
        threshold = 2 * index * product // maximum
        for axis in range(3):
            if next_crossing[axis] <= threshold:
                next_crossing[axis] += increments[axis]
                position[axis] += sign[axis]
        xy = (position[0], position[1])
        if xy not in points and xy != (0, 0) and xy != (dx, dy):
            points.append(xy)
    return points


def _reverse(height, width, has_walls):
    """Combine duplicate offsets while retaining every original active bit."""
    reverse = [[] for _ in range(HEADINGS)]
    center_x, center_y = width // 2, height // 2
    for heading, raw in enumerate(STENCILS):
        combined = {}
        for bit, (dx, dy, dt, weight) in enumerate(raw):
            # The official shared-stencil setup tests offsets at the spatial
            # center of the grid. Preserve its behavior on unusually small ROIs.
            if not (0 <= center_x + dx < width and 0 <= center_y + dy < height):
                continue
            key = (dx, dy, dt)
            previous_weight, previous_mask = combined.get(key, (0.0, 0))
            combined[key] = (previous_weight + weight, previous_mask | (1 << bit))
        for (dx, dy, dt), (weight, mask) in combined.items():
            if not weight:
                continue
            accepted_heading = (heading + dt) % HEADINGS
            linear = (-dy * width - dx) * HEADINGS + heading - accepted_heading
            walk = (
                tuple(y * width + x for x, y in _walk(-dx, -dy, -dt))
                if has_walls
                else ()
            )
            reverse[accepted_heading].append(
                (-dy, -dx, linear, weight, mask, walk, (dx, dy, dt))
            )
    return [
        tuple(item[:6] for item in sorted(items, key=lambda item: item[6]))
        for items in reverse
    ]


def _endpoint(point, width, height):
    if len(point) != 2 or not all(
        math.isfinite(value) and int(value) == value for value in point
    ):
        raise ValueError("Endpoints must be integer native pixel centers XY")
    x, y = map(int, point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Endpoint lies outside the metric")
    return x, y


def solve(cost_yx, start_xy, goal_xy, valid_yx=None, cancel=None, timeout=10):
    """Solve the same first-order fixed Elastica2 metric as official HFM.

    ``goal_xy=None`` marches the complete finite domain for numerical validation.
    Otherwise the first goal heading stops propagation before being accepted,
    matching official stopWhenAnyAccepted_Unoriented semantics. Returned arrays
    use YXtheta; active bits address the original 27 STENCILS entries.
    """
    began = time.perf_counter()
    deadline = began + timeout
    cost = np.ascontiguousarray(cost_yx, dtype=np.float64)
    if (
        cost.ndim != 2
        or min(cost.shape) < 1
        or np.any(~np.isfinite(cost))
        or np.any(cost <= 0)
    ):
        raise ValueError("HFM requires a finite, strictly positive YX cost field")
    height, width = cost.shape
    valid = (
        np.ones(cost.shape, dtype=bool)
        if valid_yx is None
        else np.ascontiguousarray(valid_yx, dtype=bool)
    )
    if valid.shape != cost.shape:
        raise ValueError("Validity dimensions do not match the cost")
    start_x, start_y = _endpoint(start_xy, width, height)
    goal = None if goal_xy is None else _endpoint(goal_xy, width, height)
    if not valid[start_y, start_x] or (
        goal is not None and not valid[goal[1], goal[0]]
    ):
        raise ValueError("HFM endpoint is masked")
    states = height * width * HEADINGS
    if states > MAX_STATES:
        return {
            "status": "resource_limit",
            "timings": {"total": time.perf_counter() - began},
            "states": states,
        }
    if cancel is not None and cancel():
        return {
            "status": "cancelled",
            "timings": {"total": time.perf_counter() - began},
        }
    values_array = np.full((height, width, HEADINGS), np.inf, dtype=np.float64)
    accepted_array = np.zeros(values_array.shape, dtype=bool)
    active_array = np.zeros(values_array.shape, dtype=np.uint32)
    coefficients = np.zeros((4, states), dtype=np.float64)
    coefficients[2] = -1.0
    speed_squared = np.reciprocal(cost)
    np.square(speed_squared, out=speed_squared)
    if np.any(~np.isfinite(speed_squared)) or np.any(speed_squared <= 0):
        raise ValueError("Cost scale cannot be represented by float64 HFM weights")
    # memoryview indexing yields Python scalars, avoiding NumPy scalar dispatch
    # in the several hundred thousand scalar updates of a typical local trace.
    values = memoryview(values_array).cast("B").cast("d")
    value_bits = memoryview(values_array).cast("B").cast("Q")
    accepted = memoryview(accepted_array).cast("B")
    active = memoryview(active_array).cast("B").cast("I")
    a, b, c, minimum = [memoryview(row).cast("B").cast("d") for row in coefficients]
    speed2 = memoryview(speed_squared).cast("B").cast("d")
    validity = memoryview(valid).cast("B")
    has_walls = not bool(np.all(valid))
    reverse = _reverse(height, width, has_walls)
    seed_pixel = start_y * width + start_x
    goal_pixel = -1 if goal is None else goal[1] * width + goal[0]
    queue = []
    index_bits = states.bit_length()
    index_mask = (1 << index_bits) - 1
    push, pop, sqrt = heapq.heappush, heapq.heappop, math.sqrt
    for heading in range(HEADINGS):
        index = seed_pixel * HEADINGS + heading
        values[index] = 0.0
        push(queue, index)
    prepared = time.perf_counter()
    pops = updates = accepted_count = 0
    peak_heap = len(queue)
    status, goal_state = "no_path", None
    while queue:
        index = pop(queue) & index_mask
        pops += 1
        if pops & 1023 == 1:
            if cancel is not None and cancel():
                status = "cancelled"
                break
            if time.perf_counter() >= deadline:
                status = "timeout"
                break
        if accepted[index]:
            continue
        value = values[index]
        pixel, heading = divmod(index, HEADINGS)
        row, column = divmod(pixel, width)
        if pixel == goal_pixel:
            status, goal_state = "success", (row, column, heading)
            break
        accepted[index] = 1
        accepted_count += 1
        for dr, dc, linear, base_weight, mask, walk in reverse[heading]:
            target_row, target_column = row + dr, column + dc
            if (
                target_row < 0
                or target_row >= height
                or target_column < 0
                or target_column >= width
            ):
                continue
            target = index + linear
            if accepted[target] or values[target] <= value:
                continue
            target_pixel = target >> 4
            if has_walls:
                if not validity[target_pixel]:
                    continue
                blocked = False
                for step in walk:
                    if not validity[pixel + step]:
                        blocked = True
                        break
                if blocked:
                    continue
            flag = active[target]
            if flag == 0:
                # Native seeds are finite with no active neighbors and cannot
                # be updated; their zero value already triggered the guard.
                minimum[target] = value
            active[target] = flag | mask
            weight = base_weight * speed2[target_pixel]
            shifted = value - minimum[target]
            weighted = weight * shifted
            aa = a[target] + weight
            bb = b[target] + weighted
            cc = c[target] + weighted * shifted
            a[target], b[target], c[target] = aa, bb, cc
            discriminant = bb * bb - aa * cc
            candidate = (
                minimum[target]
                + (bb + sqrt(discriminant if discriminant > 0 else 0.0)) / aa
            )
            values[target] = candidate
            # Positive IEEE754 encodings have the same order as their values.
            # A single integer saves tuple/float allocations for stale entries.
            push(queue, (value_bits[target] << index_bits) | target)
            updates += 1
        if len(queue) > peak_heap:
            peak_heap = len(queue)
            if peak_heap > MAX_HEAP:
                status = "resource_limit"
                break
    else:
        status = "success" if goal is None else "no_path"
    ended = time.perf_counter()
    return {
        "status": status,
        "values": values_array,
        "accepted": accepted_array,
        "active": active_array,
        "goal_state": goal_state,
        "start_xy": (start_x, start_y),
        "goal_xy": goal,
        "seed_states": [(start_y, start_x, heading) for heading in range(HEADINGS)],
        "cost": cost,
        "valid": valid,
        "stencils": STENCILS,
        "timings": {
            "preparation": prepared - began,
            "marching": ended - prepared,
            "total": ended - began,
        },
        "metadata": {
            "solver": "Python port of official first-order Elastica2 HFM",
            # pragma: allowlist nextline secret -- Upstream Git commit ID.
            "source_commit": "420533c742d46f1d17145fa31b8167529be7182d",
            "xi": 24,
            "eps": 0.1,
            "headings": HEADINGS,
            "states": states,
            "accepted_states": accepted_count,
            "updates": updates,
            "heap_pops": pops,
            "peak_heap": peak_heap,
            "array_bytes": values_array.nbytes
            + accepted_array.nbytes
            + active_array.nbytes
            + coefficients.nbytes
            + cost.nbytes
            + valid.nbytes
            + speed_squared.nbytes,
            "references_used": False,
        },
    }
