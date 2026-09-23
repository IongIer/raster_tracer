"""Endpoint-preserving numerical path postprocessing."""

from math import atan2, radians


def smooth(path, size=2):
    points = list(path)
    n = len(points)
    if n <= 2 or size <= 0:
        return points
    radius = min(int(size), max(1, (n - 1) // 4))
    xs, ys = [0.0], [0.0]
    for x, y in points:
        xs.append(xs[-1] + x)
        ys.append(ys[-1] + y)
    result = [points[0]]
    for i in range(1, n - 1):
        r = min(radius, i, n - 1 - i)
        lo, hi = i - r, i + r + 1
        result.append(((xs[hi] - xs[lo]) / (hi - lo), (ys[hi] - ys[lo]) / (hi - lo)))
    result.append(points[-1])
    return result


def simplify(path, tolerance=2):
    points = list(path)
    if len(points) <= 2:
        return points
    result = [points[0]]
    threshold = radians(tolerance)
    for point in points[1:]:
        if point == result[-1]:
            continue
        while len(result) >= 2:
            a, b = result[-2:]
            u = (b[0] - a[0], b[1] - a[1])
            v = (point[0] - b[0], point[1] - b[1])
            angle = abs(atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))
            if angle > threshold:
                break
            result.pop()
        result.append(point)
    # An entirely degenerate line still owns its original endpoints.
    if len(result) == 1:
        result.append(points[-1])
    return result
