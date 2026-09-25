"""Endpoint-preserving numerical moving mean, independent of QGIS."""


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
