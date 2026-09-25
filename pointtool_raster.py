"""Bounded numerical preparation, plus the GUI sampler installation boundary."""

import os
import time
from dataclasses import dataclass

import numpy as np
from osgeo import gdal
from qgis.core import Qgis, QgsMessageLog, QgsProject
from qgis.PyQt.QtGui import QColor

from .exceptions import (
    InvalidRasterError,
    OutsideMapError,
    ResourceLimitError,
    TraceCancelled,
)
from .utils import RasterSampler

# Limits bound our allocations/counts, not GDAL caches or total process RSS.
MAX_SEARCH_PIXELS = 8_388_608
MAX_ARRAY_BYTES = 256 * 1024 * 1024
SCRATCH_PIXELS = 65_536
SCRATCH_BYTES = SCRATCH_PIXELS * 32
# Reserve the worst-case GUI small-read cache, replacement, and snap scratch.
# A 99-pixel radius is at most 199x199 pixels, independent of trace windows.
CURSOR_RESERVE_BYTES = 6 * 1024 * 1024
WINDOW_PADDING = 1024
MIN_WINDOW_SIZE = 512
CACHE_ALIGNMENT = 256


def check_cancel(cancel):
    if cancel():
        raise TraceCancelled()


def window_bounds(
    indices, height, width, padding=WINDOW_PADDING, minimum=MIN_WINDOW_SIZE
):
    if not indices:
        raise OutsideMapError("No endpoints")
    for i, j in indices:
        if not (0 <= i < height and 0 <= j < width):
            raise OutsideMapError("Endpoint outside raster")

    def dimension(values, length):
        lo = max(0, min(values) - padding)
        hi = min(length, max(values) + padding + 1)
        needed = max(0, min(minimum, length) - (hi - lo))
        lo = max(0, lo - needed // 2)
        hi = min(length, max(hi, lo + min(minimum, length)))
        lo = min(lo, hi - min(minimum, length))
        return int(lo), int(hi)

    top, bottom = dimension([p[0] for p in indices], height)
    left, right = dimension([p[1] for p in indices], width)
    return top, bottom, left, right


def validate_budget(bounds, live_bytes=0):
    top, bottom, left, right = bounds
    pixels = (bottom - top) * (right - left)
    if bottom <= top or right <= left:
        raise InvalidRasterError("Empty raster window")
    if pixels > MAX_SEARCH_PIXELS:
        raise ResourceLimitError("Search pixel limit")
    # Budget for float64 RGB even when Byte bands use compact storage:
    # RGB float64 (24), validity (1), costs (8), temporary GDAL/native band
    # and mask/conversion headroom (9), bounded float64 computation scratch.
    if (
        live_bytes + pixels * 42 + SCRATCH_BYTES + CURSOR_RESERVE_BYTES
        > MAX_ARRAY_BYTES
    ):
        raise ResourceLimitError("Raster array budget")
    return pixels


def cache_bounds(bounds, height, width, live_bytes=0):
    """Prefetch nearby windows without expanding the requested search graph."""
    top, bottom, left, right = bounds
    for alignment in (CACHE_ALIGNMENT, CACHE_ALIGNMENT // 2, CACHE_ALIGNMENT // 4):
        aligned = (
            top // alignment * alignment,
            min(height, (bottom + alignment - 1) // alignment * alignment),
            left // alignment * alignment,
            min(width, (right + alignment - 1) // alignment * alignment),
        )
        try:
            validate_budget(aligned, live_bytes)
        except ResourceLimitError:
            continue
        return aligned
    validate_budget(bounds, live_bytes)
    return bounds


def read_rgb(dataset, source, bounds, cancel):
    top, bottom, left, right = bounds
    shape = (bottom - top, right - left)
    if min(shape) <= 0:
        raise InvalidRasterError("Empty raster read")
    valid = np.ones(shape, dtype=bool)
    bands = []
    for number in source.bands:
        check_cancel(cancel)
        band = dataset.GetRasterBand(number)
        array = band.ReadAsArray(left, top, shape[1], shape[0]) if band else None
        if array is None or array.shape != shape or np.iscomplexobj(array):
            raise InvalidRasterError("Malformed RGB band read")
        converted = (
            array if array.dtype == np.uint8 else np.asarray(array, dtype=np.float64)
        )
        del array
        mask_band = band.GetMaskBand()
        mask = (
            mask_band.ReadAsArray(left, top, shape[1], shape[0]) if mask_band else None
        )
        if mask is None or mask.shape != shape:
            raise InvalidRasterError("Malformed validity mask read")
        valid &= mask != 0
        del mask
        flat = converted.reshape(-1)
        validity = valid.reshape(-1)
        for offset in range(0, flat.size, SCRATCH_PIXELS):
            check_cancel(cancel)
            validity[offset : offset + SCRATCH_PIXELS] &= np.isfinite(
                flat[offset : offset + SCRATCH_PIXELS]
            )
        converted.setflags(write=False)
        bands.append(converted)
    valid.setflags(write=False)
    return tuple(bands), valid


def color_cost(bands, valid, color, cancel=lambda: False):
    if len(color) != 3 or not all(np.isfinite(c) for c in color):
        raise InvalidRasterError("Nonfinite trace color")
    cost = np.empty(valid.shape, dtype=np.int64)
    flat_cost = cost.reshape(-1)
    # A cropped cache may be strided. Its flat iterator copies only the current
    # bounded chunk, rather than reshape() copying the entire retained window.
    flat_valid = valid.reshape(-1) if valid.flags.c_contiguous else valid.flat
    flat_bands = [
        band.reshape(-1) if band.flags.c_contiguous else band.flat for band in bands
    ]
    for offset in range(0, cost.size, SCRATCH_PIXELS):
        check_cancel(cancel)
        end = min(cost.size, offset + SCRATCH_PIXELS)
        mask = flat_valid[offset:end]
        scratch = np.zeros(end - offset, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for values, target in zip(flat_bands, color):
                # Promote before subtraction so Byte values cannot wrap around.
                delta = np.subtract(values[offset:end], target, dtype=np.float64)
                np.square(delta, out=delta)
                scratch += delta
                del delta
        # Invalid cells are impassable; zero is only an unused storage value.
        scratch[~mask] = 0
        if (
            np.any(~np.isfinite(scratch))
            or np.any(scratch >= 2**63)
            or np.any(scratch < 0)
        ):
            raise InvalidRasterError("Color cost outside int64 range")
        flat_cost[offset:end] = scratch
    cost.setflags(write=False)
    return cost


@dataclass(frozen=True)
class RasterSnapshot:
    source: object
    bounds: tuple
    bands: tuple
    valid: object
    cost: object = None
    cost_key: object = None

    @property
    def nbytes(self):
        # Views retain their backing arrays, including a cropped prefetch cache.
        # Count each allocation once even if several views share it.
        allocations = {}
        for array in (*self.bands, self.valid, self.cost):
            if array is None:
                continue
            while isinstance(array.base, np.ndarray):
                array = array.base
            allocations[id(array)] = array.nbytes
        return sum(allocations.values())

    def covers(self, source, bounds):
        a, b, c, d = self.bounds
        top, bottom, left, right = bounds
        return (
            self.source == source
            and a <= top
            and bottom <= b
            and c <= left
            and right <= d
        )


def prepare_snapshot(work, cached, cancel):
    started = time.perf_counter()
    validate_budget(work.bounds)
    top, bottom, left, right = work.bounds
    if not (
        0 <= top < bottom <= work.source.height
        and 0 <= left < right <= work.source.width
    ):
        raise InvalidRasterError("Window outside raster")
    for row, col in (work.start, work.goal):
        if not (top <= row < bottom and left <= col < right):
            raise OutsideMapError("Endpoint outside search window")
    check_cancel(cancel)
    rgb_hit = cached is not None and cached.covers(work.source, work.bounds)
    # The scheduler evicts a non-covering cache before starting a worker. Direct
    # callers can still hold one, so account for that owner during replacement.
    live_bytes = cached.nbytes if cached is not None and not rgb_hit else 0
    if rgb_hit:
        snapshot = cached
    else:
        bounds = cache_bounds(
            work.bounds, work.source.height, work.source.width, live_bytes
        )
        dataset = None
        try:
            dataset = gdal.OpenEx(
                work.source.path,
                gdal.OF_RASTER | gdal.OF_READONLY,
                open_options=list(work.source.open_options),
            )
            if dataset is None or (dataset.RasterYSize, dataset.RasterXSize) != (
                work.source.height,
                work.source.width,
            ):
                raise InvalidRasterError("Raster source changed or cannot be opened")
            try:
                bands, valid = read_rgb(dataset, work.source, bounds, cancel)
            except (InvalidRasterError, RuntimeError):
                if bounds == work.bounds:
                    raise
                bounds, bands, valid = work.bounds, None, None
            if bands is None:
                bands, valid = read_rgb(dataset, work.source, bounds, cancel)
            snapshot = RasterSnapshot(work.source, bounds, bands, valid)
        finally:
            dataset = None
    read_duration = time.perf_counter() - started
    top, _, left, _ = snapshot.bounds
    start = (work.start[0] - top, work.start[1] - left)
    goal = (work.goal[0] - top, work.goal[1] - left)
    if not snapshot.valid[start] or not snapshot.valid[goal]:
        raise InvalidRasterError("Invalid endpoint pixel")
    color = (
        work.color
        if work.color is not None
        else tuple(float(b[goal]) for b in snapshot.bands)
    )
    key = (work.color is not None, color, "float64/int64/masked-v1")
    cost_hit = snapshot.cost is not None and snapshot.cost_key == key
    started = time.perf_counter()
    if not cost_hit:
        # A previous cost may still be held by this worker on an auto-color miss.
        # Account for it until replacement; the scheduler drops all other owners.
        pixels = snapshot.valid.size
        if (
            live_bytes
            + snapshot.nbytes
            + pixels * 8
            + SCRATCH_BYTES
            + CURSOR_RESERVE_BYTES
            > MAX_ARRAY_BYTES
        ):
            raise ResourceLimitError("Cost snapshot budget")
        try:
            costs = color_cost(snapshot.bands, snapshot.valid, color, cancel)
        except InvalidRasterError:
            if snapshot.bounds == work.bounds:
                raise
            # Prefetch must not introduce a cost-range error outside the exact
            # search window. Keep immutable RGB views, but limit these costs to
            # the requested region. Their backing storage remains accounted.
            a, b, c, d = work.bounds
            view = (slice(a - top, b - top), slice(c - left, d - left))
            snapshot = RasterSnapshot(
                snapshot.source,
                work.bounds,
                tuple(band[view] for band in snapshot.bands),
                snapshot.valid[view],
                snapshot.cost,
                snapshot.cost_key,
            )
            top, _, left, _ = snapshot.bounds
            costs = None
        if costs is None:
            # Leave the exception handler first: its traceback retains the
            # failed full-sized cost allocation until the exception is cleared.
            costs = color_cost(snapshot.bands, snapshot.valid, color, cancel)
        snapshot = RasterSnapshot(
            snapshot.source, snapshot.bounds, snapshot.bands, snapshot.valid, costs, key
        )
    check_cancel(cancel)
    a, b, c, d = work.bounds
    view = (slice(a - top, b - top), slice(c - left, d - left))
    # Views preserve the exact requested search graph regardless of cache history.
    return (
        snapshot,
        snapshot.cost[view],
        snapshot.valid[view],
        {
            "read": read_duration,
            "cost": time.perf_counter() - started,
            "rgb_cache_hit": rgb_hit,
            "cost_cache_hit": cost_hit,
            "copy_bytes": 0,
            "array_bytes": snapshot.nbytes,
        },
    )


class RasterTracingContext:
    def __init__(self, tool):
        self._tool = tool
        self.raster_sampler = None

    def reset(self):
        self.raster_sampler = None
        self._tool.to_indexes = self._tool.to_coords = None

    def _install_sampler(self, layer, reason):
        started = time.perf_counter()
        self.reset()
        if layer is None:
            return False
        try:
            sampler = RasterSampler(
                layer,
                QgsProject.instance(),
                self._tool.canvas().mapSettings().destinationCrs(),
            )
        except Exception as error:
            self._tool.report_failure("invalid_input", str(error))
            return False
        self.raster_sampler = sampler
        self._tool.to_indexes, self._tool.to_coords = (
            sampler.to_indexes,
            sampler.to_coords,
        )
        if os.environ.get("RASTER_SCRIBE_PROFILE", "0") == "1":
            QgsMessageLog.logMessage(
                f"[profiling] sampler reason={reason} duration={time.perf_counter() - started:.6f}s "
                f"shape=({sampler.height},{sampler.width})",
                "Raster Scribe",
                Qgis.MessageLevel.Info,
            )
        return True

    def ensure_sampler(self):
        if self.raster_sampler is None and self._tool.rlayer is not None:
            return self._install_sampler(self._tool.rlayer, "lazy-load")
        return self.raster_sampler is not None

    def set_sampler_for_layer(self, layer):
        return self._install_sampler(layer, "raster-change")

    def compute_window_bounds(self, indices, padding=WINDOW_PADDING):
        sampler = self.raster_sampler
        if sampler is None:
            raise OutsideMapError("No raster")
        return window_bounds(indices, sampler.height, sampler.width, padding)

    def sample_color_at_indices(self, i, j):
        if not self.ensure_sampler():
            return None
        _, bands, valid = self.raster_sampler.read_small(i, j)
        if not valid[0, 0]:
            return None
        return QColor(*(max(0, min(255, round(float(b[0, 0])))) for b in bands))

    def snap(self, i, j, radius, color):
        if color is None or radius is None:
            return i, j
        bounds, bands, valid = self.raster_sampler.read_small(i, j, radius)
        if not valid.any():
            return i, j
        cost = color_cost(bands, valid, color)
        rows, cols = np.nonzero(valid & (cost == cost[valid].min()))
        distance = (rows + (bounds[0] - i)) ** 2 + (cols + (bounds[2] - j)) ** 2
        # nonzero() yields row/column order; argmin() keeps the first distance
        # tie, matching the previous (color cost, distance, row, column) order.
        best = int(distance.argmin())
        return int(rows[best] + bounds[0]), int(cols[best] + bounds[2])
