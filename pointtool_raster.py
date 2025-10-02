"""Raster sampling and pathfinding helpers for PointTool."""

import os
import time

import numpy as np
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis, QgsMessageLog, QgsProject

from .exceptions import OutsideMapError
from .utils import PossiblyIndexedImageError, get_whole_raster


PROFILE_ENABLED = os.environ.get("RASTER_TRACER_PROFILE", "0") == "1"


class RasterTracingContext:
    """Encapsulates raster sampling state and operations."""

    def __init__(self, tool):
        self._tool = tool
        self.raster_sampler = None
        self.sample = None
        self.grid = None
        self.grid_changed = None
        self.window_origin = None
        self.window_shape = None
        self.window_padding = 1024
        self.min_window_size = 512

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self):
        self.sample = None
        self.grid = None
        self.grid_changed = None
        self.window_origin = None
        self.window_shape = None

    def _clear_conversions(self):
        tool = self._tool
        tool.to_indexes = None
        tool.to_coords = None
        tool.to_coords_provider = None
        tool.to_coords_provider2 = None

    def ensure_sampler(self):
        """Load a sampler for the current raster layer if required."""
        tool = self._tool
        if self.raster_sampler is not None or tool.rlayer is None:
            return
        try:
            sampler = get_whole_raster(
                tool.rlayer,
                QgsProject.instance(),
            )
        except PossiblyIndexedImageError:
            tool.display_message(
                "Missing Layer",
                "Can't trace indexed or gray image",
                level='Critical',
                duration=2,
            )
            self.raster_sampler = None
            self.reset()
            self._clear_conversions()
            return

        self.raster_sampler = sampler
        tool.to_indexes = sampler.to_indexes
        tool.to_coords = sampler.to_coords
        tool.to_coords_provider = sampler.to_coords_provider
        tool.to_coords_provider2 = sampler.to_coords_provider2
        self.reset()
        self.recompute_trace_grid(reason="lazy-load")

    def set_sampler_for_layer(self, layer):
        """Load raster sampler for a newly selected layer."""
        if layer is None:
            self.raster_sampler = None
            self.reset()
            self._clear_conversions()
            return False

        try:
            sampler = get_whole_raster(
                layer,
                QgsProject.instance(),
            )
        except PossiblyIndexedImageError:
            self._tool.display_message(
                "Missing Layer",
                "Can't trace indexed or gray image",
                level='Critical',
                duration=2,
            )
            self.raster_sampler = None
            self.reset()
            self._clear_conversions()
            return False

        self.raster_sampler = sampler
        tool = self._tool
        tool.to_indexes = sampler.to_indexes
        tool.to_coords = sampler.to_coords
        tool.to_coords_provider = sampler.to_coords_provider
        tool.to_coords_provider2 = sampler.to_coords_provider2
        self.reset()
        self.recompute_trace_grid(reason="raster-change")
        return True

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------
    def indices_inside_window(self, index):
        if self.window_origin is None or self.window_shape is None:
            return False
        origin_i, origin_j = self.window_origin
        height, width = self.window_shape
        i, j = index
        return origin_i <= i < origin_i + height and origin_j <= j < origin_j + width

    def compute_window_bounds(self, indices, padding):
        if self.raster_sampler is None:
            return None

        height = self.raster_sampler.height
        width = self.raster_sampler.width

        clamped_i = []
        clamped_j = []
        for i, j in indices:
            clamped_i.append(max(0, min(int(i), height - 1)))
            clamped_j.append(max(0, min(int(j), width - 1)))

        if not clamped_i or not clamped_j:
            return None

        min_i = min(clamped_i)
        max_i = max(clamped_i)
        min_j = min(clamped_j)
        max_j = max(clamped_j)

        target_padding = max(int(padding), 0)

        i_min = max(0, min_i - target_padding)
        i_max = min(height, max_i + target_padding + 1)
        j_min = max(0, min_j - target_padding)
        j_max = min(width, max_j + target_padding + 1)

        min_height = min(self.min_window_size, height)
        min_width = min(self.min_window_size, width)

        current_height = i_max - i_min
        if current_height < min_height:
            needed = min_height - current_height
            extend_top = min(i_min, needed // 2)
            extend_bottom = min(height - i_max, needed - extend_top)
            i_min = max(0, i_min - extend_top)
            i_max = min(height, i_max + extend_bottom)

        current_width = j_max - j_min
        if current_width < min_width:
            needed = min_width - current_width
            extend_left = min(j_min, needed // 2)
            extend_right = min(width - j_max, needed - extend_left)
            j_min = max(0, j_min - extend_left)
            j_max = min(width, j_max + extend_right)

        return int(i_min), int(i_max), int(j_min), int(j_max)

    def load_window(self, bounds, reason):
        if bounds is None or self.raster_sampler is None:
            return

        i_min, i_max, j_min, j_max = bounds
        load_start = time.perf_counter() if PROFILE_ENABLED else None

        bands, origin, shape = self.raster_sampler.read_window(
            i_min,
            i_max,
            j_min,
            j_max,
        )

        if bands is None or shape == (0, 0):
            return

        prep_start = time.perf_counter() if PROFILE_ENABLED else None
        cleaned_bands = [np.nan_to_num(band, copy=False) for band in bands]
        prep_duration = (
            time.perf_counter() - prep_start
        ) if PROFILE_ENABLED else None

        grid_start = time.perf_counter() if PROFILE_ENABLED else None
        grid = cleaned_bands[0] + cleaned_bands[1] + cleaned_bands[2]
        grid_duration = (
            time.perf_counter() - grid_start
        ) if PROFILE_ENABLED else None

        self.sample = tuple(cleaned_bands)
        self.grid = grid
        self.window_origin = origin
        self.window_shape = shape
        self.grid_changed = None

        total_duration = (
            time.perf_counter() - load_start
        ) if PROFILE_ENABLED else None

        if PROFILE_ENABLED:
            color_bytes = sum(arr.nbytes for arr in self.sample)
            grid_bytes = self.grid.nbytes if isinstance(self.grid, np.ndarray) else 0

            def _fmt(value):
                return f"{value:.2f}s" if value is not None else "n/a"

            QgsMessageLog.logMessage(
                (
                    "[profiling] window_prepare "
                    f"reason={reason} origin={origin} shape={shape} "
                    f"prep={_fmt(prep_duration)} grid_sum={_fmt(grid_duration)} "
                    f"total={_fmt(total_duration)} color_mb={(color_bytes / (1024 ** 2)):.1f} "
                    f"grid_mb={(grid_bytes / (1024 ** 2)):.1f}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

        self.recompute_trace_grid(reason=f"window:{reason}")

    def ensure_window_for_indices(self, indices, reason, padding=None):
        if not indices:
            return

        self.ensure_sampler()
        if self.raster_sampler is None:
            return

        if padding is None:
            padding = self.window_padding

        if all(self.indices_inside_window(index) for index in indices):
            return

        bounds = self.compute_window_bounds(indices, padding)
        self.load_window(bounds, reason)

    def to_local_indices(self, i, j):
        if self.window_origin is None:
            raise OutsideMapError
        origin_i, origin_j = self.window_origin
        local_i = i - origin_i
        local_j = j - origin_j
        if (
            local_i < 0 or local_j < 0 or
            self.window_shape is None or
            local_i >= self.window_shape[0] or
            local_j >= self.window_shape[1]
        ):
            raise OutsideMapError
        return local_i, local_j

    # ------------------------------------------------------------------
    # Grid preparation
    # ------------------------------------------------------------------
    def recompute_trace_grid(self, reason):
        if (
            not PROFILE_ENABLED and
            self.sample is None and
            self._tool.trace_color_value is None
        ):
            self.grid_changed = None
            return

        start_time = time.perf_counter() if PROFILE_ENABLED else None
        diff_duration = None

        if self.sample is None:
            self.grid_changed = None
            state = "no-sample"
        elif self._tool.trace_color_value is None:
            self.grid_changed = None
            state = "cleared"
        else:
            compute_start = time.perf_counter() if PROFILE_ENABLED else None
            r, g, b = self.sample
            r0, g0, b0 = self._tool.trace_color_value
            self.grid_changed = np.abs((r0 - r) ** 2 + (g0 - g) ** 2 + (b0 - b) ** 2)
            if PROFILE_ENABLED:
                diff_duration = time.perf_counter() - compute_start
            state = "computed"

        if PROFILE_ENABLED:
            total_duration = time.perf_counter() - start_time if start_time is not None else None
            diff_text = f"{diff_duration:.2f}s" if diff_duration is not None else "n/a"
            total_text = f"{total_duration:.2f}s" if total_duration is not None else "n/a"
            grid_changed_bytes = (
                self.grid_changed.nbytes if isinstance(self.grid_changed, np.ndarray) else 0
            )
            QgsMessageLog.logMessage(
                (
                    "[profiling] trace_color_changed "
                    f"state={state} reason={reason} diff={diff_text} "
                    f"total={total_text} grid_changed_mb={ (grid_changed_bytes / (1024 ** 2)):.1f}"
                ),
                "RasterTracer",
                Qgis.Info,
            )

    def prepare_pathfinding(self, start, goal, reason):
        self.ensure_window_for_indices([start, goal], reason=reason)

        if self.sample is None or self.grid is None:
            raise OutsideMapError

        try:
            local_start = self.to_local_indices(*start)
            local_goal = self.to_local_indices(*goal)
        except OutsideMapError:
            self.ensure_window_for_indices(
                [start, goal],
                reason=f"{reason}-grow",
                padding=self.window_padding * 2,
            )
            local_start = self.to_local_indices(*start)
            local_goal = self.to_local_indices(*goal)

        r, g, b = self.sample

        try:
            r0 = r[local_goal]
            g0 = g[local_goal]
            b0 = b[local_goal]
        except IndexError:
            raise OutsideMapError

        if self.grid_changed is None:
            grid_to_use = np.abs((r0 - r) ** 2 + (g0 - g) ** 2 + (b0 - b) ** 2)
        else:
            grid_to_use = self.grid_changed

        grid_for_path = grid_to_use.astype(np.dtype('l'))
        origin_i, origin_j = self.window_origin

        return {
            "grid": grid_for_path,
            "local_start": local_start,
            "local_goal": local_goal,
            "origin": (origin_i, origin_j),
        }

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def sample_color_at_indices(self, i, j):
        self.ensure_window_for_indices([(i, j)], reason="shortcut-sample")

        if self.sample is None:
            QgsMessageLog.logMessage(
                "[shortcut] Sampling failed – raster data unavailable",
                "RasterTracer",
                Qgis.Info,
            )
            return None

        try:
            local_i, local_j = self.to_local_indices(i, j)
        except OutsideMapError:
            QgsMessageLog.logMessage(
                "[shortcut] Sampling failed – indices outside window",
                "RasterTracer",
                Qgis.Info,
            )
            return None

        try:
            r_band, g_band, b_band = self.sample
            r_val = float(r_band[local_i, local_j])
            g_val = float(g_band[local_i, local_j])
            b_val = float(b_band[local_i, local_j])
        except (IndexError, TypeError, ValueError):
            QgsMessageLog.logMessage(
                "[shortcut] Sampling failed – invalid pixel data",
                "RasterTracer",
                Qgis.Info,
            )
            return None

        r_int = int(np.clip(round(r_val), 0, 255))
        g_int = int(np.clip(round(g_val), 0, 255))
        b_int = int(np.clip(round(b_val), 0, 255))

        return QColor(r_int, g_int, b_int)
