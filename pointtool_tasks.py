"""Background tracing task controller for PointTool."""

from qgis.core import QgsApplication

from .astar import FindPathTask


class TraceTaskController:
    """Manages the lifecycle of background ``FindPathTask`` instances."""

    def __init__(self, tool):
        self._tool = tool
        self._task = None

    @property
    def active(self):
        return self._task is not None

    def start(self, grid, local_start, local_goal, callback, vlayer):
        """Start a new tracing task, cancelling any previous one."""
        self.cancel()

        def wrapped(path, layer):
            try:
                callback(path, layer)
            finally:
                if self._task is task:
                    self._task = None

        task = FindPathTask(grid, local_start, local_goal, wrapped, vlayer)
        self._task = task
        QgsApplication.taskManager().addTask(task)
        return task

    def cancel(self):
        """Cancel the active task if any."""
        if self._task is None:
            return False
        try:
            self._task.cancel()
        except RuntimeError:
            self._task = None
            return False
        self._task = None
        return True
