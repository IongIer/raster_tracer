class OutsideMapError(ValueError):
    pass


class InvalidRasterError(ValueError):
    pass


class ResourceLimitError(Exception):
    pass


class TraceCancelled(Exception):
    pass
