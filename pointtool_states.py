"""The two live input adapters; TraceSession owns their logical state."""


class WaitingFirstPointState:
    def __init__(self, pointtool):
        self.pointtool = pointtool

    def click_lmb(self, event, layer):
        self.pointtool.accept_click(
            self.pointtool.toMapCoordinates(event.pos()), event.pos()
        )

    def click_rmb(self, event, layer):
        self.pointtool.finish_session()


class WaitingMiddlePointState(WaitingFirstPointState):
    pass
