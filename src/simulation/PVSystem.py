from simulation.GridPart import GridParticipant
from collections import deque


class PVSystem(GridParticipant):
    def __init__(self, grid, power_ts, kwp):
        super().__init__(grid)
        self.power_ts = deque(power_ts)
        self.kwp = kwp

    def _step(self):
        return self.power_ts.popleft() * self.kwp * -1
