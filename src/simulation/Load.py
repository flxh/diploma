from simulation.GridPart import GridParticipant
from collections import deque


class Load(GridParticipant):
    def __init__(self, grid, load_ts):
        super().__init__(grid)
        self.load_ts = deque(load_ts)

    def _step(self):
        return self.load_ts.popleft()
