from simulation.GridPart import GridPart
from collections import deque


class Load(GridPart):
    def __init__(self, load_ts, dt_step):
        super().__init__(dt_step)
        self.load_ts = deque(load_ts)

    def step(self):
        power = self.load_ts.popleft()
        self.consumed_energy = self.dt_step * power
