from simulation.GridParticipant import GridParticipant
from collections import deque


class Load(GridParticipant):
    def __init__(self, load_ts, dt_step):
        super().__init__()
        self.load_ts = deque(load_ts)
        self.dt_step = dt_step

    def step(self):
        power = self.load_ts.popleft()
        self.consumed_energy = self.dt_step * power
