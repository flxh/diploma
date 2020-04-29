from simulation.GridParticipant import GridParticipant
from collections import deque


class PVSystem(GridParticipant):
    def __init__(self, power_ts, kwp, dt_step):
        super().__init__()
        self.power_ts = deque(power_ts)
        self.kwp = kwp
        self.dt_step = dt_step

    def step(self):
        power = self.power_ts.popleft() * self.kwp
        self.consumed_energy = self.dt_step * power
