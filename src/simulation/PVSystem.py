from simulation.GridParticipant import GridParticipant
from collections import deque


class PVSystem(GridParticipant):
    def __init__(self, power_ts, kwp, dt_step):
        super().__init__(dt_step=dt_step)
        self.power_ts = deque(power_ts)
        self.kwp = kwp

    def step(self):
        power = self.power_ts.popleft() * self.kwp
        self.consumed_energy = self.dt_step * power
