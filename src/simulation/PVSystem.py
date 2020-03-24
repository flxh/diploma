from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP
from collections import deque


class PVSystem(GridParticipant):
    def __init__(self, power_ts, kwp):
        super().__init__()
        self.power_ts = deque(power_ts)
        self.kwp = kwp

    def step(self):
        power = self.power_ts.popleft() * self.kwp
        self.consumed_energy = TIME_STEP * power
