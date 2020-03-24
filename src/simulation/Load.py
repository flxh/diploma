from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP
from collections import deque


class Load(GridParticipant):
    def __init__(self, load_ts):
        super().__init__()
        self.load_ts = deque(load_ts)

    def step(self):
        power = self.load_ts.popleft()
        self.consumed_energy = TIME_STEP * power
