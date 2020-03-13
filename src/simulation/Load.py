from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP


class Load(GridParticipant):
    def __init__(self, time_series):
        super().__init__()
        self.time_series_iter = iter(time_series)

    def step(self):
        power = next(self.time_series_iter)
        self.consumed_energy = TIME_STEP * power
