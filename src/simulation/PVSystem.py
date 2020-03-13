from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP


class PVSystem(GridParticipant):
    def __init__(self, constant_power):
        super().__init__()
        self.power = constant_power

    def step(self):
        self.consumed_energy = TIME_STEP * self.power
