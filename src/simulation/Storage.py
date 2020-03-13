from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP

class Storage(GridParticipant):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.stored_energy = capacity/2
        self.scheduled_power = 0
        self.actual_power = None

    def soc(self):
        return self.stored_energy / self.capacity

    def step(self):
        self.actual_power = self.scheduled_power
        self.stored_energy += self.scheduled_power * TIME_STEP
        self.consumed_energy = self.scheduled_power * TIME_STEP



