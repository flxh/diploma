from simulation.GridParticipant import GridParticipant
from simulation.simulation_globals import TIME_STEP

EFFICIENCY = 0.94

class Storage(GridParticipant):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.stored_energy = capacity/2
        self.scheduled_power_ac = 0
        self.actual_power = None

    def soc(self):
        return self.stored_energy / self.capacity

    def step(self):
        # charge
        if self.scheduled_power_ac > 0:
            scheduled_power_dc = self.scheduled_power_ac * EFFICIENCY
            # charge with scheduled power or charge left capacity
            actual_power_dc = min(scheduled_power_dc, (self.capacity - self.stored_energy)/TIME_STEP)
            actual_power_ac = actual_power_dc / EFFICIENCY
        # discharge
        else:
            scheduled_power_dc = self.scheduled_power_ac / EFFICIENCY
            actual_power_dc = max(scheduled_power_dc, (0 - self.stored_energy)/TIME_STEP)
            actual_power_ac = actual_power_dc * EFFICIENCY

        self.stored_energy += actual_power_dc * TIME_STEP
        self.consumed_energy = actual_power_ac *TIME_STEP
