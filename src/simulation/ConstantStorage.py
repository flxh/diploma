from simulation.Storage import Storage
from simulation.simulation_globals import STORAGE_EFFICIENCY


class ConstantStorage(Storage):
    def __init__(self,grid, capacity, soc_initial, dt_step):
        super().__init__(grid, capacity=capacity, soc_initial=soc_initial, dt_step=dt_step)

    def _step(self):
        # charge
        if self.scheduled_power_ac > 0:
            scheduled_power_dc = self.scheduled_power_ac * STORAGE_EFFICIENCY
            # charge with scheduled power or charge left capacity
            actual_power_dc = min(scheduled_power_dc, (self.capacity - self.stored_energy)/self.dt_step)
            actual_power_ac = actual_power_dc / STORAGE_EFFICIENCY
        # discharge
        else:
            scheduled_power_dc = self.scheduled_power_ac / STORAGE_EFFICIENCY
            actual_power_dc = max(scheduled_power_dc, (0 - self.stored_energy)/self.dt_step)
            actual_power_ac = actual_power_dc * STORAGE_EFFICIENCY

        self.stored_energy += actual_power_dc * self.dt_step
        return actual_power_ac
