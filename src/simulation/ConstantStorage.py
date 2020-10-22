from simulation.Storage import Storage

EFFICIENCY = 0.94

class ConstantStorage(Storage):
    def __init__(self, capacity, soc_initial, dt_step):
        super().__init__(capacity=capacity, soc_initial=soc_initial, dt_step=dt_step)

    def step(self):
        # charge
        if self.scheduled_power_ac > 0:
            scheduled_power_dc = self.scheduled_power_ac * EFFICIENCY
            # charge with scheduled power or charge left capacity
            actual_power_dc = min(scheduled_power_dc, (self.capacity - self.stored_energy)/self.dt_step)
            actual_power_ac = actual_power_dc / EFFICIENCY
        # discharge
        else:
            scheduled_power_dc = self.scheduled_power_ac / EFFICIENCY
            actual_power_dc = max(scheduled_power_dc, (0 - self.stored_energy)/self.dt_step)
            actual_power_ac = actual_power_dc * EFFICIENCY

        self.stored_energy += actual_power_dc * self.dt_step
        self.consumed_energy = actual_power_ac *self.dt_step
