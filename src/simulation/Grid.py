class Grid:
    def __init__(self, grid_parts, max_power_to_utility, dt_step):
        self.parts = grid_parts
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
        self.max_power_to_utility = max_power_to_utility

        self.dt_step = dt_step

    def meter_energy_from_parts(self):
        period_energy_consumed = 0
        for p in self.parts:
            period_energy_consumed += p.consumed_energy

        if period_energy_consumed > 0:
            self.energy_bought += period_energy_consumed
        else:
            energy_outbound = -period_energy_consumed
            energy_to_grid = min(energy_outbound, self.dt_step*self.max_power_to_utility)

            self.energy_sold += energy_to_grid
            self.energy_wasted += (energy_outbound - energy_to_grid)

    def reset_meter(self):
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
