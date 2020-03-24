from simulation.simulation_globals import TIME_STEP, MAX_POWER_TO_GRID


class Grid:
    def __init__(self, grid_parts, max_power_to_utility):
        self.parts = grid_parts
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
        self.max_power_to_utility = max_power_to_utility

    def meter_energy_from_parts(self):
        period_energy_consumed = 0
        for p in self.parts:
            period_energy_consumed += p.consumed_energy

        if period_energy_consumed > 0:
            self.energy_bought += period_energy_consumed
        else:
            energy_outbound = -period_energy_consumed
            energy_to_grid = min(energy_outbound, TIME_STEP*MAX_POWER_TO_GRID)

            self.energy_sold += energy_to_grid
            self.energy_wasted += (energy_outbound - energy_to_grid)

    def reset_meter(self):
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
