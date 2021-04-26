class Grid:
    def __init__(self, max_power_to_utility, dt_step):
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
        self.max_power_to_utility = max_power_to_utility
        self.power_balance = 0

        self.dt_step = dt_step

    def add_draw(self, power_draw):
        self.power_balance += power_draw

    def update_meters(self):
        period_energy_consumed = self.power_balance * self.dt_step

        if period_energy_consumed > 0:
            self.energy_bought += period_energy_consumed
        else:
            energy_outbound = -period_energy_consumed
            energy_to_grid = min(energy_outbound, self.dt_step*self.max_power_to_utility)

            self.energy_sold += energy_to_grid
            self.energy_wasted += (energy_outbound - energy_to_grid)

        self.power_balance = 0

    def reset_meter(self):
        self.energy_bought = 0
        self.energy_sold = 0
        self.energy_wasted = 0
