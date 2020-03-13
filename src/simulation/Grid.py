
class Grid:
    def __init__(self, grid_parts):
        self.parts = grid_parts
        self.energy_bought = 0
        self.energy_sold = 0

    def meter_energy_from_parts(self):
        period_energy_consumed = 0
        for p in self.parts:
            period_energy_consumed += p.consumed_energy

        if period_energy_consumed > 0:
            self.energy_bought += period_energy_consumed
        else:
            self.energy_sold += -period_energy_consumed

    def reset_meter(self):
        self.energy_bought = 0
        self.energy_sold = 0
