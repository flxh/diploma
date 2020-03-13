from simulation.Grid import Grid
from simulation.Load import Load
from simulation.Storage import Storage
from simulation.PVSystem import PVSystem
from timeseriesprediction.utils import load_total_power_from_mat_file


class Environment:
    def __init__(self):
        power_values_all = load_total_power_from_mat_file('loadprofiles_1min.mat')
        power_values_h0 = power_values_all[:, 0]

        self.pv_system = PVSystem(-500)
        self.load = Load(power_values_h0)
        self.storage = Storage(3600000)

        self.grid = Grid([self.pv_system, self.load, self.storage])

    def _build_state(self):
        return self.storage.soc(), self.load.consumed_energy, self.pv_system.consumed_energy

    def _step_grid_parts(self):
        for p in [self.pv_system, self.load, self.storage]:
            p.step()

    def reset(self):
        self._step_grid_parts()
        return self._build_state()

    def step(self, action):
        # carry out action
        self.storage.scheduled_power = action

        # parts can be stepped and metered in finer steps than RL algorithm to record fluctuations
        ## step all grid parts
        self._step_grid_parts()
        ## collect energy 'packages' from all parts an meter it
        self.grid.meter_energy_from_parts()

        #read meters
        print(self.grid.energy_bought)
        print(self.grid.energy_sold)

        #reset meter for next environment step
        self.grid.reset_meter()

        return self._build_state()