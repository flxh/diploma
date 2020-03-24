from simulation.Grid import Grid
from simulation.Load import Load
from simulation.Storage import Storage
from simulation.PVSystem import PVSystem
from collections import deque
import numpy as np
from simulation.simulation_globals import JOULES_PER_KWH, MAX_POWER_TO_GRID, KWP, CAPACITY

INFO_HEADER = ['SOC', 'LOAD_CONSUM', 'PV_COMSUM', 'STORAGE_CONSUM', 'STORAGE_SCHEDULED_POWER', 'GRID_BOUGHT', 'GRID_SOLD', 'GRID_WASTED']


class Environment:
    def __init__(self, tail_len, episode_container):
        self.pv_system = PVSystem(episode_container.pv_ts, KWP)
        self.load = Load(episode_container.load_ts)
        self.storage = Storage(CAPACITY)

        self.buy_price_ts = deque(episode_container.buy_price_ts)
        self.sell_price_ts = deque(episode_container.sell_price_ts)

        self.grid = Grid([self.pv_system, self.load, self.storage], MAX_POWER_TO_GRID)
        self.single_states = deque(maxlen=tail_len)

    def _build_single_state(self):
        return self.storage.soc(), self.load.consumed_energy, self.pv_system.consumed_energy

    def _build_aux_info(self):
        return self.storage.soc(), self.load.consumed_energy, self.pv_system.consumed_energy, self.storage.consumed_energy, self.storage.scheduled_power_ac, self.grid.energy_bought, self.grid.energy_sold, self.grid.energy_wasted

    def _step_grid_parts(self):
        for p in [self.pv_system, self.load, self.storage]:
            p.step()

    def reset(self):
        while len(self.single_states) < self.single_states.maxlen:
            self.step([0])
        return np.array(self.single_states)

    def step(self, action):
        # carry out action
        self.storage.scheduled_power_ac = action[0] *4000
        # parts can be stepped and metered in finer steps than RL algorithm to record fluctuations
        ## step all grid parts
        self._step_grid_parts()
        ## collect energy 'packages' from all parts an meter it
        self.grid.meter_energy_from_parts()

        buy_price = self.buy_price_ts.popleft()
        sell_price = self.sell_price_ts.popleft()

        done = not self.buy_price_ts


        reward = (sell_price * self.grid.energy_sold - buy_price * self.grid.energy_bought) / JOULES_PER_KWH
        # prevent drain towards end of episode
        reward += ((sell_price * self.storage.stored_energy / 2) / JOULES_PER_KWH if done else 0)

        scaled_reward = reward / np.sqrt(3e-3)

        #read meters
        #print(self.grid.energy_bought)
        #print(self.grid.energy_sold)
        #print(len(self.buy_price_ts))

        #reset meter for next environment step

        self.single_states.append(self._build_single_state())
        aux_info = self._build_aux_info()
        self.grid.reset_meter()

        return np.array(self.single_states), scaled_reward, done, aux_info
