from simulation.Grid import Grid
from simulation.Load import Load
from simulation.ValueMapStorage import ValueMapStorage
from simulation.PVSystem import PVSystem
from collections import deque
import numpy as np
from simulation.simulation_globals import JOULES_PER_KWH, MAX_POWER_TO_GRID, KWP, CAPACITY, BATTERY_VALUE_MAP_FILE, CONVERTER_VALUE_MAP_FILE

INFO_HEADER = ['SOC', 'LOAD_CONSUM', 'PV_COMSUM', 'STORAGE_CONSUM', 'STORAGE_SCHEDULED_POWER', 'GRID_BOUGHT', 'GRID_SOLD', 'GRID_WASTED']


class Environment:
    def __init__(self, tail_len, episode_container, dt_sim_step, soc_reward=0, soc_initial=0.5, sim_steps_per_action=1):
        self.pv_system = PVSystem(episode_container.pv_ts, KWP, dt_sim_step)
        self.load = Load(episode_container.load_ts, dt_sim_step)
        self.storage = ValueMapStorage(CAPACITY, soc_initial, dt_sim_step, CONVERTER_VALUE_MAP_FILE, BATTERY_VALUE_MAP_FILE)
        self.soc_reward = soc_reward

        self.sim_steps_per_action = sim_steps_per_action

        self.year_position = None

        self.buy_price_ts = deque(episode_container.buy_price_ts)
        self.sell_price_ts = deque(episode_container.sell_price_ts)
        self.year_cycle_ts = deque(episode_container.year_cycle_ts)

        self.grid = Grid([self.pv_system, self.load, self.storage], MAX_POWER_TO_GRID,dt_sim_step)
        self.single_states = deque(maxlen=tail_len)

    def _build_single_state(self, e_load, e_pv):
        return self.storage.soc(), e_load, e_pv, self.year_position

    def _build_aux_info(self, e_load, e_pv, e_storage):
        return self.storage.soc(), e_load, e_pv, e_storage, self.storage.scheduled_power_ac, self.grid.energy_bought, self.grid.energy_sold, self.grid.energy_wasted

    def _step_grid_parts(self):
        for p in [self.pv_system, self.load, self.storage]:
            p.step()

    def reset(self):
        while len(self.single_states) < self.single_states.maxlen:
            self.step([0])
        return np.array(self.single_states)

    def step(self, action):
        # carry out action
        action = np.clip(action, -1., 1.)
        self.storage.scheduled_power_ac = action[0] * 1500
        # parts can be stepped and metered in finer steps than RL algorithm to record fluctuations

        done = None
        e_pv = 0
        e_load = 0
        e_storage = 0

        for _ in range(self.sim_steps_per_action):
            ## step all grid parts
            self._step_grid_parts()
            ## collect energy 'packages' from all parts an meter it
            self.grid.meter_energy_from_parts()

            buy_price = self.buy_price_ts.popleft()
            sell_price = self.sell_price_ts.popleft()
            self.year_position = self.year_cycle_ts.popleft()

            e_pv += self.pv_system.consumed_energy
            e_load += self.load.consumed_energy
            e_storage += self.storage.consumed_energy

            done = not self.buy_price_ts
            if done:
                break


        reward = (sell_price * self.grid.energy_sold - buy_price * self.grid.energy_bought) / JOULES_PER_KWH
        # prevent drain towards end of episode
        reward += ((sell_price * self.storage.stored_energy / 2) / JOULES_PER_KWH if done else 0)

        scaled_reward = reward / np.sqrt(3e-3)  -  self.soc_reward * np.abs((2*self.storage.soc() - 1))

        #reset meter for next environment step

        self.single_states.append(self._build_single_state(e_load, e_pv))
        aux_info = self._build_aux_info(e_load, e_pv, e_storage)
        self.grid.reset_meter()

        return np.asarray(self.single_states), scaled_reward, done, aux_info
