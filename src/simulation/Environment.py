from simulation.Grid import Grid
from simulation.Load import Load
from simulation.ValueMapStorage import ValueMapStorage
from simulation.ConstantStorage import ConstantStorage
from simulation.PVSystem import PVSystem
from collections import deque
import numpy as np
from simulation.simulation_globals import JOULES_PER_KWH, MAX_POWER_TO_GRID, KWP, CAPACITY, BATTERY_VALUE_MAP_FILE, CONVERTER_VALUE_MAP_FILE, CONVERTER_MAX_POWER

INFO_HEADER = ['SOC', 'LOAD_CONSUM', 'PV_COMSUM', 'STORAGE_CONSUM', 'STORAGE_SCHEDULED_POWER', 'GRID_BOUGHT', 'GRID_SOLD', 'GRID_WASTED']


class Environment:
    def __init__(self, tail_len, episode_container, dt_sim_step, soc_reward=0, soc_initial=0.5, sim_steps_per_action=1):
        self.pv_system = PVSystem(episode_container.pv_ts, KWP, dt_sim_step)
        self.load = Load(episode_container.load_ts, dt_sim_step)
        #self.storage = ValueMapStorage(CAPACITY, soc_initial, dt_sim_step, CONVERTER_VALUE_MAP_FILE, BATTERY_VALUE_MAP_FILE)
        self.storage = ConstantStorage(CAPACITY, soc_initial, dt_sim_step)
        self.soc_reward = soc_reward

        self.sim_steps_per_action = sim_steps_per_action

        self.year_position = None

        self.buy_price_ts = deque(episode_container.buy_price_ts)
        self.sell_price_ts = deque(episode_container.sell_price_ts)
        self.year_cycle_ts = deque(episode_container.year_cycle_ts)


        self.grid = Grid([self.pv_system, self.load, self.storage], MAX_POWER_TO_GRID,dt_sim_step)
        self.single_states = deque(maxlen=tail_len)

    def _build_single_state(self, e_load, e_pv):
        single_state = self.storage.soc(), e_load, e_pv, self.year_position
        if np.isnan(single_state).any():
            raise ValueError(f"State must not contain Nan - State: {single_state}")
        return single_state

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
        # Aktion trimmen
        # Wird benötigt, da stochastische Aktionen Grenzwerte überschreiten
        action = np.clip(action, -1., 1.)
        # Leistung des Speichers einstellen
        action = action*CONVERTER_MAX_POWER

        self.storage.scheduled_power_ac = action[0]

        done = None
        e_pv = 0
        e_load = 0
        e_storage = 0

        # n Simulationsschritte ausführen
        for _ in range(self.sim_steps_per_action):
            # Alle Systemkomponenten einen Zeitschritt ausführen
            self._step_grid_parts()
            # Energiemengen des Systemkomponenten einsammeln
            self.grid.meter_energy_from_parts()
            # Energiepreise aus der Zeitreihe abrufen
            buy_price = self.buy_price_ts.popleft()
            sell_price = self.sell_price_ts.popleft()
            self.year_position = self.year_cycle_ts.popleft()

            # Energiemenge der einzelnen Komponenten addieren
            # dient ausschließlich der Dokumentation des Trainings
            e_pv += self.pv_system.consumed_energy
            e_load += self.load.consumed_energy
            e_storage += self.storage.consumed_energy

            # Prüfen ob Episode zu Ende ist
            done = not self.buy_price_ts
            if done:
                break

        # Belohnungssignal berechnen
        reward = (sell_price * self.grid.energy_sold
                  - buy_price * self.grid.energy_bought) / JOULES_PER_KWH

        # Bewertung der gespeicherten Energie am Ende einer Episode
        reward += ((sell_price * self.storage.stored_energy / 2)
                   / JOULES_PER_KWH if done else 0)

        # Belohnungssignal skalieren und SOC Regulierung
        scaled_reward = ((reward / np.sqrt(3e-3)  \
                        -  self.soc_reward * np.abs((2*self.storage.soc() - 1)))+1.7) / 5**0.5
        # Einzelnen Zustandsvektor speichern
        self.single_states.append(self._build_single_state(e_load, e_pv))
        # Zusatzinformation zur Dokumentation generieren
        aux_info = self._build_aux_info(e_load, e_pv, e_storage)
        # Energiezähler zurücksetzen
        self.grid.reset_meter()

        return np.asarray(self.single_states), scaled_reward, done, aux_info
