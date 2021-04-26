from simulation.GridPart import GridParticipant
from abc import abstractmethod
import numpy as np

EFFICIENCY = 0.94

class Storage(GridParticipant):
    def __init__(self, grid, capacity, soc_initial, dt_step):
        super().__init__(grid)
        self.dt_step = dt_step
        self.capacity = capacity
        self.stored_energy = capacity * (soc_initial * 0.98 +0.01)
        self.scheduled_power_ac = 0
        self.actual_power = None

    def soc(self):
        return self.stored_energy / self.capacity

    @abstractmethod
    def _step(self):
        '''
        updates internal state of each grid part
        This method must update consumed_energy
        :return:
        '''
        pass
