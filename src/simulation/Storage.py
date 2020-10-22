from simulation.GridParticipant import GridParticipant
from abc import abstractmethod

EFFICIENCY = 0.94

class Storage(GridParticipant):
    def __init__(self, capacity, soc_initial, dt_step):
        super().__init__(dt_step=dt_step)
        self.capacity = capacity
        self.stored_energy = capacity * soc_initial
        self.scheduled_power_ac = 0
        self.actual_power = None

    def soc(self):
        return self.stored_energy / self.capacity

    @abstractmethod
    def step(self):
        '''
        updates internal state of each grid part
        This method must update consumed_energy
        :return:
        '''
        pass
