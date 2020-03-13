from abc import abstractmethod
from simulation.simulation_globals import TIME_STEP


class GridParticipant:
    def __init__(self):
        self.consumed_energy = None

    @abstractmethod
    def step(self):
        '''
        updates internal state of each grid part
        This method must update consumed_energy
        :return:
        '''
        pass
