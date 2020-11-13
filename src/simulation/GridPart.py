from abc import abstractmethod

class GridParticipant:
    def __init__(self,dt_step):
        self.consumed_energy = None
        self.dt_step = dt_step

    @abstractmethod
    def step(self):
        '''
        updates internal state of each grid part
        This method must update consumed_energy
        :return:
        '''
        pass
