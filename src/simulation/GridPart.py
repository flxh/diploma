from abc import abstractmethod


class GridParticipant:
    def __init__(self, grid):
        self.grid = grid

    def step(self):
        power_draw = self._step()
        self.grid.add_draw(power_draw)
        return power_draw

    @abstractmethod
    def _step(self):
        '''
        updates internal state of each grid part
        And returns the AC power drawn from the grid
        :return:
        '''
        pass
