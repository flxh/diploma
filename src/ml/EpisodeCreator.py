from itertools import cycle
import numpy as np


class EpisodeCreator:

    def __init__(self, episode_queue, load_data, irradiation_data, year_cycle, buy_price_data, sell_price_data):

        self.load_data = load_data
        self.irradiation_data = irradiation_data
        self.buy_data = buy_price_data
        self.sell_data = sell_price_data
        self.year_cycle_data = year_cycle

        pv_data = np.array(irradiation_data)
        self.pv_iterator = cycle(pv_data)
        self.load_iterator = cycle(load_data)
        self.buy_price_iterator = cycle(buy_price_data)
        self.sell_price_iterator = cycle(sell_price_data)
        self.year_cycle_iterator = cycle(year_cycle)

        self.episode_queue = episode_queue

    def fill_queue(self, episode_steps):

        while True:
            load_ts = []
            pv_ts = []
            year_cycle_ts = []
            buy_price_ts = []
            sell_price_ts = []
            for _ in range(episode_steps):
                load_ts.append(next(self.load_iterator))
                pv_ts.append(next(self.pv_iterator))
                year_cycle_ts.append(next(self.year_cycle_iterator))
                buy_price_ts.append(next(self.buy_price_iterator))
                sell_price_ts.append(next(self.sell_price_iterator))
            self.episode_queue.put(EpisodeContainer(load_ts, pv_ts, year_cycle_ts, buy_price_ts, sell_price_ts))


class EpisodeContainer:

    def __init__(self, load_ts, pv_ts, year_cycle_ts, buy_price_ts, sell_price_ts):
        ts_lengths = [len(load_ts), len(pv_ts), len(year_cycle_ts), len(buy_price_ts), len(sell_price_ts)]
        if not all(ts_lengths[0] == tsl for tsl in ts_lengths):
            raise ValueError("All time series must be equal in length")

        self.load_ts = load_ts
        self.pv_ts = pv_ts
        self.year_cycle_ts = year_cycle_ts
        self.buy_price_ts = buy_price_ts
        self.sell_price_ts = sell_price_ts