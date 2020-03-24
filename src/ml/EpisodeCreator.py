from itertools import cycle
import numpy as np


class EpisodeCreator:

    def __init__(self, episode_queue, load_data, irradiation_data, buy_price_data, sell_price_data, episode_days):
        if len(load_data) % 96 != 0:
            raise ValueError('Load data must consist of full days with 15 min sample intervals')

        if len(irradiation_data) % 96 != 0:
            raise ValueError('Irradiation data must consist of full days with 15 min sample intervals')

        if len(buy_price_data) % 96 != 0:
            raise ValueError('Buy price data must consist of full days with 15 min sample intervals')

        if len(sell_price_data) % 96 != 0:
            raise ValueError('Sell price data must consist of full days with 15 min sample intervals')

        self.load_data = load_data
        self.irradiation_data = irradiation_data
        self.buy_data = buy_price_data
        self.sell_data = sell_price_data

        pv_data = np.array(irradiation_data)
        self.pv_iterator = cycle(pv_data)
        self.load_iterator = cycle(load_data)
        self.buy_price_iterator = cycle(buy_price_data)
        self.sell_price_iterator = cycle(sell_price_data)

        self.episode_queue = episode_queue
        self.episode_steps = episode_days * 96

    def create_evaluation_episode(self, eval_len_days):
        load_ts = []
        pv_ts = []
        buy_price_ts = []
        sell_price_ts = []

        load_iter = cycle(self.load_data)
        pv_iter = cycle(self.irradiation_data)
        buy_iter = cycle(self.buy_data)
        sell_iter = cycle(self.sell_data)


        for _ in range(eval_len_days  * 96):
            load_ts.append(next(load_iter))
            pv_ts.append(next(pv_iter))
            buy_price_ts.append(next(buy_iter))
            sell_price_ts.append(next(sell_iter))

        return EpisodeContainer(load_ts, pv_ts, buy_price_ts, sell_price_ts)

    def fill_queue(self):
        while True:
            load_ts = []
            pv_ts = []
            buy_price_ts = []
            sell_price_ts = []
            for _ in range(self.episode_steps):
                load_ts.append(next(self.load_iterator))
                pv_ts.append(next(self.pv_iterator))
                buy_price_ts.append(next(self.buy_price_iterator))
                sell_price_ts.append(next(self.sell_price_iterator))

            self.episode_queue.put(EpisodeContainer(load_ts, pv_ts, buy_price_ts, sell_price_ts))


class EpisodeContainer:

    def __init__(self, load_ts, pv_ts, buy_price_ts, sell_price_ts):
        ts_lengths = [len(load_ts), len(pv_ts), len(buy_price_ts), len(sell_price_ts)]
        if not all(ts_lengths[0] == tsl for tsl in ts_lengths):
            raise ValueError("All time series must be equal in length")

        self.load_ts = load_ts
        self.pv_ts = pv_ts
        self.buy_price_ts = buy_price_ts
        self.sell_price_ts = sell_price_ts