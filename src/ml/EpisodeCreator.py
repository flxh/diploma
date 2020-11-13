from itertools import cycle
import numpy as np
import pandas as pd
import scipy.signal as sgn
import scipy.io as sio
import random
import pickle as pkl

MINUTES_PER_DAY = 24*60
MINUTES_PER_YEAR = MINUTES_PER_DAY*365

# [1, 11, 17, 25, 26, 27, 29, 44, 46, 47, 51, 54, 56, 57, 59, 60, 66, 67, 70, 71, 72, 73]


def split_episode(episodes, eval_set_size):
    eval_indices = set(random.sample(range(len(episodes)), eval_set_size))

    eval_episodes = [x for i,x in enumerate(episodes) if i in eval_indices]
    training_episodes = [x for i,x in enumerate(episodes) if i not in eval_indices]

    return training_episodes, eval_episodes


def dump_episodes(episodes, path):
    with open(path, 'wb') as file:
        for e in episodes:
            pkl.dump(e, file)


def create_episodes_constant_price(load_years, irradiation_years, buy_price, sell_price, days_per_episode, down_sample_rate=1):
    assert load_years.shape[1] == MINUTES_PER_YEAR
    assert irradiation_years.shape[1] == MINUTES_PER_YEAR # time resolution of 1 minute

    year_cycle_ts = [-np.cos(((x+MINUTES_PER_DAY*10)/(MINUTES_PER_YEAR))*2*np.pi) for x in range(MINUTES_PER_YEAR)]
    buy_price_ts = [buy_price]*MINUTES_PER_YEAR
    sell_price_ts = [sell_price]*MINUTES_PER_YEAR

    assert load_years.shape[1] == irradiation_years.shape[1] == len(year_cycle_ts) == len(sell_price_ts) == len(buy_price_ts)

    load_ts = np.hstack(load_years)
    irradiation_ts = np.hstack(irradiation_years)

    ts_len_lcm = np.lcm.reduce([len(ts) for ts in [buy_price_ts, sell_price_ts, load_ts, irradiation_ts, year_cycle_ts]])
    print(ts_len_lcm)

    samples_per_episode = days_per_episode * MINUTES_PER_DAY
    
    buy_price_ts = np.resize(buy_price_ts, ts_len_lcm)
    sell_price_ts = np.resize(sell_price_ts, ts_len_lcm)
    load_ts = np.resize(load_ts, ts_len_lcm)
    irradiation_ts = np.resize(irradiation_ts, ts_len_lcm)
    year_cycle_ts = np.resize(year_cycle_ts, ts_len_lcm)

    episodes = []
    id = 0
    for i in range(samples_per_episode, ts_len_lcm, samples_per_episode):
        episodes.append(EpisodeContainer(
            id,
            load_ts[i-samples_per_episode:i:down_sample_rate],
            irradiation_ts[i-samples_per_episode:i:down_sample_rate],
            year_cycle_ts[i-samples_per_episode:i:down_sample_rate],
            buy_price_ts[i-samples_per_episode:i:down_sample_rate],
            sell_price_ts[i-samples_per_episode:i:down_sample_rate]))
        id += 1
    return episodes


def load_irraditaion_data(path):
    df = pd.read_csv(path, names=['datetime', 'obb', 'tharandt'], header=0, delimiter=';')
    df.fillna(0, inplace=True)

    ts_thar = np.array(df['tharandt'])
    ts_thar_resampled = sgn.resample_poly(ts_thar, 10,1)
    ts_thar_resampled = ts_thar_resampled[:-(len(ts_thar_resampled)%MINUTES_PER_YEAR)]
    ts_thar_resampled_years = np.reshape(ts_thar_resampled, (-1, MINUTES_PER_YEAR))

    return ts_thar_resampled_years


def load_total_power_from_mat_file(path, day_start=0, day_end=365, included_years=None):
    mat_contents = sio.loadmat(path)
    pges = mat_contents['PL1'] + mat_contents['PL2'] + mat_contents['PL3']
    pges = np.swapaxes(pges, 0,1)

    if included_years is not None:
        years = pges[included_years, day_start*MINUTES_PER_DAY: day_end*MINUTES_PER_DAY]
    else:
        years = pges[:, day_start*MINUTES_PER_DAY: day_end*MINUTES_PER_DAY]
    return years


class EpisodeLoader:

    def __init__(self, episode_queue, episode_file_path):
        self.episode_file_path = episode_file_path
        self.episode_queue = episode_queue

    def fill_queue(self):
        while True:
            with open(self.episode_file_path, 'rb') as file:
                while True:
                    try:
                        self.episode_queue.put(pkl.load(file))
                    except EOFError:
                        break


class EpisodeContainer:

    def __init__(self, id, load_ts, pv_ts, year_cycle_ts, buy_price_ts, sell_price_ts):
        ts_lengths = [len(load_ts), len(pv_ts), len(year_cycle_ts), len(buy_price_ts), len(sell_price_ts)]
        if not all(ts_lengths[0] == tsl for tsl in ts_lengths):
            raise ValueError("All time series must be equal in length")
        self.id = id
        self.load_ts = load_ts
        self.pv_ts = pv_ts
        self.year_cycle_ts = year_cycle_ts
        self.buy_price_ts = buy_price_ts
        self.sell_price_ts = sell_price_ts