import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_files = {
    'ARIMA': 'arima_prediction.csv',
    'LSTM': 'lstm_prediction.csv',
    'Naive Zero Change': 'zero_change_prediction.csv',
    'Naive Season': 'season_prediction.csv'
}

def calaculate_eval_mectrics_from_file(filepath):
    df = pd.read_csv(filepath, names=['true', 'prediction'], delimiter=';')
    return (
        r2_score(df['true'], df['prediction']),
        mean_squared_error(df['true'], df['prediction']),
        mean_absolute_error(df['true'], df['prediction']),
        np.mean(np.sign(df['true']) == np.sign(df['prediction']))
    )


for key, value in data_files.items():
    r2, mse, mae, mda = calaculate_eval_mectrics_from_file(value)
    print('\n',key)
    print('R2:  ', r2)
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('MDA: ', mda)

