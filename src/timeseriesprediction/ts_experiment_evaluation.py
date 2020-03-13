import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_files = {
    'ARIMA': r'timeseriesprediction/arima_prediction_all.csv',
    'LSTM': r'timeseriesprediction/lstm_prediction_all.csv',
    'LSTM Last 10k': r'timeseriesprediction/lstm_prediction_last10k.csv',
    'Naive Zero Change': r'timeseriesprediction/zero_change_prediction_all.csv',
    'Naive Zero Change 5 steps': r'timeseriesprediction/zero_change_prediction_5s_all.csv',
    'Naive Season': r'timeseriesprediction/season_prediction_all.csv'
}

def calaculate_eval_mectrics_from_file(filepath):
    df = pd.read_csv(filepath, names=['true', 'prediction'], delimiter=';')
    print(np.mean((df['true']-np.mean(df['true']))**2))
    print(np.mean((df['true']-df['prediction'])**2))

    return (
        r2_score(df['true'], df['prediction']),
        mean_squared_error(df['true'], df['prediction']),
        mean_absolute_error(df['true'], df['prediction'])
    )


for key, value in data_files.items():
    r2, mse, mae = calaculate_eval_mectrics_from_file(value)
    print('\n',key)
    print('R2:  ', r2)
    print('MSE: ', mse)
    print('MAE: ', mae)

