import math
import pandas as pd
from model_func import *
from tinkoff_api_request import get_trading_data
from my_client_config import TOKEN
import numpy as np


def retraining_model(config_bots):
    path_model = config_bots['path_model']
    name_model = config_bots['name_model']
    figi = config_bots['figi']
    interval_time = config_bots['interval_time']
    last_day_for_training = config_bots['day_for_training']
    last_order = config_bots['order']
    last_num_values_for_predict = config_bots['num_values_for_predict']
    last_num_predictions = config_bots['num_predictions']
    spread_days_percent = config_bots['spread_days_percent']
    spread_order_percent = config_bots['spread_order_percent']
    spread_num_values_for_predict = config_bots['spread_num_values_for_predict']

    lower_day_limit = max(math.floor(last_day_for_training * (1 - spread_days_percent)), 1)
    upper_day_limit = math.ceil(last_day_for_training * (1 + spread_days_percent)) + 1

    lower_order_limit = max(1, math.floor(last_order * (1 - spread_order_percent)))
    upper_order_limit = min(math.ceil(last_order * (1 + spread_order_percent)), 35) + 1

    data_for_test = get_trading_data(token=TOKEN, figi=figi, delta_day=14, interval_time=interval_time)
    data_for_test = preprocess_data_for_predict(data=data_for_test, selected_features=['open', 'close', 'high', 'low'])
    data_for_training = get_trading_data(token=TOKEN, figi=figi, delta_day=upper_day_limit + 4,
                                         interval_time=interval_time)

    error = []

    for day in range(lower_day_limit, upper_day_limit):
        print(f'Model {name_model} training : day = {day}')

        for order in range(lower_order_limit, upper_order_limit):
            data = get_data_for_training(data=data_for_training, delta_day=day)

            data = preprocess_data_for_predict(data=data, selected_features=['open', 'close', 'high', 'low'])
            model = fit_model(data=data, order=order)

            lower_num_values_for_predict = max(math.floor(last_num_values_for_predict * (1 - spread_num_values_for_predict)), 1)
            upper_num_values_for_predict = math.ceil(last_num_values_for_predict * (1 + spread_num_values_for_predict)) + 1

            if lower_num_values_for_predict <= order:
                lower_num_values_for_predict = order

            for num_values_for_predict in range(lower_num_values_for_predict, upper_num_values_for_predict):
                for num_predictions in range(1, last_num_predictions + 1):
                    test_data = get_test_data(data=data_for_test, num_values_in_array=num_values_for_predict, num_predictions=num_predictions)

                    local_error = []

                    for values_stock, real_future_values in test_data:
                        predict_future_values = predict_next_values(model=model, data=values_stock, steps=num_predictions)
                        if type(predict_future_values) != str:
                            local_error.append(calc_local_error(predict_data=predict_future_values, real_data=real_future_values))

                    avg_local_error = calc_avg_local_error(local_error=local_error)
                    error.append({
                        "day": day,
                        "order": order,
                        "num_values_for_predict": num_values_for_predict,
                        "num_predictions": num_predictions,
                        "absolute_error": avg_local_error['absolute_error'],
                        "relative_error": avg_local_error['relative_error'],
                        "mean_absolute_error": np.mean(avg_local_error['absolute_error']),
                        "mean_relative_error": np.mean(avg_local_error['relative_error'])
                    })

    best_config = get_best_config_model(error=error)

    data = get_data_for_training(data=data_for_training, delta_day=best_config['day'])

    data = preprocess_data_for_predict(data=data, selected_features=['open', 'close', 'high', 'low'])
    best_model = fit_model(data=data, order=best_config['order'])

    save_model(best_model, path_model)

    print(f'Model {name_model} completed retraining')

    return {
        "path_model": path_model,
        "name_model": name_model,
        "figi": figi,
        "interval_time": interval_time,
        "day_for_training": best_config['day'],
        "order": best_config['order'],
        "mean_relative_error": best_config['mean_relative_error'],
        "mean_absolute_error": best_config['mean_absolute_error'],
        "num_values_for_predict": best_config['num_values_for_predict'],
        "num_predictions": best_config['num_predictions'],
        "spread_days_percent": spread_days_percent,
        "spread_order_percent": spread_order_percent,
        "spread_num_values_for_predict": spread_num_values_for_predict
    }


def get_test_data(data, num_values_in_array, num_predictions):
    arrays = []
    for start_index in range(len(data) - (num_values_in_array + num_predictions)):
        end_index_values = start_index + num_values_in_array
        end_index_predictions = end_index_values + num_predictions
        values_array = data[start_index:end_index_values]
        predictions_array = data[end_index_values:end_index_predictions]
        arrays.append([values_array, predictions_array])
    return arrays


def calc_local_error(predict_data, real_data):
    absolute_error = calc_local_absolute_error(predict_data=predict_data, real_data=real_data)
    relative_error = calc_local_relative_error(predict_data=predict_data, real_data=real_data)

    return {'absolute_error': absolute_error, 'relative_error': relative_error}


def calc_local_relative_error(predict_data, real_data):
    delta_values = np.abs(real_data - predict_data)

    if delta_values.ndim == 1:
        relative_error = delta_values / np.abs(real_data)
        return np.mean(relative_error)
    else:
        relative_error = delta_values / np.abs(real_data)
        return np.mean(relative_error, axis=0)


def calc_local_absolute_error(predict_data, real_data):
    delta_values = np.abs(real_data - predict_data)

    if delta_values.ndim == 1:
        return np.mean(delta_values)
    else:
        return np.mean(delta_values, axis=0)


def calc_avg_local_error(local_error):
    absolute_errors = [entry['absolute_error'] for entry in local_error]
    relative_errors = [entry['relative_error'] for entry in local_error]

    avg_absolute_error = np.mean(absolute_errors, axis=0)
    avg_relative_error = np.mean(relative_errors, axis=0)

    return {'absolute_error': avg_absolute_error, 'relative_error': avg_relative_error}


def get_best_config_model(error):
    best_entry = min(error, key=lambda x: x['mean_relative_error'])
    return best_entry


def get_data_for_training(data, delta_day):
    time_col = data['time']
    start_time = time_col.iloc[-1] - pd.Timedelta(days=delta_day)

    selected_data = data[data['time'] >= start_time]
    return selected_data
