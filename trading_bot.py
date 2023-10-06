from tinkoff_api_request import get_trading_data, create_order, check_status_order
from model_func import preprocess_data_for_predict, predict_next_values
from my_client_config import EXCHANGE_COMMISSION
import numpy as np
import math
from datetime import datetime, timedelta, timezone


def trading_bot(model, token: str, account_id: str, config_bot):
    figi                   = config_bot['parameters_model']['figi']
    interval_time          = config_bot['parameters_model']['interval_time']
    num_values_for_predict = config_bot['limitations_technical']['num_values_for_predict']
    num_predictions        = config_bot['limitations_technical']['num_predictions']
    model_accuracy         = config_bot['limitations_technical']['model_accuracy']
    min_price_increment    = config_bot['limitations_cash']['min_price_increment']

    data = get_trading_data(token=token,
                            figi=figi,
                            delta_day=get_delta_day(num_values_for_predict, interval_time),
                            interval_time=interval_time)

    data = preprocess_data_for_predict(data=data,
                                       selected_features=['open', 'close', 'high', 'low'],
                                       count_rows=num_values_for_predict)

    predict_values = predict_next_values(model=model,
                                         data=data,
                                         steps=num_predictions)

    benefit, buy, sell = buy_sell_benefit(predict_values=predict_values,
                                          model_accuracy=model_accuracy,
                                          min_step_price=min_price_increment)
    current_time = datetime.now(timezone.utc)
    current_time = current_time.replace(second=0, microsecond=0)
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

    print(f'time: {formatted_time}\n'
          f'benefit: {benefit}\n'
          f'buy: {buy}\n'
          f'sell: {sell}\n'
          f'profit: {(sell - buy) - (sell + buy) * EXCHANGE_COMMISSION}')

    if benefit:
        min_count_lot       = config_bot['limitations_cash']['min_count_lot']
        min_count_money     = config_bot['limitations_cash']['min_count_money']
        current_count_lot   = config_bot['limitations_cash']['current_count_lot']
        current_count_money = config_bot['limitations_cash']['current_count_money']
        quantity            = config_bot['limitations_cash']['quantity']
        stock_in_lot        = config_bot['limitations_cash']['stock_in_lot']

        price_buy = buy * quantity * stock_in_lot

        ability_create_buy_order = current_count_money - price_buy > min_count_money
        ability_create_sell_order = current_count_lot - quantity >= min_count_lot

        order_data = {}

        if ability_create_buy_order:
            order_id_buy, order_buy_data = create_order_data(token=token,
                                                             account_id=account_id,
                                                             figi=figi,
                                                             quantity=quantity,
                                                             price=buy,
                                                             direction_type="BUY",
                                                             order_t="LIMIT",
                                                             stock_in_lot=stock_in_lot,
                                                             live_time_order=interval_time)

            if order_id_buy:
                order_data[f'{order_id_buy}'] = order_buy_data

        if ability_create_sell_order:
            order_id_sell, order_sell_data = create_order_data(token=token,
                                                               account_id=account_id,
                                                               figi=figi,
                                                               quantity=quantity,
                                                               price=sell,
                                                               direction_type="SELL",
                                                               order_t="LIMIT",
                                                               stock_in_lot=stock_in_lot,
                                                               live_time_order=interval_time)

            if order_id_sell:
                order_data[f'{order_id_sell}'] = order_sell_data

        if order_data:
            return order_data

    return "NOT ORDER"


def buy_sell_benefit(predict_values, model_accuracy, min_step_price):
    predict_buy = np.min(predict_values)
    predict_sell = np.max(predict_values)

    buy = round((1 + model_accuracy) * predict_buy / min_step_price) * min_step_price
    sell = round((1 - model_accuracy) * predict_sell / min_step_price) * min_step_price

    if sell - buy > (sell + buy) * EXCHANGE_COMMISSION:
        return True, buy, sell
    else:
        return False, buy, sell


def create_order_data(token, account_id, figi, quantity, price, direction_type, order_t, stock_in_lot, live_time_order):
    order_info = {}

    order_id = create_order(token=token, account_id=account_id,
                            figi=figi, quantity=quantity, price=price,
                            direction_type=direction_type, order_t=order_t)

    if order_id:
        current_time = datetime.now(timezone.utc)
        current_time = current_time.replace(second=0, microsecond=0)
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S%z')

        buy_or_sell = "BUY" if direction_type == "BUY" else "SELL"
        status = check_status_order(token=token, account_id=account_id, order_id=order_id)
        order_info = {
            "buy_or_sell": buy_or_sell,
            "price": price * quantity * stock_in_lot,
            "status": status,
            "time": formatted_time,
            "live_time_order": live_time_order
        }

    return order_id, order_info


def get_delta_day(num_values_for_predict, interval_time):
    interval_time_int = get_interval_minutes(interval_time)
    all_interval_time_in_minute = num_values_for_predict * interval_time_int

    minutes_in_a_day = 24 * 60
    delta_day = math.ceil(all_interval_time_in_minute / minutes_in_a_day)

    return adjust_delta_for_weekend(delta_day)


def adjust_delta_for_weekend(delta_day):
    current_date = datetime.now()
    target_date = current_date - timedelta(days=delta_day)

    while is_weekend(target_date):
        target_date -= timedelta(days=1)
        delta_day += 1

    return delta_day


def is_weekend(date):
    return date.weekday() >= 5


def get_interval_minutes(interval_minutes_str):
    return int(interval_minutes_str[:-1])
