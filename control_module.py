import json
from trading_bot import trading_bot, get_interval_minutes
from retraining_module import retraining_model
from datetime import datetime, time
import time as t
from model_func import load_model
from my_client_config import TOKEN, account_id
from tinkoff_api_request import check_status_order, cansel_order
import threading
from concurrent.futures import ThreadPoolExecutor
import os


def main():
    while True:
        while is_weekday():
            while exchange_open():
                start_bots(datetime.now().time())

            if now_night():
                retraining_models()


def start_bots(current_time):
    bots_to_start = []
    config_bots = get_config_bots()

    def error_handler(exception, bot_name):
        print(f"An error occurred in thread for bot {bot_name}: {exception}")

    with ThreadPoolExecutor() as executor:
        for bot_name in config_bots:
            bot_config = config_bots[bot_name]
            parameters_model = bot_config["parameters_model"]

            interval_minutes = get_interval_minutes(parameters_model["interval_time"])
            if current_time.minute % interval_minutes == 0 and current_time.second == 4:
                bots_to_start.append(bot_name)

        t.sleep(1)

        if bots_to_start:
            futures = []
            for bot_name in bots_to_start:
                future = executor.submit(start_bot, bot_name, config_bots)
                future.add_done_callback(lambda f: error_handler(f.exception(), bot_name))
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()


def start_bot(bot_name, all_configs):
    bot_config = all_configs[bot_name]
    start_config = bot_config

    print(f'Start config bot: {bot_config["parameters_model"]["name_model"]}\n'
          f'{bot_config}')

    model = load_model(bot_config['parameters_model']['path_model'])

    order_data = trading_bot(model=model,
                             token=TOKEN,
                             account_id=account_id,
                             config_bot=bot_config)

    if order_data != "NOT ORDER":
        if not os.path.exists(bot_config['path_order_dump']):
            with open(bot_config['path_order_dump'], 'w') as file:
                json.dump({}, file)

        try:
            with open(bot_config['path_order_dump'], 'r') as file:
                existing_data = json.load(file)
        except Exception as e:
            print(f"An error occurred while loading JSON: {e}")
            existing_data = {}

        t.sleep(60 * get_interval_minutes(bot_config['parameters_model']['interval_time']) - 15)

        for order_id in order_data:
            order_info = order_data[order_id]

            status = check_status_order(token=TOKEN, account_id=account_id, order_id=order_id)

            if status == "FILL":
                if order_info['buy_or_sell'] == "BUY":
                    bot_config['limitations_cash']['current_count_money'] -= order_info['price']
                    bot_config['limitations_cash']['current_count_lot'] += bot_config['limitations_cash']['quantity']
                    print(f'Исполнена покупка {order_id}')

                if order_info['buy_or_sell'] == "SELL":
                    bot_config['limitations_cash']['current_count_money'] += order_info['price']
                    bot_config['limitations_cash']['current_count_lot'] -= bot_config['limitations_cash']['quantity']
                    print(f'Исполнена продажа {order_id}')

            else:
                cansel_order(token=TOKEN, account_id=account_id, order_id=order_id)
                print(f'Отменена заявка {order_id}')
                status = "CANCELLED"

            order_info['status'] = status

            try:
                existing_data[order_id] = order_info
            except Exception as e:
                print(f'Error occurred while adding order: {e}')

        try:
            with open(bot_config['path_order_dump'], 'w') as file:
                json.dump(existing_data, file, indent=4)
        except Exception as e:
            print(f'Новые ордера не добавлены: {e}')

        print(f'start_config: {start_config}')
        print(f'new_config: {bot_config}')

        try:
            with open(bot_config['path_to_config'], 'w') as config_file:
                json.dump(bot_config, config_file, indent=4)
        except Exception as e:
            print(f'Новый конфиг не сохранён: {e}')

def get_config_bots():
    with open('path_to_config_bots.json', 'r') as file:
        config_paths = json.load(file)

    config_bots = {}

    for bot_name, config_path in config_paths.items():
        with open(config_path, 'r') as file:
            bot_config = json.load(file)
            config_bots[bot_name] = bot_config

    return config_bots


def is_weekday():
    today = datetime.now().weekday()
    return today < 5


def exchange_open():
    current_time = datetime.now().time()

    if time(10, 10) <= current_time < time(14):
        return True
    elif time(14, 10) <= current_time < time(18, 45):
        return True
    elif time(19, 10) <= current_time < time(23, 50):
        return True

    return False


def now_night():
    current_time = datetime.now().time()

    if time(23, 59) == current_time:
        return True

    return False


def retraining_models():
    config_bots = get_config_bots()

    with ThreadPoolExecutor() as executor:
        model_args = [(bot_name, config_bots[bot_name]) for bot_name in config_bots]
        executor.map(retraining_one_model, model_args)


def retraining_one_model(args):
    bot_name, bot_config = args
    parameters_model = bot_config["parameters_model"]

    bot_config['parameters_model'] = retraining_model(parameters_model)

    bot_config['limitations_technical']['model_accuracy']         = bot_config['parameters_model']['mean_relative_error']
    bot_config['limitations_technical']['num_values_for_predict'] = bot_config['parameters_model']['num_values_for_predict']
    bot_config['limitations_technical']['num_predictions']        = bot_config['parameters_model']['num_predictions']

    with open(bot_config['path_to_config'], 'w') as config_file:
        json.dump(bot_config, config_file, indent=4)


if __name__ == "__main__":


    main()
    # t.sleep(3)
    # config_bots = get_config_bots()
    #
    # for bot_name in config_bots:
    #     bot_config = config_bots[bot_name]
    #     print(bot_config)
    #
    #     bot_config['limitations_cash']['current_count_money'] = 10
    #     with open(bot_config['path_to_config'], 'w') as config_file:
    #         json.dump(bot_config, config_file, indent=4)
    #
    #     config_bots = get_config_bots()
    #     bot_config = config_bots[bot_name]
    #     print(bot_config)

        # existing_data = []
        #
        # print(bot_config['path_order_dump'])
        #
        #
        # with open(bot_config['path_order_dump'], 'r') as file:
        #     existing_data = json.load(file)
        #
        # print(existing_data['asd'])