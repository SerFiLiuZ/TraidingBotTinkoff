from datetime import timedelta
import pandas as pd

from decimal import Decimal

from tinkoff.invest import (
    Client,
    CandleInterval,
    OrderType,
    OrderDirection
)

from tinkoff.invest.utils import (
    decimal_to_quotation,
    now
)


def get_trading_data(token: str, figi: str, delta_day: int, interval_time: str):
    interval = get_candle_interval(interval_time)
    candles_data = []
    with Client(token) as client:
        for candle in client.get_all_candles(
                figi=figi,
                from_=now() - timedelta(days=delta_day),
                interval=interval,
        ):
            candles_data.append({
                'time': candle.time,
                'open': money_to_decimal(candle.open),
                'close': money_to_decimal(candle.close),
                'high': money_to_decimal(candle.high),
                'low': money_to_decimal(candle.low),
                'volume': candle.volume
            })

    df = pd.DataFrame(candles_data)
    df['time'] = pd.to_datetime(df['time'])
    filtered_df = df[~df['time'].dt.weekday.isin([5, 6])]

    return filtered_df


def create_order(token: str, account_id: str, figi: str, quantity: int,
                 price: float, direction_type: str, order_t: str = "LIMIT"):
    with Client(token) as client:
        if order_t == "LIMIT":
            order_type = OrderType.ORDER_TYPE_LIMIT

        if order_t == "MARKET":
            order_type = OrderType.ORDER_TYPE_MARKET

        if order_t == "BESTPRICE":
            order_type = OrderType.ORDER_TYPE_BESTPRICE

        if direction_type == "BUY":
            direction = OrderDirection.ORDER_DIRECTION_BUY

        if direction_type == "SELL":
            direction = OrderDirection.ORDER_DIRECTION_SELL

        try:
            response = client.orders.post_order(
                figi=figi,
                quantity=quantity,
                price=decimal_to_quotation(Decimal(price)),
                direction=direction,
                account_id=account_id,
                order_type=order_type,
            )
            print(f'create order: {response.order_id} : direction_type: {direction_type} : price : {price}')
            return response.order_id

        except Exception as error:
            print(f"Posting trade limit order failed. Exception: {error}")
            return 0


def check_status_order(token: str, account_id: str, order_id: str) -> str:
    """
    :param token:
    :param account_id:
    :param order_id:
    :return: FILL - заявка исполнена, REJECTED - отклонена, CANCELLED - отменена пользователем, NEW - новая, PARTIALLYFILL - частично исполнена
    """
    with Client(token) as client:
        try:
            order_state = str(client.orders.get_order_state(account_id=account_id, order_id=order_id)
                              .execution_report_status)
            status_parts = order_state.split("_")
            status_code = status_parts[-1]
        except Exception as e:
            print(f'ERROR: check_status_order {e}')
            status_code = "NOT_FOUND"

        return status_code


def cansel_order(token: str, account_id: str, order_id: str):
    with Client(token) as client:
        try:
            client.orders.cancel_order(account_id=account_id, order_id=order_id)
        except Exception as error:
            print(f"Failed to cancel orders. Error: {error}")


def get_candle_interval(interval_time: str):
    if interval_time == '1m':
        return CandleInterval.CANDLE_INTERVAL_1_MIN
    if interval_time == '2m':
        return CandleInterval.CANDLE_INTERVAL_2_MIN
    if interval_time == '3m':
        return CandleInterval.CANDLE_INTERVAL_3_MIN
    if interval_time == '5m':
        return CandleInterval.CANDLE_INTERVAL_5_MIN
    if interval_time == '10m':
        return CandleInterval.CANDLE_INTERVAL_10_MIN
    if interval_time == '15m':
        return CandleInterval.CANDLE_INTERVAL_15_MIN
    if interval_time == '30m':
        return CandleInterval.CANDLE_INTERVAL_30_MIN
    if interval_time == '1h':
        return CandleInterval.CANDLE_INTERVAL_HOUR
    if interval_time == '2h':
        return CandleInterval.CANDLE_INTERVAL_2_HOUR
    if interval_time == '4h':
        return CandleInterval.CANDLE_INTERVAL_4_HOUR
    if interval_time == '24h':
        return CandleInterval.CANDLE_INTERVAL_DAY


def money_to_decimal(money):
    return money.units + money.nano/10**9
