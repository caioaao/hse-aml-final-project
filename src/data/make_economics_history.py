import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import datetime
import calendar
from yfinance import Ticker
from forex_python.converter import CurrencyRates
from functools import reduce


def next_month(m):
    _, month_end = m
    first_day = month_end + datetime.timedelta(days=1)
    return (first_day, first_day.replace(
        day=calendar.monthrange(first_day.year, first_day.month)[1]))


def all_months():
    first_month = (datetime.date(2013, 1, 1), datetime.date(2013, 1, 31))
    last_month = first_month
    months = [first_month]
    for i in range(1, 35):
        last_month = next_month(last_month)
        months.append(last_month)
    return months


def build_moex_df(first_month, last_month):
    moex = Ticker('IMOEX.ME').history(start=first_month[0], end=last_month[1],
                                      interval='1mo')
    moex = moex.reset_index().drop(columns=['Date', 'Volume',
                                            'Dividends', 'Stock Splits'])
    moex.rename(columns=lambda x: 'MOEX_%s' % x.lower(), inplace=True)
    moex['date_block_num'] = np.arange(2, 35)
    return moex


def build_forex_df(forex_src, quote, base):
    open_prices = []
    close_prices = []
    for beg_month, end_month in tqdm(months):
        open_prices.append(forex_src.get_rate(quote, base,
                                              date_obj=beg_month))
        close_prices.append(forex_src.get_rate(quote, base,
                                               date_obj=end_month))
    return pd.DataFrame({'date_block_num': range(0, 35),
                         '%s%s_open' % (quote, base): open_prices,
                         '%s%s_close' % (quote, base): close_prices})


if __name__ == '__main__':
    import sys
    output_path = sys.argv[1]

    months = all_months()
    first_month = months[0]
    last_month = months[-1]

    moex = build_moex_df(first_month, last_month)

    forex_src = CurrencyRates()
    forex_dfs = [build_forex_df(forex_src, quote, 'RUB')
                 for quote in ['CNY', 'EUR', 'USD']]

    df = pd.DataFrame({'date_block_num': range(0, 35)})

    df = reduce(lambda df, df_b: df.merge(df_b, on='date_block_num',
                                          how='left', sort=False),
                forex_dfs + [moex], df)
    df.to_parquet(output_path)
