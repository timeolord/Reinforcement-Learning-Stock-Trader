import datetime
import logging
import os
import pickle
import time

import pandas as pd
import yfinance as yf
from muzero import muzerobot
from muzero.games.trading import STOCK_LIST, TRADE_DATE, INITIAL_MONEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def dateToString(date):
    return date.strftime('%Y-%m-%d')


def getTickerHistory(tickerName, date, dayInterval, interval, retries=3):
    delta = datetime.timedelta(days=dayInterval)
    ticker = yf.Ticker(tickerName)
    for attempt in range(retries):
        try:
            history = ticker.history(start=dateToString(date - delta), end=dateToString(date), interval=interval)
            pickle.dump(history, open(f"Stocks/{tickerName.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "wb"))
            return history
        except Exception as e:
            logger.warning(f"attempt {attempt + 1}/{retries} failed fetching {tickerName}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    raise RuntimeError(f"failed to fetch {tickerName} after {retries} attempts")


dateList = [datetime.date(day=1, month=12, year=2019), datetime.date(day=30, month=1, year=2020),
            datetime.date(day=30, month=3, year=2020), datetime.date(day=29, month=5, year=2020),
            datetime.date(day=28, month=7, year=2020), datetime.date(day=26, month=9, year=2020),
            datetime.date(day=25, month=11, year=2020), datetime.date(day=24, month=1, year=2021)]

today = datetime.date.today()


def getData():
    interval = "1m"
    dayInterval = 7
    for stocks in STOCK_LIST:
        for dates in dateList:
            try:
                pickle.load(open(f"Stocks/{stocks.upper()}History{dates.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
            except Exception:
                try:
                    getTickerHistory(stocks, dates, dayInterval, interval)
                    logger.info(f"fetched {stocks} for {dates}")
                except RuntimeError as e:
                    logger.error(str(e))
                time.sleep(5)


def getDataForToday():
    interval = "1h"
    dayInterval = 60
    dates = today
    for stocks in STOCK_LIST:
        try:
            pickle.load(open(f"Stocks/{stocks.upper()}History{dates.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
        except Exception:
            try:
                getTickerHistory(stocks, dates, dayInterval, interval)
                logger.info(f"fetched {stocks} for {dates}")
            except RuntimeError as e:
                logger.error(str(e))
            time.sleep(5)


def openData(tickerName, date, interval):
    history = pickle.load(open(f"Stocks/{tickerName.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
    return history


def combineList():
    date = dateList[0]
    interval = "1y"
    for stocks in STOCK_LIST:
        listy = []
        for dates in dateList:
            data = openData(stocks, dates, "1h")
            try:
                data = data.tz_localize('US/Eastern')
            except TypeError:
                data = data.tz_convert(None)
            listy.append(data)
        history = pd.concat(listy)
        pickle.dump(history, open(f"Stocks/{stocks.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "wb"))


test = False


def runModel():
    muzero = muzerobot.MuZero("trading")
    # muzero.load_model("path/to/model.checkpoint", "path/to/replay_buffer.pkl")
    if not test:
        muzero.train()
    else:
        muzero.test()


def fixDataFrames():
    date = datetime.date(year=2019, month=12, day=1)
    before = "AAPL"
    after = "NIO"
    history1 = openData(before, date, "1y")
    history = openData(after, date, "1y")
    history = history.iloc[len(history1) * 2: len(history) + 100]
    pickle.dump(history, open(f"Stocks/{after.upper()}History{date.strftime('%Y_%m_%d')}_1y.p", "wb"))
    logger.info("fixed NIO dataframe")


if __name__ == "__main__":
    runModel()

