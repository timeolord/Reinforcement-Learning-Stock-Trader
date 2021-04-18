import datetime
import os
import pickle
import random
import warnings
from random import sample
from statistics import mean
# import tensorflow as tf
import pandas as pd
import yfinance as yf
# from stable_baselines import PPO2, A2C, ACKTR
# from stable_baselines.common import make_vec_env
# from stable_baselines.common.policies import MlpPolicy, mlp_extractor, ActorCriticPolicy, nature_cnn
# from stable_baselines.common.tf_layers import linear
from y_finance_env import y_finance_env
import time
import optuna
from muzero import muzerobot


def randomDate():
    year = random.randint(2020, 2021)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime.date(year=year, month=month, day=day)


def dateToString(date):
    return date.strftime('%Y-%m-%d')


def getTickerHistory(tickerName, date, dayInterval, interval):
    delta = datetime.timedelta(days=dayInterval)
    ticker = yf.Ticker(tickerName)
    history = ticker.history(start=dateToString(date - delta), end=dateToString(date), interval=interval)
    pickle.dump(history, open(f"{tickerName.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "wb"))
    return history


# Spares , "AMZN"

stockList = ["SPY", "AAPL", "NIO", "F", "XLF", "GE", "GM", "T", "TQQQ", "QQQ", "MSFT", "K", "C"]

dateList = [datetime.date(day=1, month=12, year=2019), datetime.date(day=30, month=1, year=2020),
            datetime.date(day=30, month=3, year=2020), datetime.date(day=29, month=5, year=2020),
            datetime.date(day=28, month=7, year=2020), datetime.date(day=26, month=9, year=2020),
            datetime.date(day=25, month=11, year=2020), datetime.date(day=24, month=1, year=2021)]

today = datetime.date(day=7, month=7, year=2021)

initialMoney = 1000


def getData():
    interval = "1m"
    dayInterval = 7
    for stocks in stockList:
        for dates in dateList:
            try:
                pickle.load(open(f"{stocks.upper()}History{dates.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
            except Exception:
                print(getTickerHistory(stocks, dates, dayInterval, interval))
                time.sleep(5)


def getDataForToday():
    interval = "1h"
    dayInterval = 60
    dates = today
    for stocks in stockList:
        try:
            pickle.load(open(f"{stocks.upper()}History{dates.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
        except Exception:
            print(getTickerHistory(stocks, dates, dayInterval, interval))
            time.sleep(5)


def openData(tickerName, date, interval):
    history = pickle.load(open(f"Stocks/{tickerName.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
    return history


def combineList():
    date = dateList[0]
    interval = "1y"
    for stocks in stockList:
        listy = []
        for dates in dateList:
            data = openData(stocks, dates, "1h")
            try:
                data = data.tz_localize('US/Eastern')
            except:
                data = data.tz_convert(None)
            listy.append(data)
        history = pd.concat(listy)
        pickle.dump(history, open(f"{stocks.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "wb"))


test = False


def runModel():
    global muzero
    if not test:
        muzero = muzerobot.MuZero("trading")
    else:
        muzero = muzerobot.MuZero("trading")
    fileName = "~/Music/model.checkpoint"
    fileName = os.path.expanduser(fileName)
    # muzero.load_model(fileName, "/home/allanlinux/Music/replay_buffer.pkl")
    if not test:
        muzero.train()
    else:
        muzero.test()


def fixDataFrames():
    global stockList
    stockList = ["SPY", "AAPL", "NIO", "F", "XLF", "GE", "GM", "T", "TQQQ", "QQQ", "MSFT", "K", "C"]
    date = datetime.date(year=2019, month=12, day=1)
    before = "AAPL"
    after = "NIO"
    history1 = openData(before, date, "1y")
    history = openData(after, date, "1y")
    history = history.iloc[len(history1) * 2: len(history) + 100]
    pickle.dump(history, open(f"Stocks/{after.upper()}History{date.strftime('%Y_%m_%d')}_1y.p", "wb"))
    print("baller")


if __name__ == "__main__":
    runModel()

