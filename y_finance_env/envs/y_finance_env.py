import datetime
import math

import gym
import pickle
from gym import spaces
import numpy as np
from dataclasses import dataclass


@dataclass
class Stock:
    sharePrice: int
    shares: int


class y_finance_env(gym.Env):

    def __init__(self, stockList, date, interval, initialMoney):
        self.date = date
        self.stockList = stockList
        self.initialMoney = initialMoney
        self.stocks = []
        self.positionsPerStock = 5
        for stocks in stockList:
            # print(os.getcwd())
            l = [None] * self.positionsPerStock
            history = pickle.load(open(f"Stocks/{stocks.upper()}History{date.strftime('%Y_%m_%d')}_{interval}.p", "rb"))
            self.stocks.append((history, l, stocks.upper()))
        self.historyLength = 24
        self.money = self.initialMoney
        self.idleCost = -10
        # sets the index chronologically
        self.index = self.historyLength
        self.baselineMoney = self.initialMoney
        self.debug = False
        self.endingMoney = 0
        self.action = [0, 0, 0, 0]
        self.tradingFee = 0.02
        self.heldPositions = 0
        # NOOP, Buy, Sell
        # if Buy 0 = 5% of total money
        # 1 = 10%
        # 2 = 15%
        # 3 = 20%
        # if sell 0 = 25%, 1 = 50%, 2 = 75%, 3 = 100%
        # 100 is for the 100 slots that it can hold a position
        # 14 is for the amount of stocks that it can choose to trade
        self.action_space = spaces.MultiDiscrete([3, 4, self.positionsPerStock, len(self.stockList)])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(stockList) * 2, self.historyLength, 1),
                                            dtype=np.float32)
        pass

    def getNaturalGrowth(self, stock):
        growth = ((stock[0].iloc[self.index].loc["Close"]
                   - stock[0].iloc[0].loc["Close"]) /
                  stock[0].iloc[0].loc["Close"]) * 100
        # print(stock[2])
        # print(stock[0].iloc[0].loc["Close"], stock[0].iloc[self.index].loc["Close"])
        return growth

    def getMarketGrowth(self):
        sum = 0
        i = 0
        for shares in self.stocks:
            growth = self.getNaturalGrowth(shares)
            # print(f"{self.stockList[i]}: {growth:.1f}")
            sum += growth
            i += 1
        val = sum / len(self.stocks)
        # print(val)
        return val

    def getTradingGrowth(self):
        netValue = self.getNetWorth()
        percentGrowth = ((netValue - self.initialMoney) / self.initialMoney) * 100
        return percentGrowth

    def reset(self):
        self.money = self.initialMoney
        for stock in self.stocks:
            stock[1].clear()
            for i in range(self.positionsPerStock):
                stock[1].append(None)
        self.index = self.historyLength
        observation = self.getObservation()
        return observation

    def getCurrentSharePrice(self):
        return self.stocks[self.action[3]][0].iloc[self.index].loc["Close"]

    def getCurrentStockPrice(self, stock):
        return self.stocks[stock][0].iloc[self.index].loc["Close"]

    def sharesValue(self, stocks):
        sum = 0
        for shares in stocks[1]:
            sum += shares.shares * self.getCurrentSharePrice()
        return sum

    def getNetWorth(self):
        sum = 0
        for stocks in self.stocks:
            for shares in stocks[1]:
                if shares is not None:
                    sum += shares.shares * stocks[0].iloc[self.index].loc["Close"]
        return sum + self.money

    def getMaxLength(self):
        length = []
        for stocks in self.stocks:
            length.append(len(stocks[0] - 1))
        return min(length) - 1

    def step(self, action):

        action = np.unravel_index(action, (3, 4, self.positionsPerStock, len(self.stockList)))
        done = False
        self.action = action
        # print(len(self.stocks))
        # print(action[0], action[1], action[2], action[3])

        # self.money -= 10

        reward = 0
        if self.debug:
            print(f"Current Step: {self.index}")
        if self.index < self.getMaxLength():
            self.index += 1
        else:
            # print(f"Remaining money: {self.money:.1f} Ticker: {self.ticker}")
            # print(len(self.shares))
            done = True
            print("Finished a game!")
            reward = self.getReward()

        if not done:
            # Buy
            if self.heldPositions == 0:
                reward = self.idleCost
            if action[0] == 1:
                self.buy(action)
            # Sell
            if action[0] == 2:
                self.sell(action)
            if reward == 0:
                reward = self.getReward()
            # print(marketGrowth)
            # print(self.getMarketGrowth())
            # print(f"{reward:.1f}%")

        # growth = self.getTradingGrowth() - self.getMarketGrowth()
        if self.debug:
            marketGrowth = self.getMarketGrowth()
            print(f"Reward {reward:.1f}, Trader Growth {self.getTradingGrowth():.1f}%,"
                  f" Market Growth {marketGrowth:.1f}%, "
                  f"Action {self.action}, "
                  f"Networth {self.getNetWorth():.1f}, "
                  f"Money {self.money:.1f}")

        # print(reward)

        info = {}
        # print(f"Money: {self.money}. Reward: {reward}")

        observation = self.getObservation()
        return observation, reward, done, info

    def getReward(self):
        marketGrowth = self.getMarketGrowth()
        traderGrowth = self.getTradingGrowth()
        return (traderGrowth - marketGrowth) / 10

    def getNetWorthGrowth(self):
        netValue = self.getNetWorth()
        percentGrowth = ((netValue - self.initialMoney) / self.initialMoney) * 100
        return percentGrowth

    def getObservation(self):
        # print(self.stocks)
        stonks = []
        arr = 0
        for stock in self.stocks:
            close = stock[0].iloc[self.index - self.historyLength: self.index]["Close"].to_numpy()
            for i in range(len(close)):
                if math.isnan(close[i]):
                    print("found Nan")
                    close[i] = 0
            stonks.append(close)
            volume = stock[0].iloc[self.index - self.historyLength: self.index]["Volume"].to_numpy()
            for i in range(len(volume)):
                if math.isnan(volume[i]):
                    print("found Nan")
                    volume[i] = 0
            stonks.append(volume)
        # return self.stocks[self.action[3]][0].iloc[self.index - self.historyLength: self.index]["Close"].to_numpy()
        arr = np.stack(stonks, axis=0)
        arr = np.expand_dims(arr, -1)
        # print(arr.shape, self.stockList, self.date)
        return arr

    def getLegalMoves(self):
        moves = []
        action = (0, 0, 0, 0)
        num = np.ravel_multi_index(action, (3, 4, self.positionsPerStock, len(self.stockList)))
        moves.append(num)
        for a in range(4):
            for b in range(self.positionsPerStock):
                for c in range(len(self.stockList)):
                    multiplier = (action[1] + 1) * 25
                    multiplier = multiplier / 100
                    moneyToSpend = self.money * multiplier
                    if self.getCurrentStockPrice(c) != 0 and moneyToSpend >= self.getCurrentStockPrice(c) and self.stocks[c][1][b] is None:
                        action = (1, a, b, c)
                        num = np.ravel_multi_index(action, (3, 4, self.positionsPerStock, len(self.stockList)))
                        moves.append(num)

        for b in range(self.positionsPerStock):
            for c in range(len(self.stockList)):
                if self.getCurrentStockPrice(c) != 0 and len(self.stocks[c][1]) != 0 and self.stocks[c][1][b] is not None:
                    action = (2, 0, b, c)
                    num = np.ravel_multi_index(action, (3, 4, self.positionsPerStock, len(self.stockList)))
                    moves.append(num)
        # for move in moves:
        #    print(np.unravel_index(move, (3, 4, self.positionsPerStock, len(self.stockList))))
        return moves

    def sell(self, action):
        selectedStock = self.stocks[action[3]]
        if len(selectedStock[1]) != 0:
            if action[2] < len(selectedStock[1]) and selectedStock[1][action[2]] is not None:
                stock = selectedStock[1].pop(action[2])
            else:
                return self.idleCost
            amountToSell = stock.shares
            percentProfit = ((self.getCurrentSharePrice() - stock.sharePrice) / stock.sharePrice) * 100
            self.money += self.getCurrentSharePrice() * stock.shares - (self.getCurrentSharePrice() * stock.shares) * \
                          self.tradingFee
            self.heldPositions -= 1
            if self.debug:
                print(
                    f"Sold {amountToSell:.0f} shares of {self.stockList[action[3]]} at "
                    f"{self.getCurrentSharePrice():.1f} and made {percentProfit:.1f}%")
        else:
            return self.idleCost

    def buy(self, action):
        multiplier = (action[1] + 1) * 25
        multiplier = multiplier / 100
        moneyToSpend = self.money * multiplier
        if moneyToSpend >= self.getCurrentSharePrice():
            amountToBuy = moneyToSpend // self.getCurrentSharePrice()
            if (self.money - (self.getCurrentSharePrice() * amountToBuy)) > 0:
                if self.stocks[self.action[3]][1][action[2]] is None:
                    self.stocks[self.action[3]][1].insert(action[2], Stock(self.getCurrentSharePrice(), amountToBuy))
                    self.money -= (amountToBuy * self.getCurrentSharePrice()) + \
                              (amountToBuy * self.getCurrentSharePrice()) * self.tradingFee
                    self.heldPositions += 1
                    if self.debug:
                        print(
                            f"Bought {amountToBuy:.0f} shares of {self.stockList[action[3]]} at {self.getCurrentSharePrice():.1f}")
        return self.idleCost

    def render(self, mode='console'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    actualDate = datetime.date(day=1, month=12, year=2019)
    stockList = ["SPY", "AAPL", "NIO", "F", "XLF", "GE", "GM", "T", "TQQQ", "QQQ", "MSFT", "K", "C"]
    env = y_finance_env(stockList, actualDate, "1y", 1000)
    obs = env.reset()
    env.debug = True
    done = False
    while not done:
        env.getLegalMoves()
        i = np.ravel_multi_index((1, 0, 0, 7), (3, 4, env.positionsPerStock, len(env.stockList)))
        obs, reward, done, _ = env.step(i)

# print(env.getMaxLength())
# print(len(env.stocks[0][0]))
#
# print(obs.shape)

# print(env.history.iloc[env.index - 24: env.index].loc["Close"])
# print(env.sharePrice)
