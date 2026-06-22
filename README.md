# Reinforcement Learning Stock Trader

A stock trading agent trained with [MuZero](https://github.com/werner-duvaud/muzero-general) on historical Yahoo Finance data. The agent learns to trade a portfolio of 13 equities starting with $1000, using 1-minute OHLCV bars from 2019–2021.

## How it works

MuZero builds an internal world model without needing to know the rules of the environment. Here it treats the stock market as a game: the observation is recent price history, and each action is a trade decision. Self-play workers generate experience in parallel via Ray, a replay buffer stores it, and a trainer updates the network weights — all logged to TensorBoard.

**Stocks traded:** SPY, AAPL, NIO, F, XLF, GE, GM, T, TQQQ, QQQ, MSFT, K, C

## Project layout

```
TradingBot.py          entry point — data fetching and model launch
y_finance_env/         custom OpenAI Gym environment wrapping yfinance data
muzero/
  muzerobot.py         MuZero class, Ray worker orchestration, training loop
  models.py            representation, dynamics, and prediction networks
  self_play.py         parallel self-play workers
  trainer.py           weight update worker
  replay_buffer.py     replay buffer and reanalysis
  shared_storage.py    shared state across Ray workers
  diagnose_model.py    trajectory comparison diagnostics
Stocks/                pickled pandas DataFrames of historical price data
```

## Requirements

Linux only — Ray does not support Windows or macOS.

```
numpy torch tensorboard gym ray seaborn nevergrad yfinance optuna pandas
```

Install with:

```bash
pip install -r muzero/requirements.txt yfinance optuna pandas
```

## Usage

Start a Ray cluster, then run:

```bash
python TradingBot.py
```

Training metrics are written to `~/Music/` and can be viewed with:

```bash
tensorboard --logdir ~/Music
```

The model checkpoint and replay buffer are saved to `~/Music/model.checkpoint` and `~/Music/replay_buffer.pkl` every 2 minutes during training.

To resume from a checkpoint, uncomment and update the `load_model` call in `TradingBot.py`:

```python
muzero.load_model("~/Music/model.checkpoint", "~/Music/replay_buffer.pkl")
```

## Data

Historical data is cached as pickle files under `Stocks/`. The repo ships with 1-year daily bars for all 13 tickers. To fetch fresh 1-minute bars across the date range used for training, call `getData()` in `TradingBot.py` — it skips any ticker/date pairs already cached.

## Credits

MuZero implementation based on [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general).
