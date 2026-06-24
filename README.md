# Reinforcement Learning Stock Trader

A stock trading agent trained with [MuZero](https://github.com/werner-duvaud/muzero-general) on historical Yahoo Finance data. The agent learns to trade a portfolio of 13 equities starting with $1000, using 1-minute OHLCV bars from 2019-2021.

## Overview

MuZero builds an internal world model without needing to know the rules of the environment. Here it treats the stock market as a game: the observation is recent price history, and each action is a trade decision. Self-play workers generate experience in parallel via Ray, a replay buffer stores it, and a trainer updates the network weights, all logged to TensorBoard.

Stocks traded: SPY, AAPL, NIO, F, XLF, GE, GM, T, TQQQ, QQQ, MSFT, K, C

## Project layout

```
TradingBot.py          entry point for data fetching and model launch
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

## Setup

Linux only, Ray does not support Windows or macOS.

The repo ships with a `flake.nix` and `.envrc`. With [nix](https://nixos.org/) and [direnv](https://direnv.net/) installed:

```bash
direnv allow
```

This drops you into a shell with Python, PyTorch (CUDA), Ray, TensorBoard, and all other dependencies available. `nevergrad` is not packaged in nixpkgs so it is installed into a local `.venv` automatically on first entry.

To enter the shell manually without direnv:

```bash
nix develop
```

Note: CUDA packages require accepting an unfree license. The flake handles this automatically via `config.allowUnfree = true`. If you want CPU-only PyTorch, change `torchWithCuda` to `torch` in `flake.nix`.

## Usage

Run training directly:

```bash
python TradingBot.py
```

Ray is initialized in local mode automatically, no external cluster setup needed.

Training metrics are written to the path configured in `MuZeroConfig.results_path` and can be viewed with TensorBoard:

```bash
tensorboard --logdir results/
```

The model checkpoint and replay buffer are saved to `results/` every 2 minutes during training.

To resume from a checkpoint, uncomment and update the `load_model` call in `TradingBot.py`:

```python
muzero.load_model("results/model.checkpoint", "results/replay_buffer.pkl")
```

## Data

Historical data is cached as pickle files under `Stocks/`. The repo ships with 1-year hourly bars for all 13 tickers. To fetch fresh data, call `getData()` in `TradingBot.py`, it skips any ticker/date pairs already cached under `Stocks/`.

## Credits

MuZero implementation based on [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general).
