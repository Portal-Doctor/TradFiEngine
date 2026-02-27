# TradFiBot — Automated Crypto Trading Engine

A reinforcement learning–based auto-trader that learns to trade alt-coins using historical data, paper trading, and eventually live execution via Coinbase.

## Architecture

```
Historical Data → Gymnasium Env → RL Agent → Paper Trading → Live (CCXT/Coinbase)
                      ↑
              MACD, RSI, fees, arbitrage
```

## Objectives

- **Up to 10 trades/day** — Capped to reduce fees and overtrading
- **Double capital in 30 days** — Target growth (aspirational)

---

## Modes (Phases)

### 1. Learning (Training)

The engine trains an RL agent (PPO) on historical OHLCV data. The agent learns to maximize portfolio value while respecting fees and trade limits.

**Training loop:**

- **Total timesteps:** Default 100,000 environment steps (configurable via `--timesteps` or `config/training.total_timesteps`).
- **Episodes:** Each episode runs for a fixed number of bars (`episode_bars`, default 500 ≈ 21 days at 1h). The PPO algorithm runs many episodes until `total_timesteps` is reached.
- **Iterations per cycle:** One “training cycle” = one call to `model.learn(total_timesteps)`. With 500 bars/episode and 100K steps, that’s ~200 episodes per cycle.

**Historical windowing (no lookahead):**

- Each episode starts at a **random point in history**. The start bar is sampled from `[window_size, len(data) - episode_bars]`, so every episode sees a different segment.
- The agent **only observes past data** at each step. At bar `t`, the observation is the window `[t - 60, t)` of features (MACD, RSI, Bollinger, ATR, returns). Future bars are never visible.
- The environment uses the close price at bar `t` to execute the agent’s action, mimicking real-time: the agent decides before knowing bar `t+1`.

**Observation space:** Flattened window of 60 bars × N features (MACD, RSI, Bollinger, ATR, returns).

**Action space:** Continuous `[0, 1]` = target position (0 = all cash, 1 = max 95% in asset).

**Rewards:** Change in portfolio value (minus fees), with a small penalty for trades that don’t beat the minimum profit threshold.

---

### 2. Paper Trading

Validates the trained model with simulated capital before risking real funds. No real orders are placed.

**Process:**

- Loads historical data (or fetches via CCXT) for the chosen symbol and timeframe.
- Runs the trained model (or a simple hold strategy if no model) over the data.
- Simulates orders through a paper broker with config fees (taker ~0.6%, maker ~0.4%).
- Reports initial vs final balance and return %.

**Use:** Verify that the model’s learned behavior translates to positive returns on unseen history before going live.

---

### 3. Real Trading

Live execution on Coinbase via CCXT. Requires API credentials and a trained model.

**Requirements:**

- `.env` with `COINBASE_API_KEY`, `COINBASE_SECRET`, `COINBASE_PASSPHRASE`
- A trained model (e.g. `checkpoints/tradfibot`)
- Coinbase sub-profile with limited funds (recommended)

**Flow:**

- Connects to Coinbase, checks balance and current price.
- `--dry-run`: Connect and report only; no orders.
- Full live mode (when implemented): fetch latest OHLCV, compute indicators, get model action, execute via `broker.create_market_order()`, repeat on each new candle.

**Safety:** Always paper-trade first and use a sub-profile with capped capital.

---

## Project Structure

```
TradFiBot/
├── config/           # Fees, symbols, hyperparameters
├── src/
│   ├── indicators/   # MACD, RSI, Bollinger, ATR
│   ├── environment/  # Gymnasium trading env
│   ├── engine/       # DataIngestor, StrategyBrain, Executor
│   ├── core/         # Circuit breaker, order tracker
│   └── brokers/      # Paper & CCXT brokers
├── data/             # Historical data (gitignored)
├── scripts/          # Training, paper, live runners
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create `.env` for live trading (do not commit):

```
COINBASE_API_KEY=...
COINBASE_SECRET=...
COINBASE_PASSPHRASE=...
```

## Usage

**Start the engine (recommended):**

```bash
python main.py
```

Select a phase from the menu:

1. **Learning** — Train on historical data
2. **Paper Trading** — Validate with simulated capital
3. **Real Trading** — Live execution via Coinbase

**CLI (alternative):**

```bash
python scripts/train.py --data fetch --symbol BTC-USDT --timesteps 100000
python scripts/paper_trade.py --symbol BTC-USDT --model checkpoints/tradfibot
python scripts/live_trade.py --symbol BTC-USDT --dry-run
```

## Technical Improvements

**Training**

- **Look-ahead bias prevention:** Observation uses only past data `[t-60, t)`; no future bars.
- **Feature scaling:** Optional `use_log_returns` for stationarity (config: `features.use_log_returns`).
- **Walk-forward optimization:** `python scripts/train_walkforward.py --train-months 6 --test-months 1` — rolling window training.

**Paper Trading**

- **Slippage simulation:** 0.05–0.1% worse fill (config: `paper.slippage_pct`).
- **Fee accounting:** Taker/maker fees on every simulated trade.
- **Latency logging:** Time-to-execution metrics (config: `paper.log_latency`).

**Live Trading**

- **Circuit breaker:** Kill switch if daily drawdown exceeds threshold (config: `live.max_daily_drawdown_pct`).
- **API resilience:** Exponential backoff for 429/5xx (config: `live.api_retry_max`).
- **Order tracking:** SQLite for open orders; crash recovery (config: `live.order_db_path`).
- **Secrets:** `.env` + `.gitignore` (never hardcode keys).

**Architecture**

- **Producer-Consumer:** `DataIngestor` (price feeds) → `StrategyBrain` (ML signal) → `Executor` (orders).

## Configuration

Key settings in `config/default.yaml`:

| Setting                         | Default | Description                                |
| ------------------------------- | ------- | ------------------------------------------ |
| `objectives.max_trades_per_day` | 10      | Max trades per simulated day               |
| `env.episode_bars`              | 500     | Bars per training episode (~21 days at 1h) |
| `env.window_size`               | 60      | Lookback bars in observation               |
| `training.total_timesteps`      | 100000  | Total env steps per training run           |
| `fees.taker`                    | 0.006   | Taker fee (market orders)                  |

## Safety

- Train and paper-trade before going live.
- Use a Coinbase sub-profile with capped funds.
- Start with `--dry-run` when testing live connectivity.
