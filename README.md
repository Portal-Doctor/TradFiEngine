# TradFiBot — Automated Crypto Trading Engine

A reinforcement learning–based auto-trader that learns to trade alt-coins using historical data, paper trading, and live execution via Coinbase.

## How It Works

1. **Learning** — Train a PPO agent on multi-asset historical OHLCV. The model observes all symbols at once and outputs allocation weights across assets.
2. **Paper Trading** — Evaluate the trained model on a historical date range. Runs sequentially (no random starts), reports Total Return, Max Drawdown, Sharpe.
3. **Live Trading** — Connect to Coinbase, wait for candle close, get model signal, execute via broker. Orders are logged to `data/orders.db` for the dashboard.

```
Multi-Symbol Data → load_multi_symbol → MultiAssetTradingEnv → PPO → Paper (historical) → Live (CCXT/Coinbase)
                                              ↑
                                    MACD, RSI, fees, aligned timestamps
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows: use .venv/Scripts/activate on Git Bash
pip install -r requirements.txt
```

**Training** (multi-asset, fetches data via CCXT):
```bash
python scripts/train.py
```

**Paper trading** (historical evaluation, default 2022-01-01–2022-06-30):
```bash
python scripts/paper_trade.py
```

Or run `python main.py` for the interactive menu.

## Commands

### Main Menu (`python main.py`)

| Choice | Phase         | What it does                                      |
|--------|---------------|---------------------------------------------------|
| 1      | Learning      | Trains the model and saves to `checkpoints/`      |
| 2      | Paper Trading | Runs paper trade, prints results                  |
| 3      | Real Trading  | Connects to Coinbase, dry-run or live             |
| 0      | Exit          | Quits                                             |

### CLI (recommended for multi-asset)

| Command | Purpose |
|---------|---------|
| `python scripts/train.py` | Phase 1: Train multi-asset model (symbols from config) |
| `python scripts/paper_trade.py` | Phase 2: Evaluate on historical date range |
| `python scripts/live_trade.py [options]` | Phase 3: Connect + dry-run |
| `python scripts/live_loop.py [options]` | Phase 3: Full live loop (candle-aligned) |
| `python scripts/train_walkforward.py [options]` | Rolling-window training |
| `streamlit run scripts/dashboard.py` | Monitoring dashboard |

### Command Options

**train.py** — Multi-asset training (symbols from `config.env.symbols`)
- `--data PATH` — OHLCV CSV path, or `fetch` (default)
- `--timeframe TF` — 1h, 4h, etc. (default: 1h)
- `--limit N` — Bars to fetch per symbol (default: 2000)
- `--timesteps N` — Training steps (default: from config)
- `--save PATH` — Model save path (default: checkpoints/tradfibot)

**paper_trade.py** — Historical evaluation on sliced date range
- `--model PATH` — Trained model (default: checkpoints/tradfibot)
- `--start YYYY-MM-DD` — Start date (default: 2022-01-01)
- `--end YYYY-MM-DD` — End date (default: 2022-06-30)
- `--timeframe TF` — Candle timeframe (default: 1h)
- `--limit N` — Bars to fetch per symbol (default: 2000)

**live_trade.py**
- `--symbol SYMBOL` — Default: BTC-USDT
- `--dry-run` — Connect only, no orders
- `--model PATH` — Model path

**live_loop.py**
- `--symbol SYMBOL` — Single symbol (legacy)
- `--symbols LIST` — Comma-separated pairs, e.g. BTC-USDT,ETH-USDT,SOL-USDT (default: from config `symbols.live`)
- `--timeframe TF` — Default: 1h
- `--model PATH` — Model path
- `--broker ccxt|coinbase` — Broker choice
- `--dry-run` — Log would-be trades, no orders
- `--threshold F` — Hysteresis for to_signal (default: 0.05)

**train_walkforward.py**
- `--data`, `--symbol`, `--timeframe`, `--limit`
- `--train-months N` — Train window (default: 6)
- `--test-months N` — Test window (default: 1)
- `--timesteps N` — Per fold (default: 50000)
- `--save-prefix PATH` — e.g. checkpoints/wf_ (saves fold1, fold2, …)

**dashboard** — No args. Reads `data/orders.db`. Run in a separate terminal.

---

## Modes (Phases)

### 1. Learning (Training)

Train the multi-asset PPO agent on aligned historical OHLCV. Symbols come from `config.env.symbols` (e.g. BTC-USDT, ETH-USDT, SOL-USDT, …).

**How to run:**
```bash
python scripts/train.py
# Or with overrides:
python scripts/train.py --data fetch --timeframe 1h --limit 2000 --timesteps 100000 --save checkpoints/tradfibot
```

**Training loop:**

- **Total timesteps:** Default 100,000 (config: `training.total_timesteps`).
- **Episodes:** Each episode runs for `episode_bars` (default 500 ≈ 21 days at 1h). Random start per episode.
- **Data:** `load_multi_symbol` fetches all symbols via DataIngestor, aligns on timestamps, adds indicators.

**Observation:** `(num_symbols, window_size, num_features)` — all symbols at once, rolling window per symbol.

**Action:** `(num_symbols,)` — allocation weights (0–1) across assets, normalized so sum ≤ 1; remainder in cash.

**Rewards:** Portfolio equity change (minus fees). No lookahead: agent sees only past bars.

---

### 2. Paper Trading (Historical Evaluation)

Evaluates the trained multi-asset model on a **historical date range**. Runs sequentially (no random starts) and reports metrics.

**How to run:**
```bash
python scripts/paper_trade.py
# Or with a custom period:
python scripts/paper_trade.py --model checkpoints/tradfibot --start 2023-01-01 --end 2023-12-31
```

**Process:**

- Loads multi-symbol data via `load_multi_symbol`, slices by `--start` and `--end`.
- Creates `MultiAssetTradingEnv` in `paper_mode=True` (deterministic, runs full period).
- Runs `model.predict(obs, deterministic=True)` bar-by-bar; no real orders.
- Computes Total Return, Max Drawdown, Sharpe ratio.

**Use:** Validate the model on out-of-sample history before going live. If no model is found, runs with random actions (for testing).

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
│   ├── core/         # Circuit breaker, order tracker, SQLite logger
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

### How to Get Coinbase API Keys

See [How to Get Coinbase API Keys](https://www.youtube.com/watch?v=nuuiMkkzWxc) for a step-by-step video walkthrough.

### Stable Baselines 3 Model Management

See [Stable Baselines 3 Model Management](https://www.youtube.com/watch?v=dLP-2Y6yu70) for saving, loading, and managing PPO models.

### Telegram Bot Setup

For push alerts on trades, circuit breaker, errors, and warnings:

1. Open Telegram and search for **@BotFather**. Send `/newbot` and follow the prompts to create a bot and get your **API Token**.
2. Search for **@userinfobot** to get your **Chat ID**.
3. Add these to your `.env` file:

```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**CCXT (multi-exchange):** Create `.env`:

```
COINBASE_API_KEY=...
COINBASE_SECRET=...
COINBASE_PASSPHRASE=...
```

**Coinbase CDP (Advanced Trade):** Uses different keys. Install `pip install coinbase-advanced-py`:

```
COINBASE_API_KEY="organizations/{org_id}/apiKeys/{key_id}"
COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----\n"
```

Or pass `key_file="path/to/cdp_api_key.json"` to `CoinbaseBroker`. Map internal symbols: `BTC-USDT` → `BTC-USD`.

## Typical Workflow

1. **Train:** `python scripts/train.py` — trains multi-asset model on config symbols, saves to `checkpoints/tradfibot`
2. **Paper:** `python scripts/paper_trade.py` — evaluates on 2022-01-01–2022-06-30; or `--start` / `--end` for custom range
3. **Validate:** Run paper on different date ranges; check Total Return, Max Drawdown, Sharpe
4. **Live (dry-run):** `python scripts/live_trade.py --dry-run` to test connectivity
5. **Live loop:** `python scripts/live_loop.py --timeframe 1h` (or `--dry-run` first)
6. **Monitor:** `streamlit run scripts/dashboard.py` in another terminal

### Docker Workflow

All services share the `checkpoints/` volume so models flow from training → paper → live. **Because all three stages are in one file, you must manage which one is active.**

| Phase   | Command                                | Purpose                        |
|---------|----------------------------------------|--------------------------------|
| Train   | `docker compose run --rm trainer`      | Run training, save model, exit |
| Paper   | `docker compose up -d paper-trader`    | Test with simulated capital    |
| Live    | `docker compose up -d live-engine dashboard` | Go live + monitor          |

**Step-by-step:**

1. **Train:** Run `docker compose run --rm trainer`. Wait for it to finish.
2. **Test:** Run `docker compose up -d paper-trader`. Monitor the logs to ensure the model makes good decisions.
3. **Go Live:** Run `docker compose stop paper-trader`, then `docker compose up -d live-engine` (and optionally `dashboard`).

The trainer saves to `/app/checkpoints/`; paper-trader and live-engine load from the same path. `TRADING_MODE=PAPER` uses `PaperBroker`; `TRADING_MODE=LIVE` uses CCXT/Coinbase.

### Docker / Mac Optimization

**Optimization strategy for Intel Mac**

1. **File performance (bind mounts)**  
   macOS can slow down when sharing files between the host (APFS) and the Docker VM (Linux) due to bind mount latency.

   - Place the TradFiBot project in your Home directory (`/Users/yourname/...`).
   - In Docker Desktop: **Settings → Resources → File Sharing**, make sure your Home directory is included.
   - If writes to `orders.db` are slow, move `data/` off the bind mount and onto a Docker **Named Volume**.

2. **Resource allocation (CPU/RAM)**  
   On Intel Macs, limit Docker’s usage so it doesn’t starve the host.

   - Go to **Docker Desktop → Settings → Resources**.
   - **CPUs:** Use 2–3 of the available cores.
   - **Memory:** Allocate 4–6 GB.

### Retraining & Resetting

To retrain the model, clear old data and run the trainer again. Because `data/` is on your Mac and mapped into Docker, you can edit it directly.

**1. Stop active services**  
Stop all containers before changing data:

```bash
docker compose down
```

**2. Erase paper trade data**  
Simulated paper orders live in `data/paper/orders.db`. Remove it to reset paper history:

```bash
rm -f data/paper/orders.db
```

To also clear live order history:

```bash
rm -f data/orders.db
```

To fully retrain and discard the old model, remove checkpoints:

```bash
rm -rf checkpoints/*
```

**3. Run retraining**  
Start training again:

```bash
docker compose run --rm trainer
```

### Health Endpoint

The live-engine and paper-trader expose a health check at `/health` for monitoring:

- **Live:** `curl http://localhost:5000/health`
- **Paper:** `curl http://localhost:5001/health`

Response: `{"status": "healthy", "mode": "LIVE"}` (or `"PAPER"`).

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

**Executor (Coinbase-ready)**

- **Precision:** Amount rounded to exchange `base_increment` (avoids API rejection).
- **Pre-flight:** Balance check before order; avoids "Insufficient Funds" loop.
- **Retries:** Configurable retries with backoff on execution failure.

**Coinbase integration checklist**

| Feature         | Importance | Implementation                            |
| --------------- | ---------- | ----------------------------------------- |
| Product mapping | High       | Map internal BTC to Coinbase BTC-USD/USDC |
| Maker vs taker  | Medium     | Limit orders → maker fees; Market → taker |
| Time in force   | Medium     | GTC or IOC for limit orders               |

**Architecture**

- **Producer-Consumer:** `DataIngestor` (price feeds) → `StrategyBrain` (ML signal) → `Executor` (orders).
- **StrategyBrain live/paper:** If trained on 1h candles, call `predict()` only once per completed candle. For `to_signal()`, compute `current_pct = (position * price) / total_portfolio_value` from live balance.
- **StateBuffer:** Use `StateBuffer(feature_cols, window_size)` to maintain the last N bars; `append(new_row)` then `get_obs()` so live obs matches training exactly.
- **Scaler:** Save `StandardScaler`/`MinMaxScaler` from training and pass to `StrategyBrain(scaler=...)` for live normalization.

**PaperBroker execution monitoring:** Set `log_executions=True` (default); logs each simulated trade with mid price and slippage for spread comparison.

**Executor + Strategy sync (main loop):**

- Before **buy**: Executor pre-flight checks sufficient USD/USDT; avoid "Insufficient Funds" from fees or held orders.
- Before **sell**: Executor pre-flight checks sufficient base asset (e.g. BTC) to sell.

**Live main loop:** `python scripts/live_loop.py --symbol BTC-USDT --timeframe 1h` — waits for candle close (temporal alignment); `--dry-run` logs would-be trades.

**Dashboard:** `streamlit run scripts/dashboard.py` — KPIs (Total Return %, Win Rate, Daily Max Drawdown), equity curve, Asset Performance (PnL + Slippage per symbol), Trade Markers, Exposure Ratio (stay under 95%), position pie (live prices), Recent Executions with Slippage ($) column. Auto-detects data source: reads from `data/paper/orders.db` when paper-trader is active, `data/orders.db` when live-engine is active (or set `DASHBOARD_MODE=PAPER|LIVE` to override).

**SQLiteLogger** — Connective tissue between Executor and Dashboard. Logs every trade with fee and status. For sells, computes `realized_pnl` via AVG cost: `((Sell Price − Avg Buy Price) × Amount) − Sell Fee`.

**Telemetry:** Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` for push alerts on trades, circuit breaker, and errors.

**orders.db schema** (SQLiteLogger + dashboard + OrderTracker):

| Column       | Type | Description                          |
|--------------|------|--------------------------------------|
| order_id     | TEXT | Primary key                          |
| created_at   | TEXT | ISO 8601                             |
| symbol       | TEXT | e.g. BTC-USD                         |
| side         | TEXT | buy \| sell                          |
| amount       | REAL | Base currency                        |
| price        | REAL | Execution price                      |
| fee          | REAL | Fee paid (default 0)                 |
| realized_pnl | REAL | Realized P&L (default 0)             |
| status       | TEXT | open \| filled \| cancelled \| failed |

**Advanced Trade checklist:** Market buy = quote_size (USD); market sell = base_size (crypto). Symbol mapping: `BTC-USDT` → `BTC-USD`. Use `uuid.uuid4()` for client_order_id (implemented). Rate limits ~30 req/s; keep polling low.

**Kill switch:** On 403 Forbidden or Insufficient Funds, the live loop halts trading (target 0% = all cash) and logs. Restart manually to retry.

**Live Reality (hardening):**

| Factor        | Description                                              | Mitigation                                  |
|---------------|----------------------------------------------------------|---------------------------------------------|
| FIFO vs. AVG  | Exchanges track tax lots via First-In-First-Out          | AVG used for bot performance; use FIFO for tax |
| Partial fills | Market order might fill in multiple chunks at diff prices | Executor should aggregate partial fills before logging |
| Slippage      | Fill price often worse than signal price in volatile mkts | Dashboard: Slippage = Signal Price − Fill Price |

## Configuration

Key settings in `config/default.yaml`:

| Setting                         | Default | Description                                |
| ------------------------------- | ------- | ------------------------------------------ |
| `env.symbols`                   | BTC-USDT, ETH-USDT, … | Symbols for training and paper  |
| `env.window_size`               | 60      | Lookback bars in observation               |
| `env.episode_bars`              | 500     | Bars per training episode (~21 days at 1h) |
| `env.starting_cash`             | 10000   | Initial capital                            |
| `training.data_source`          | fetch   | `fetch` (CCXT) or CSV path                 |
| `training.total_timesteps`      | 100000  | Total env steps per training run           |
| `fees.taker`                    | 0.006   | Taker fee (market orders)                  |

## Phase Isolation (Idempotency)

Each phase uses only its own paths. Running one never affects the others.

| Phase | Writes | Reads |
|-------|--------|-------|
| 1. Learning | `checkpoints/` | config, data (fetch/CSV) |
| 2. Paper Trading | `data/paper/orders.db` | config, `checkpoints/`, fetch |
| 3. Live Trading | `data/orders.db` | config, `checkpoints/` |

Configure paths in `config/default.yaml` under `paths`.

## Safety

- Train and paper-trade before going live.
- Use a Coinbase sub-profile with capped funds.
- Start with `--dry-run` when testing live connectivity.
