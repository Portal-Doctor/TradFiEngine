"""
Live trading main loop: DataIngestor -> StateBuffer -> StrategyBrain -> Executor.

Temporal alignment: waits for candle close before processing. Never trades on incomplete candles.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def _run_health_server(port: int = 5000) -> None:
    """Run Flask health server in background thread."""
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "mode": os.getenv("TRADING_MODE", "LIVE")}), 200

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def get_total_portfolio_value(broker, symbols: list[str], quote_currency: str = "USDT") -> float:
    """Total portfolio value: quote balance + sum of (base * price) for each symbol."""
    quote_bal = broker.get_balance(quote_currency).total
    total = quote_bal
    for symbol in symbols:
        base, _ = (symbol.split("-") + [quote_currency])[:2]
        try:
            price = broker.get_price(symbol)
            base_bal = broker.get_balance(base).total
            total += base_bal * price
        except Exception:
            pass
    return total


def get_current_exposure(broker, symbol: str, total_value: float | None = None, quote_currency: str = "USDT") -> float:
    """
    Current position as fraction of total portfolio.
    current_pct = (position * price) / total_portfolio_value
    """
    base, quote = (symbol.split("-") + [quote_currency])[:2]
    price = broker.get_price(symbol)
    bal_base = broker.get_balance(base)
    position_value = bal_base.total * price
    if total_value is None:
        bal_quote = broker.get_balance(quote)
        total_value = bal_quote.total + position_value
    if total_value <= 0:
        return 0.0
    return position_value / total_value


def calculate_order_size(
    target_pct: float,
    current_pct: float,
    total_value: float,
    price: float,
    max_position_pct: float = 0.95,
) -> tuple[str, float]:
    """
    Return (side, amount) for Executor.
    amount is in base currency (e.g. BTC).
    target_pct from model is in [0,1]; scaled by max_position_pct.
    """
    target_value = total_value * target_pct * max_position_pct
    current_value = total_value * current_pct  # position_value
    diff_value = target_value - current_value

    if abs(diff_value) < price * 0.0001:  # Min order threshold
        return "hold", 0.0

    amount = abs(diff_value) / price if price > 0 else 0.0
    side = "buy" if diff_value > 0 else "sell"
    return side, amount


def main():
    from config import load_config
    from src.engine import DataIngestor, Executor, StateBuffer, StrategyBrain
    from src.indicators import add_all_indicators
    from src.core import CircuitBreaker, OrderTracker, SQLiteLogger
    from src import telemetry

    config = load_config()
    paths = config.get("paths", {})
    default_model = str(Path(paths.get("checkpoints", "checkpoints")) / "tradfibot")

    env_cfg_pre = config.get("env", {})
    symbols_cfg = config.get("symbols", {})
    default_symbols = (
        env_cfg_pre.get("symbols")
        or symbols_cfg.get("live")
        or symbols_cfg.get("training")
        or [symbols_cfg.get("default", "BTC-USDT")]
    )
    if not isinstance(default_symbols, list):
        default_symbols = [default_symbols] if default_symbols else ["BTC-USDT"]

    parser = argparse.ArgumentParser(
        description="Live trading loop: Ingestor -> StateBuffer -> Brain -> Executor"
    )
    parser.add_argument("--symbol", type=str, help="Single symbol (legacy); use --symbols for multiple")
    parser.add_argument("--symbols", type=str, help="Comma-separated pairs, e.g. BTC-USDT,ETH-USDT,SOL-USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--broker", type=str, default="ccxt", choices=["ccxt", "coinbase"])
    parser.add_argument("--dry-run", action="store_true", help="Log only; no real orders")
    parser.add_argument("--threshold", type=float, default=0.05, help="Hysteresis for to_signal")
    args = parser.parse_args()

    # Resolve symbol list: --symbols > --symbol > config
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [s if isinstance(s, str) else str(s) for s in default_symbols]
    if not symbols:
        symbols = ["BTC-USDT"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("TradFiEngine.LiveLoop")

    # Start health check server in background
    flask_thread = threading.Thread(target=_run_health_server, kwargs={"port": 5000}, daemon=True)
    flask_thread.start()
    log.info("Health check server listening on port 5000")

    env_cfg = config.get("env", {})
    ind_cfg = config.get("indicators", {})
    feat_cfg = config.get("features", {})
    live_cfg = config.get("live", {})

    window_size = env_cfg.get("window_size", 60)
    max_position_pct = env_cfg.get("max_position_pct", 0.95)

    # Broker (TRADING_MODE=PAPER uses PaperBroker; else CCXT/Coinbase)
    trading_mode = os.environ.get("TRADING_MODE", "LIVE").upper()
    if trading_mode == "PAPER":
        from src.brokers import PaperBroker
        paper_cfg = config.get("paper", {})
        fees_cfg = config.get("fees", {})
        broker = PaperBroker(
            initial_balance=env_cfg.get("initial_balance", 10_000.0),
            taker_fee=fees_cfg.get("taker", 0.006),
            maker_fee=fees_cfg.get("maker", 0.004),
            slippage_pct=paper_cfg.get("slippage_pct", 0.0008),
        )
        log.info("Using PaperBroker (TRADING_MODE=PAPER)")
    elif args.broker == "coinbase":
        try:
            from src.brokers import CoinbaseBroker
            if CoinbaseBroker is None:
                raise ImportError("CoinbaseBroker requires: pip install coinbase-advanced-py")
            broker = CoinbaseBroker()
        except ImportError as e:
            msg = f"CoinbaseBroker unavailable: {e}"
            log.error(msg)
            telemetry.alert_error(msg)
            return
    else:
        from src.brokers import CCXTBroker
        broker = CCXTBroker(exchange_id="coinbase")
        broker._api_retry_max = live_cfg.get("api_retry_max", 5)

    circuit = CircuitBreaker(max_daily_drawdown_pct=live_cfg.get("max_daily_drawdown_pct", 5.0))
    paths = config.get("paths", {})
    paper_cfg = config.get("paper", {})
    if trading_mode == "PAPER":
        order_db = paper_cfg.get("order_db_path") or paths.get("order_db_paper", "data/paper/orders.db")
    else:
        order_db = live_cfg.get("order_db_path") or paths.get("order_db_live", paths.get("order_db", "data/orders.db"))
    sqlite_logger = SQLiteLogger(db_path=order_db)  # Full schema; init before OrderTracker
    order_tracker = OrderTracker(db_path=order_db)

    # Per-symbol ingestors and buffers
    ingestors = {
        s: DataIngestor(source="fetch", symbol=s, timeframe=args.timeframe, limit=200)
        for s in symbols
    }
    buffers: dict[str, object] = {}
    feature_cols: list[str] | None = None

    # Bootstrap: fetch historical for each symbol, add indicators, init buffers
    log.info("Bootstrapping StateBuffer for %s...", symbols)
    for symbol in symbols:
        df = ingestors[symbol].fetch_historical()
        if df is None or len(df) < window_size:
            msg = f"Insufficient data for {symbol} (need >= {window_size} bars)"
            log.error(msg)
            telemetry.alert_error(msg)
            return
        df = add_all_indicators(
            df,
            macd_fast=ind_cfg.get("macd", {}).get("fast", 12),
            macd_slow=ind_cfg.get("macd", {}).get("slow", 26),
            macd_signal=ind_cfg.get("macd", {}).get("signal", 9),
            rsi_length=ind_cfg.get("rsi", {}).get("length", 14),
            use_log_returns=feat_cfg.get("use_log_returns", False),
        )
        df = df.dropna().reset_index(drop=True)
        skip = {"open", "high", "low", "close", "volume", "timestamp"}
        cols = [c for c in df.columns if c.lower() not in skip]
        if feature_cols is None:
            feature_cols = cols
        buf = StateBuffer(feature_cols=feature_cols, window_size=window_size)
        for i in range(max(0, len(df) - window_size), len(df)):
            buf.append(df.iloc[i])
        buffers[symbol] = buf

    obs_dim = window_size * len(feature_cols)
    brain = StrategyBrain(model_path=args.model, expected_obs_dim=obs_dim)
    executor = Executor(broker=broker)

    log.info("Live loop started: %s %s (dry_run=%s)", symbols, args.timeframe, args.dry_run)

    def on_new_candle() -> None:
        # PaperBroker requires set_price before any calls; fetch latest bar per symbol
        if trading_mode == "PAPER":
            for sym in symbols:
                ing = ingestors[sym]
                df_raw = ing.fetch_historical()
                if df_raw is not None and len(df_raw) >= 2:
                    df_full = add_all_indicators(
                        df_raw,
                        macd_fast=ind_cfg.get("macd", {}).get("fast", 12),
                        macd_slow=ind_cfg.get("macd", {}).get("slow", 26),
                        macd_signal=ind_cfg.get("macd", {}).get("signal", 9),
                        rsi_length=ind_cfg.get("rsi", {}).get("length", 14),
                        use_log_returns=feat_cfg.get("use_log_returns", False),
                    )
                    df_full = df_full.dropna()
                    if len(df_full) >= 2:
                        close = float(df_full.iloc[-2]["close"])
                        broker.set_price(sym, close)
                time.sleep(0.5)

        try:
            total_value = get_total_portfolio_value(broker, symbols)
        except Exception as e:
            telemetry.alert_error(f"Failed to get portfolio value: {e}")
            log.exception("Portfolio value fetch failed: %s", e)
            return
        circuit.record_day_start(total_value)
        if not circuit.check(total_value):
            dd = (circuit._day_start_value - total_value) / circuit._day_start_value * 100 if circuit._day_start_value else 0
            log.warning("Circuit breaker tripped; skipping trade")
            telemetry.alert_circuit_breaker(dd)
            return
        if kill_switch[0]:
            return

        budget_per_symbol = total_value / len(symbols) if symbols else 0

        for symbol in symbols:
            ingestor = ingestors[symbol]
            buffer = buffers[symbol]

            df_raw = ingestor.fetch_historical()
            if df_raw is None or len(df_raw) < 2:
                continue

            df_full = add_all_indicators(
                df_raw,
                macd_fast=ind_cfg.get("macd", {}).get("fast", 12),
                macd_slow=ind_cfg.get("macd", {}).get("slow", 26),
                macd_signal=ind_cfg.get("macd", {}).get("signal", 9),
                rsi_length=ind_cfg.get("rsi", {}).get("length", 14),
                use_log_returns=feat_cfg.get("use_log_returns", False),
            )
            df_full = df_full.dropna().reset_index(drop=True)
            if len(df_full) < 2:
                continue

            bar = df_full.iloc[-2]  # Last closed candle
            buffer.append(bar)
            if not buffer.is_ready:
                continue

            price = float(bar["close"]) if "close" in bar else broker.get_price(symbol)
            if trading_mode == "PAPER":
                broker.set_price(symbol, price)
            obs = buffer.get_obs()
            target_pct = brain.predict(obs)
            current_pct = get_current_exposure(broker, symbol, total_value=total_value)
            signal = brain.to_signal(target_pct, current_pct, threshold=args.threshold)

            if signal == "hold":
                continue

            side, amount = calculate_order_size(
                target_pct, current_pct, budget_per_symbol, price, max_position_pct
            )
            if amount <= 0:
                continue

            if args.dry_run:
                log.info("DRY-RUN would %s %.8f %s", side.upper(), amount, symbol)
                continue

            try:
                result = executor.execute(symbol, side, amount)
                if result.success or result.order_id:
                    log.info("Executed %s %.8f %s", side.upper(), amount, symbol)
                    order_tracker.add_filled(result.order_id or "", symbol, side, amount, price)
                    sqlite_logger.log_order(result, signal_price=price)
                    telemetry.alert_trade(side, amount, symbol, price)
                else:
                    err = result.error_message or ""
                    if "403" in err or "forbidden" in err.lower() or "insufficient" in err.lower():
                        kill_switch[0] = True
                        telemetry.alert_error(f"Kill switch: {err}")
                        log.error("KILL SWITCH: %s — target set to 0%% (all cash). Restart to retry.", err)
                    else:
                        log.warning("Execution failed %s: %s", symbol, err)
                        telemetry.alert_warning(f"Execution failed {symbol}: {err}")
            except Exception as e:
                err = str(e).lower()
                if "403" in err or "forbidden" in err or "insufficient" in err:
                    kill_switch[0] = True
                    telemetry.alert_error(f"Kill switch: {e}")
                    log.error("KILL SWITCH: %s — target set to 0%% (all cash). Restart to retry.", e)
                else:
                    telemetry.alert_error(f"Execution error {symbol}: {e}")
                    log.exception("Execution error %s: %s", symbol, e)

            time.sleep(0.5)  # Rate limit between symbol fetches

    kill_switch = [False]  # Mutable so on_new_candle can set it

    # Main loop
    try:
        while True:
            DataIngestor.sleep_until_next_candle(args.timeframe)
            on_new_candle()
    except KeyboardInterrupt:
        log.info("Live loop stopped by user")
    except Exception as e:
        telemetry.alert_error(f"Live loop crashed: {e}")
        log.exception("Live loop fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
