"""
TradFiBot Monitoring Dashboard — Multi-Symbol Edition.

Run: streamlit run scripts/dashboard.py

Reads from data/orders.db. Set TELEGRAM_BOT_TOKEN in .env for alerts.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import load_config

st.set_page_config(page_title="TradFiBot Multi-Monitor", layout="wide")
st.title("🤖 TradFiBot Multi-Symbol Monitor")


def get_order_db_path() -> Path:
    """Auto-detect paper vs live: use DASHBOARD_MODE/TRADING_MODE env, or most recently modified db."""
    config = load_config()
    paths = config.get("paths", {})
    paper_path = Path(paths.get("order_db_paper", "data/paper/orders.db"))
    live_path = Path(paths.get("order_db_live", paths.get("order_db", "data/orders.db")))
    mode = os.environ.get("DASHBOARD_MODE") or os.environ.get("TRADING_MODE", "").upper()
    if mode == "PAPER":
        return paper_path
    if mode == "LIVE":
        return live_path
    # Auto-detect: use whichever db was modified more recently
    paper_mtime = paper_path.stat().st_mtime if paper_path.exists() else 0
    live_mtime = live_path.stat().st_mtime if live_path.exists() else 0
    if paper_mtime >= live_mtime and paper_mtime > 0:
        return paper_path
    return live_path


def get_live_price(symbol: str) -> float | None:
    """Fetch current price via CCXT (public, no API keys)."""
    try:
        import ccxt
        exchange = ccxt.coinbase()
        sym = symbol.replace("-", "/")
        ticker = exchange.fetch_ticker(sym)
        return float(ticker.get("last") or ticker.get("close") or 0)
    except Exception:
        return None


def _get_data_source_label() -> str:
    """Human-readable label for current data source."""
    db = get_order_db_path()
    if "paper" in str(db):
        return "Paper Trading"
    return "Live Trading"


@st.cache_data(ttl=10)
def load_orders(db_path: Path) -> pd.DataFrame:
    """Refresh every 10 seconds. Cache keyed by path so paper/live switch invalidates."""
    if not db_path.exists():
        return pd.DataFrame(columns=["order_id", "symbol", "side", "amount", "price", "status", "created_at", "fee"])
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM orders ORDER BY created_at", conn)
    conn.close()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def compute_equity_curve(df: pd.DataFrame, initial_balance: float = 10_000.0) -> pd.DataFrame:
    """Derive equity over time from order stream."""
    filled = df[df["status"] == "filled"] if "status" in df.columns else df
    if filled.empty:
        return pd.DataFrame({"created_at": [], "equity": []})
    rows = []
    cash = initial_balance
    position = 0.0
    for _, r in filled.iterrows():
        side = str(r["side"]).lower()
        amt = float(r["amount"])
        price = float(r["price"])
        if side == "buy":
            cash -= amt * price
            position += amt
        else:
            cash += amt * price
            position -= amt
        equity = cash + position * price
        rows.append({"created_at": r["created_at"], "equity": equity})
    return pd.DataFrame(rows) if rows else pd.DataFrame({"created_at": [], "equity": []})


def main() -> None:
    db_path = get_order_db_path()
    df = load_orders(db_path)
    config = load_config()
    initial = config.get("env", {}).get("initial_balance", 10_000.0)
    symbols_in_orders = sorted(df["symbol"].dropna().unique().tolist()) if not df.empty and "symbol" in df.columns else []
    all_symbols = ["All"] + symbols_in_orders

    # Sidebar
    with st.sidebar:
        st.header("Controls")
        st.metric("Data Source", _get_data_source_label(), help="Auto-detected from paper-trader or live-engine")
        symbol_filter = st.selectbox("Filter by Symbol", options=all_symbols, index=0, help="Filter charts and executions")
        st.divider()
        st.metric("Bot Status", "🟢 LIVE")

    # --- Top Row KPIs ---
    col1, col2, col3, col4, col5 = st.columns(5)

    total_trades = len(df)
    col1.metric("Total Trades", total_trades)

    filled = df[df["status"] == "filled"] if not df.empty else pd.DataFrame()
    df_filtered = df[df["symbol"] == symbol_filter] if symbol_filter != "All" and not df.empty else df
    filled_filtered = filled[filled["symbol"] == symbol_filter] if symbol_filter != "All" and not filled.empty else filled
    last_signal = "N/A"
    if not filled.empty:
        last = filled.iloc[-1]
        last_signal = f"{last['side'].upper()} {last['amount']:.4f} {last['symbol']}"
    col2.metric("Last Signal", last_signal)

    # Win rate (realized_pnl > 0 on sells)
    win_rate = "—"
    filled_sells = filled[(filled["side"].str.lower() == "sell")] if not filled.empty else pd.DataFrame()
    if not filled_sells.empty and "realized_pnl" in filled_sells.columns:
        wins = (filled_sells["realized_pnl"] > 0).sum()
        win_rate = f"{(wins / len(filled_sells) * 100):.1f}%"
    col3.metric("Win Rate (Sells)", win_rate)

    # Total return
    eq = compute_equity_curve(df, initial)
    total_return = "—"
    if not eq.empty:
        final = eq["equity"].iloc[-1]
        total_return = f"{((final - initial) / initial * 100):.1f}%"
    col4.metric("Total Return %", total_return)

    # Daily Max Drawdown (avoid Kill Switch)
    daily_dd = "—"
    if not eq.empty:
        eq["date"] = pd.to_datetime(eq["created_at"]).dt.date
        today = pd.Timestamp.now().date()
        today_eq = eq[eq["date"] == today]
        if not today_eq.empty:
            day_start = today_eq["equity"].iloc[0]
            current = eq["equity"].iloc[-1]
            if day_start > 0:
                daily_dd = f"{((day_start - current) / day_start * 100):.1f}%"
        else:
            current = eq["equity"].iloc[-1]
            prev = eq[eq["date"] < today]
            day_start = prev["equity"].iloc[-1] if not prev.empty else current
            if day_start > 0:
                daily_dd = f"{((day_start - current) / day_start * 100):.1f}%"
    col5.metric("Daily Max Drawdown", daily_dd)

    # --- Asset Performance Table ---
    if not filled.empty and symbols_in_orders:
        st.subheader("Asset Performance")
        perf_rows = []
        for sym in symbols_in_orders:
            sym_filled = filled[filled["symbol"] == sym]
            trade_count = len(sym_filled)
            realized_pnl = sym_filled["realized_pnl"].sum() if "realized_pnl" in sym_filled.columns else 0.0
            slippage = 0.0
            if "signal_price" in sym_filled.columns and "price" in sym_filled.columns:
                slippage = (sym_filled["signal_price"] - sym_filled["price"]).sum()
            perf_rows.append({
                "Symbol": sym, "Trades": trade_count, "Realized PnL ($)": round(realized_pnl, 2),
                "Slippage ($)": round(slippage, 4),
            })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            perf_df["Cumulative PnL"] = perf_df["Realized PnL ($)"].apply(lambda x: f"${x:,.2f}")
            show_cols = [c for c in ["Symbol", "Trades", "Cumulative PnL", "Slippage ($)"] if c in perf_df.columns]
            st.dataframe(perf_df[show_cols], use_container_width=True, hide_index=True)

    # --- Charts (two-column) ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Equity Curve")
        if not eq.empty and len(eq) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["created_at"], y=eq["equity"], name="Portfolio Value", mode="lines+markers"))
            fig.update_layout(xaxis_title="Time", yaxis_title="Value (USD)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data yet.")
        st.subheader("Price & Execution" + (f" — {symbol_filter}" if symbol_filter != "All" else ""))
        if not filled_filtered.empty:
            buys = filled_filtered[filled_filtered["side"].str.lower() == "buy"]
            sells = filled_filtered[filled_filtered["side"].str.lower() == "sell"]
            fig = go.Figure()
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["created_at"], y=buys["price"],
                    name="Buy", mode="markers", marker=dict(size=10, color="green", symbol="triangle-up")
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["created_at"], y=sells["price"],
                    name="Sell", mode="markers", marker=dict(size=10, color="red", symbol="triangle-down")
                ))
            fig.update_layout(xaxis_title="Time", yaxis_title="Price", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Allocation (live prices)")
        if not filled.empty:
            cash = initial
            positions: dict[str, float] = {}  # base -> amount
            for _, r in filled.iterrows():
                side = str(r["side"]).lower()
                amt = float(r["amount"])
                price = float(r["price"])
                base, quote = (str(r.get("symbol", "BTC-USDT")).split("-") + ["USDT"])[:2]
                if side == "buy":
                    cash -= amt * price
                    positions[base] = positions.get(base, 0) + amt
                else:
                    cash += amt * price
                    positions[base] = positions.get(base, 0) - amt
            slice_data = [{"Asset": "Cash (USD)", "Value": max(0, cash)}]
            for base, amt in positions.items():
                if amt > 0:
                    sym = f"{base}-USDT"
                    time.sleep(0.2)  # Rate limit between price fetches
                    live_price = get_live_price(sym)
                    price = live_price if live_price else 0.0
                    slice_data.append({"Asset": base, "Value": amt * price})
            total = sum(s["Value"] for s in slice_data)
            if total > 0:
                pos_value = total - max(0, cash)
                exposure_pct = (pos_value / total * 100) if total > 0 else 0
                st.metric("Exposure Ratio", f"{exposure_pct:.1f}%", help="Stay under 95%")
                if exposure_pct > 95:
                    st.warning("Exposure exceeds 95%.")
                fig = px.pie(pd.DataFrame(slice_data), values="Value", names="Asset", title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positions or live price unavailable.")
        else:
            st.info("No orders yet.")

    st.subheader("Raw Order Log" + (f" — {symbol_filter}" if symbol_filter != "All" else ""))
    display = df_filtered.tail(50).copy()
    if not display.empty and "signal_price" in display.columns and "price" in display.columns:
        display["Slippage ($)"] = (display["signal_price"] - display["price"]).round(4)
    cols = ["created_at", "symbol", "side", "amount", "price", "Slippage ($)", "status"]
    display = display[[c for c in cols if c in display.columns]] if not display.empty else display
    st.dataframe(display, use_container_width=True)


if __name__ == "__main__":
    main()
