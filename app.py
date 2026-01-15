# app.py
# Streamlit app: Stock vs Index vs "Idiosyncratic" (beta-hedged) performance
#
# Idio performance here = cumulative return of (stock daily return - beta * index daily return),
# where beta is estimated over the selected lookback window using daily log returns.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from typing import Tuple

st.set_page_config(page_title="Idio Performance Dashboard", layout="wide")

st.title("Idiosyncratic Stock Performance (Beta-Hedged vs Index)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Stock ticker", value="APGE").strip().upper()
    lookback = st.selectbox(
        "Lookback period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )

    # User requested: XBI, SPX, XLV, BBC (BBC likely meant IBB). We'll support both labels.
    index_label = st.selectbox("Index", ["XBI", "SPX", "XLV", "IBB", "BBC"], index=0)

    # Map friendly labels to Yahoo tickers
    INDEX_MAP = {
        "XBI": "XBI",      # SPDR S&P Biotech ETF
        "SPX": "^GSPC",    # S&P 500 index
        "XLV": "XLV",      # Health Care Select Sector SPDR
        "IBB": "IBB",      # iShares Nasdaq Biotechnology ETF
        "BBC": "IBB",      # treat "BBC" as alias for IBB (common biotech benchmark)
    }
    index_ticker = INDEX_MAP[index_label]

    st.divider()
    st.subheader("Beta Calculation")
    
    use_custom_beta_lookback = st.checkbox("Use custom beta lookback", value=False)
    if use_custom_beta_lookback:
        beta_lookback_days = st.number_input(
            "Beta lookback (days)",
            min_value=30,
            max_value=1000,
            value=126,
            step=1,
            help="Number of trading days to use for beta estimation"
        )
    else:
        beta_lookback_days = 126  # Default 126 days (~6 months)

# --- Helpers ---
def fetch_prices(tkr: str, period: str) -> pd.Series:
    df = yf.download(tkr, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{tkr}'.")
    # Prefer Adj Close when available
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].dropna()
    s.name = tkr
    return s

def calc_beta(logret_stock: pd.Series, logret_index: pd.Series, lookback_days: int = 126) -> float:
    """
    Calculate beta using OLS regression with intercept.
    
    Args:
        logret_stock: Stock log returns
        logret_index: Index log returns
        lookback_days: Number of most recent trading days to use for beta estimation
    
    Returns:
        Beta coefficient: Cov(r_stock, r_index) / Var(r_index)
    """
    aligned = pd.concat([logret_stock, logret_index], axis=1).dropna()
    
    # Use only the most recent N days
    if len(aligned) > lookback_days:
        aligned = aligned.iloc[-lookback_days:]
    
    if aligned.shape[0] < 30:
        raise ValueError(f"Not enough overlapping daily data to estimate beta (need ~30+ days, got {aligned.shape[0]}).")
    
    s = aligned.iloc[:, 0]  # stock returns
    m = aligned.iloc[:, 1]  # index returns
    
    # OLS with intercept: stock = alpha + beta * index + error
    # beta = Cov(r_stock, r_index) / Var(r_index)
    var_m = np.var(m, ddof=1)
    if var_m <= 0:
        raise ValueError("Index variance is zero over this window; cannot estimate beta.")
    beta = np.cov(s, m, ddof=1)[0, 1] / var_m
    return float(beta)

def to_cum_perf_from_logrets(logrets: pd.Series) -> pd.Series:
    # Convert log returns to cumulative performance (starting at 0%)
    # cum = exp(cumsum(logret)) - 1
    return np.exp(logrets.cumsum()) - 1

def to_cum_perf_from_simple(simple_rets: pd.Series) -> pd.Series:
    return (1 + simple_rets).cumprod() - 1

# --- Main ---
# Auto-run when inputs are provided
if ticker:
    try:
        with st.spinner("Fetching data..."):
            px_stock = fetch_prices(ticker, lookback)
            px_index = fetch_prices(index_ticker, lookback)

        # Align to common date range
        df_px = pd.concat([px_stock, px_index], axis=1).dropna()
        if df_px.shape[0] < 35:
            st.warning("Very short overlap window — results may be noisy.")

        # Daily log returns (for beta estimation only)
        logret = np.log(df_px / df_px.shift(1)).dropna()
        logret_stock = logret.iloc[:, 0]
        logret_index = logret.iloc[:, 1]

        beta = calc_beta(logret_stock, logret_index, lookback_days=beta_lookback_days)

        # Daily simple returns (for cumulative series and vol stats)
        simp_ret = df_px.pct_change().dropna()
        r_stock = simp_ret.iloc[:, 0]
        r_index = simp_ret.iloc[:, 1]

        # Cumulative performances (all in simple-return space for consistency)
        cum_stock = to_cum_perf_from_simple(r_stock)
        cum_index = to_cum_perf_from_simple(r_index)

        # "Idio" series: Beta-hedged in simple-return space (true hedge P&L)
        idio_ret = r_stock - beta * r_index
        cum_idio = to_cum_perf_from_simple(idio_ret)

        # --- Chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_stock.index, y=cum_stock.values, mode="lines", name=f"{ticker}"))
        fig.add_trace(go.Scatter(x=cum_index.index, y=cum_index.values, mode="lines", name=f"{index_label} ({index_ticker})"))
        fig.add_trace(go.Scatter(x=cum_idio.index, y=cum_idio.values, mode="lines", name=f"Beta-hedged / Idio (simple-return hedge P&L)"))

        fig.update_layout(
            title=f"{ticker} vs {index_label} and Idiosyncratic Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative return",
            hovermode="x unified",
            legend_title="Series",
            height=550,
        )

        # --- Layout ---
        # Fixed width: summary column is 20% (5% narrower than previous 25%)
        c1, c2 = st.columns([4, 1], gap="large")

        with c1:
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Model summary")
            st.metric("Estimated beta", f"{beta:.3f}")
            st.caption(
                f"Beta estimated using the most recent {beta_lookback_days} trading days (OLS with intercept)."
            )
            st.write("**Interpretation**")
            idio_formula = f"r_stock − {beta:.3f}·r_index"
            st.write(
                f"- **Stock**: raw cumulative performance of {ticker}\n"
                f"- **Index**: raw cumulative performance of {index_label}\n"
                f"- **Beta-hedged**: cumulative performance of daily simple-return residuals (true hedge P&L): {idio_formula}"
            )

            # Quick stats (all in simple returns for consistency)
            aligned = pd.concat([r_stock, r_index, idio_ret], axis=1).dropna()
            aligned.columns = ["r_stock", "r_index", "idio_ret"]
            ann_factor = 252.0

            vol_stock = aligned["r_stock"].std(ddof=1) * np.sqrt(ann_factor)
            vol_index = aligned["r_index"].std(ddof=1) * np.sqrt(ann_factor)
            vol_idio = aligned["idio_ret"].std(ddof=1) * np.sqrt(ann_factor)

            st.subheader("Vol (annualized)")
            st.write(pd.DataFrame(
                {
                    "Vol": [vol_stock, vol_index, vol_idio]
                },
                index=[ticker, f"{index_label} ({index_ticker})", "Beta-hedged / Idio"]
            ).style.format({"Vol": "{:.2%}"}))

        # Optional: show data
        with st.expander("Show underlying data"):
            out = pd.DataFrame({
                f"{ticker}_AdjClose": df_px.iloc[:, 0],
                f"{index_label}_AdjClose": df_px.iloc[:, 1],
                f"{ticker}_cum": cum_stock.reindex(df_px.index),
                f"{index_label}_cum": cum_index.reindex(df_px.index),
                "idio_cum": cum_idio.reindex(df_px.index),
            })
            st.dataframe(out.dropna(), use_container_width=True)

    except Exception as e:
        st.error(str(e))

