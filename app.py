import streamlit as st
import matplotlib.pyplot as plt
from core_model import run_pipeline
import numpy as np
import pandas as pd

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Stock Signal Dashboard",
    layout="wide"
)

# Title of the app
st.title("📈 Stock Prediction & Backtest Dashboard")

# Sidebar settings for user input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input(
    "Ticker (e.g. IFX.DE, RHM.DE, AAPL, MSFT)",
    value="IFX.DE"
)
run_button = st.sidebar.button("Run Analysis")

if run_button:
    with st.spinner("Running pipeline..."):
        output = run_pipeline(ticker)

    # Safely unpack the output dictionary
    summary_df = output.get("summary_df")
    results = output.get("results")
    backtest_results = output.get("backtest_results")
    feat_df = output.get("feat_df")
    rl_results = output.get("rl_results")
    # predicted prices and RL recommended action
    predicted_price_ml = output.get("predicted_price_ml")
    predicted_price_rl = output.get("predicted_price_rl")
    rl_action_last = output.get("rl_action_last")

    last_price = feat_df["Close"].iloc[-1] if feat_df is not None else None

    # Display last price
    if last_price is None:
        st.warning("Last price is not available.")
    else:
        st.write(f"Last close price: {last_price:.2f} €")
    # Display predicted next-day prices
    if predicted_price_ml is not None and not np.isnan(predicted_price_ml):
        st.write(f"Predicted next-day close (ML): {predicted_price_ml:.2f} €")
    if predicted_price_rl is not None and not np.isnan(predicted_price_rl):
        st.write(f"Predicted next-day close (RL): {predicted_price_rl:.2f} €")
    if rl_action_last:
        st.write(f"RL recommended action: **{rl_action_last}**")

    st.subheader(f"Overview for {ticker}")

    # --- Show performance summary table ---
    st.subheader("Model Performance Summary")
    if summary_df is not None:
        # Convert numeric columns to readable format
        summary_df["Total Return (%)"] = summary_df["Total Return (%)"].map("{:.2f}".format)
        summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].map("{:.2f}".format)
        summary_df["Max Drawdown (%)"] = summary_df["Max Drawdown (%)"].map("{:.2f}".format)
        st.dataframe(summary_df.reset_index(drop=True))

    # --- Show best models ---
    if summary_df is not None and not summary_df.empty:
        # Convert numeric strings back to floats for comparison
        temp_df = summary_df.copy()
        # Attempt to convert string columns to floats
        for col in ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Accuracy"]:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        best_return_row = temp_df.iloc[temp_df['Total Return (%)'].idxmax()]
        best_acc_row = temp_df.iloc[temp_df['Accuracy'].idxmax()]
        best_sharpe_row = temp_df.iloc[temp_df['Sharpe Ratio'].idxmax()]

        st.markdown(f"""
        **Best Total Return:** {best_return_row['Model']}  
        **Best Accuracy:** {best_acc_row['Model']}  
        **Best Sharpe Ratio:** {best_sharpe_row['Model']}
        """)

    # --- Price + indicators plot ---
    st.subheader("Price / Bands / RSI / MACD")
    if feat_df is not None:
        fig1 = plt.figure(figsize=(12, 8))
        ax = fig1.add_subplot(111)
        ax.plot(feat_df.index, feat_df["Close"], label="Close")
        ax.set_title(f"{ticker} Close Price")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig1)

    # --- Portfolio curves comparison ---
    st.subheader("Backtest Portfolio Value")
    if backtest_results:
        fig2 = plt.figure(figsize=(12, 6))
        ax2 = fig2.add_subplot(111)
        # Plot each portfolio curve
        for model_name, res in backtest_results.items():
            ax2.plot(res["portfolio_value"], label=model_name)
        ax2.axhline(y=10000, color="black", ls="--", alpha=0.5)
        ax2.set_title("Portfolio Value Over Time (€)")
        ax2.set_xlabel("Day (test period)")
        ax2.set_ylabel("Portfolio Value (€)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

    # --- Risk metrics bar chart ---
    st.subheader("Risk / Return Metrics")
    if summary_df is not None:
        metrics_plot_df = summary_df.set_index("Model")[["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]]
        st.bar_chart(metrics_plot_df)

    # --- Raw technical feature preview ---
    st.subheader("Engineered Feature Sample")
    if feat_df is not None:
        st.write(feat_df.tail(10))

else:
    st.info("👈 Pick a ticker in the sidebar and click 'Run Analysis'.")
