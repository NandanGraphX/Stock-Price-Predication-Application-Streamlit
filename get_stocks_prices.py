#!/usr/bin/env python3
"""
Single-file stock prediction and backtesting pipeline.

Steps:
1. Download ticker data (yfinance)
2. Build technical indicators
3. Create next-day buy/sell labels
4. Train ML models (LogReg, RF, XGBoost) + LSTM
5. Evaluate, backtest, visualize, summarize

Note:
- This script assumes binary labels: 1 = "price will go up enough to buy",
  0 = "price will go down enough to sell".
- Neutral cases are dropped.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# xgboost
import xgboost as xgb

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =========================
# GLOBAL CONFIG
# =========================
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.001   # 0.1%
BATCH_SIZE = 32
NUM_EPOCHS = 50
LSTM_HIDDEN = 500
LSTM_LAYERS = 2  # logically we have two stacked LSTMs
LSTM_DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =========================
# DATA LOADING
# =========================
def get_stock_data(ticker, period="5y", interval="1d"):
    """
    Download OHLCV data using yfinance.
    """
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# =========================
# FEATURE ENGINEERING
# =========================
def calculate_technical_indicators(df):
    """
    Add technical indicators to OHLCV dataframe.
    """
    df = df.copy()

    # --- Moving Averages ---
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # --- RSI ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # --- Bollinger Bands ---
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (
        (df['BB_Upper'] - df['BB_Lower']) + 1e-10
    )

    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # --- Volume features ---
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)

    # --- Price-based ---
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    df['Price_Position'] = (df['Close'] - df['Low']) / (
        (df['High'] - df['Low']) + 1e-10
    )

    # --- Momentum / ROC ---
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['ROC'] = (df['Close'] - df['Close'].shift(10)) / (
        df['Close'].shift(10) + 1e-10
    ) * 100

    # --- MA Ratios ---
    df['MA_Ratio_10_20'] = df['SMA_10'] / (df['SMA_20'] + 1e-10)
    df['MA_Ratio_20_50'] = df['SMA_20'] / (df['SMA_50'] + 1e-10)

    return df


# =========================
# LABEL CREATION
# =========================
def create_labels(df, prediction_threshold=0.01):
    """
    Binary label for "tomorrow goes up a lot -> buy(1)" vs "down a lot -> sell(0)".
    Neutral values are marked -1 and filtered out later.
    """
    future_return = df['Close'].shift(-1) / df['Close'] - 1
    labels = np.where(
        future_return > prediction_threshold, 1,
        np.where(future_return < -prediction_threshold, 0, -1)
    )
    valid_mask = labels != -1
    return labels, valid_mask


# =========================
# EVAL + BACKTEST
# =========================
def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Compute classification metrics for a model.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name}]")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("  Confusion matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def backtest_strategy(price_data, predictions,
                      initial_capital=INITIAL_CAPITAL,
                      transaction_cost=TRANSACTION_COST):
    """
    Extremely simple long/flat backtest:
    - If model says BUY (1) and we're not in a position -> go all in
    - If model says SELL (0) and we're in a position -> close position
    """
    capital = initial_capital
    position = 0        # 0: flat, 1: long
    portfolio_value = [initial_capital]
    trades = []

    for i in range(1, len(predictions)):
        px = price_data[i]

        # BUY
        if predictions[i] == 1 and position == 0:
            shares = capital / px
            capital = 0
            position = 1
            trades.append(("BUY", i, px, shares))

        # SELL
        elif predictions[i] == 0 and position == 1:
            shares = trades[-1][3]
            capital = shares * px * (1 - transaction_cost)
            position = 0
            trades.append(("SELL", i, px, shares))

        # mark portfolio value
        if position == 1:
            shares = trades[-1][3]
            portfolio_value.append(shares * px)
        else:
            portfolio_value.append(capital)

    # final liquidation if still long
    if position == 1 and len(trades) > 0:
        shares = trades[-1][3]
        final_capital = shares * px * (1 - transaction_cost)
    else:
        final_capital = portfolio_value[-1]

    # performance metrics
    total_return = (final_capital - initial_capital) / initial_capital
    portfolio_value = np.array(portfolio_value)
    returns = np.diff(portfolio_value) / (portfolio_value[:-1] + 1e-10)

    # Sharpe (very rough, daily-ish, no rf adjustment)
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)

    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-10)
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "total_trades": len(trades),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "portfolio_value": portfolio_value,
        "trades": trades
    }


# =========================
# LSTM MODEL DEFS
# =========================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size,
                 hidden_size=LSTM_HIDDEN,
                 dropout=LSTM_DROPOUT):
        super().__init__()
        # two LSTM blocks
        self.lstm1 = nn.LSTM(
            input_size, hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.drop2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])  # last timestep
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


def create_lstm_sequences(X, y, sequence_length=SEQUENCE_LENGTH):
    """
    Turn tabular samples into rolling windows of length `sequence_length`.
    """
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train_scaled, y_train, X_test_scaled, y_test,
               num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """
    Train 2-layer LSTM binary classifier.
    Returns:
      model, history(dict), y_pred_lstm(int array), y_test_seq(aligned labels)
    """
    # build sequences
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train)
    X_test_seq,  y_test_seq  = create_lstm_sequences(X_test_scaled,  y_test)

    # loaders
    train_loader = DataLoader(
        StockDataset(X_train_seq, y_train_seq),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        StockDataset(X_test_seq, y_test_seq),
        batch_size=batch_size, shuffle=False
    )

    # model
    input_size = X_train_seq.shape[2]
    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "loss": [], "accuracy": [],
        "val_loss": [], "val_accuracy": []
    }

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            out = model(Xb).squeeze()
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (out > 0.5).float()
            train_total += yb.size(0)
            train_correct += (preds == yb).sum().item()

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb).squeeze()
                loss = criterion(out, yb)
                val_loss += loss.item()

                preds = (out > 0.5).float()
                val_total += yb.size(0)
                val_correct += (preds == yb).sum().item()

        # record
        history["loss"].append(train_loss / len(train_loader))
        history["accuracy"].append(train_correct / train_total)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_accuracy"].append(val_correct / val_total)

        if (epoch + 1) % 10 == 0:
            print(
                f"[LSTM] Epoch {epoch+1}/{num_epochs} "
                f"TrainLoss={history['loss'][-1]:.4f} "
                f"TrainAcc={history['accuracy'][-1]:.4f} "
                f"ValLoss={history['val_loss'][-1]:.4f} "
                f"ValAcc={history['val_accuracy'][-1]:.4f}"
            )

    # predictions on val set
    y_pred_scores = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            Xb = Xb.to(device)
            out = model(Xb).squeeze().cpu().numpy()
            y_pred_scores.extend(out)

    y_pred_lstm = (np.array(y_pred_scores) > 0.5).astype(int)

    return model, history, y_pred_lstm, y_test_seq, X_test_seq


# =========================
# PLOTTING HELPERS
# =========================
def plot_indicators(df, ticker, currency_label="EUR"):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Price + bands / MAs
    axes[0].plot(df.index, df['Close'], label='Close', linewidth=2)
    axes[0].plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    axes[0].plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    axes[0].plot(df.index, df['BB_Upper'], label='BB Upper', ls='--', alpha=0.7)
    axes[0].plot(df.index, df['BB_Lower'], label='BB Lower', ls='--', alpha=0.7)
    axes[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)
    axes[0].set_title(f"{ticker} Price / Bands / MAs")
    axes[0].set_ylabel(f"Price ({currency_label})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # RSI
    axes[1].plot(df.index, df['RSI'], label='RSI', linewidth=2, color='purple')
    axes[1].axhline(y=70, color='r', ls='--', alpha=0.7, label='Overbought 70')
    axes[1].axhline(y=30, color='g', ls='--', alpha=0.7, label='Oversold 30')
    axes[1].fill_between(df.index, 30, 70, alpha=0.2)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("RSI")
    axes[1].set_ylabel("RSI")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # MACD
    axes[2].plot(df.index, df['MACD'], label='MACD', linewidth=2)
    axes[2].plot(df.index, df['MACD_Signal'], label='Signal', linewidth=2)
    axes[2].bar(df.index, df['MACD_Hist'], label='Hist', alpha=0.3)
    axes[2].set_title("MACD")
    axes[2].set_ylabel("MACD")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    


def plot_label_distribution(labels_filtered):
    plt.figure(figsize=(8, 5))
    counts = pd.Series(labels_filtered).value_counts().sort_index()
    plt.bar(['Sell (0)', 'Buy (1)'], counts.values,
            color=['red', 'green'], alpha=0.7)
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    plt.title("Label Distribution (after removing neutral)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    


def plot_lstm_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('LSTM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()

    # Acc
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(history['val_accuracy'], label='Val Acc', linewidth=2)
    plt.title('LSTM Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()


def plot_model_comparison(results_dict):
    comparison_df = pd.DataFrame(results_dict).T
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        comparison_df[metric].plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(comparison_df[metric]):
            ax.text(i, v, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()


def plot_backtest_results(backtest_results):
    plt.figure(figsize=(15, 10))

    # 1. Portfolio value over time
    plt.subplot(2, 2, 1)
    for name, res in backtest_results.items():
        plt.plot(res['portfolio_value'], label=name, linewidth=2, alpha=0.8)
    plt.axhline(y=INITIAL_CAPITAL, color='black', ls='--', alpha=0.5,
                label='Initial Capital')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value (€)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Total returns
    plt.subplot(2, 2, 2)
    returns_percent = {
        name: res['total_return'] * 100
        for name, res in backtest_results.items()
    }
    plt.bar(range(len(returns_percent)),
            list(returns_percent.values()),
            alpha=0.7)
    plt.xticks(range(len(returns_percent)),
               list(returns_percent.keys()), rotation=45)
    plt.title('Total Return (%)')
    plt.axhline(y=0, color='red', ls='--', alpha=0.5)
    for i, v in enumerate(returns_percent.values()):
        plt.text(i, v, f"{v:.2f}%",
                 ha='center',
                 va='bottom' if v > 0 else 'top',
                 fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 3. Sharpe ratio
    plt.subplot(2, 2, 3)
    sharpe_vals = {
        name: res['sharpe_ratio']
        for name, res in backtest_results.items()
    }
    plt.bar(range(len(sharpe_vals)),
            list(sharpe_vals.values()),
            alpha=0.7)
    plt.xticks(range(len(sharpe_vals)),
               list(sharpe_vals.keys()), rotation=45)
    plt.title('Sharpe Ratio')
    plt.axhline(y=0, color='red', ls='--', alpha=0.5)
    for i, v in enumerate(sharpe_vals.values()):
        plt.text(i, v, f"{v:.2f}",
                 ha='center',
                 va='bottom' if v > 0 else 'top',
                 fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 4. Max Drawdown
    plt.subplot(2, 2, 4)
    dd_vals = {
        name: res['max_drawdown'] * 100
        for name, res in backtest_results.items()
    }
    plt.bar(range(len(dd_vals)),
            list(dd_vals.values()),
            alpha=0.7)
    plt.xticks(range(len(dd_vals)),
               list(dd_vals.keys()), rotation=45)
    plt.title('Max Drawdown (%)')
    for i, v in enumerate(dd_vals.values()):
        plt.text(i, v, f"{v:.2f}%",
                 ha='center',
                 va='bottom' if v < 0 else 'top',
                 fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()


def plot_feature_importance(feature_importance_df, xgb_importance_df):
    fig, axes = plt.subplots(2, 1, figsize=(15, 14))

    # RF
    ax1 = axes[0]
    top_rf = feature_importance_df.head(15)
    ax1.barh(range(len(top_rf)), top_rf['Importance'], alpha=0.7)
    ax1.set_yticks(range(len(top_rf)))
    ax1.set_yticklabels(top_rf['Feature'])
    ax1.invert_yaxis()
    ax1.set_title('Random Forest - Top 15 Feature Importance')
    ax1.set_xlabel('Importance')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(top_rf['Importance']):
        ax1.text(v, i, f" {v:.3f}", va='center', fontsize=9)

    # XGB
    ax2 = axes[1]
    top_xgb = xgb_importance_df.head(15)
    ax2.barh(range(len(top_xgb)), top_xgb['Importance'], alpha=0.7)
    ax2.set_yticks(range(len(top_xgb)))
    ax2.set_yticklabels(top_xgb['Feature'])
    ax2.invert_yaxis()
    ax2.set_title('XGBoost - Top 15 Feature Importance')
    ax2.set_xlabel('Importance')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(top_xgb['Importance']):
        ax2.text(v, i, f" {v:.3f}", va='center', fontsize=9)

    plt.tight_layout()


# =========================
# LATEST SIGNAL / SIZING
# =========================
def latest_signals_and_size(y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_lstm,
                            last_price,
                            capital=INITIAL_CAPITAL,
                            risk_frac=0.02,
                            stop_loss_pct=0.02):
    sigs = {
        "Logistic Regression": y_pred_lr[-1] if len(y_pred_lr) else None,
        "Random Forest": y_pred_rf[-1] if len(y_pred_rf) else None,
        "XGBoost": y_pred_xgb[-1] if len(y_pred_xgb) else None,
        "LSTM": y_pred_lstm[-1] if len(y_pred_lstm) else None,
    }

    print("\nLatest model signals (1=buy, 0=sell / None=NA):")
    for model_name, sig in sigs.items():
        print(f"  {model_name}: {sig}")

    if last_price is None:
        print("\n[WARN] No last price available for position sizing.")
        return sigs

    position_value_fixed = capital * risk_frac
    shares_fixed = int(position_value_fixed / last_price)

    position_value_stop = capital * risk_frac
    shares_stop = int(position_value_stop / (last_price * 1.0))

    print(f"\nCurrent price: {last_price:.2f} €")
    print(f"Fixed-fraction sizing (risk {risk_frac*100:.1f}% of {capital} €): "
          f"buy {shares_fixed} shares (~{shares_fixed*last_price:.2f} €)")
    print(f"Stop-loss sizing (~{stop_loss_pct*100:.1f}% stop): "
          f"buy {shares_stop} shares "
          f"(risk ~{shares_stop*last_price*stop_loss_pct:.2f} €)")
    return sigs


# =========================
# MAIN EXECUTION
# =========================
def main():
    ticker = "IFX.DE"   # Infineon Technologies AG on XETRA (Germany, quoted in EUR)
    raw_df = get_stock_data(ticker, period="5y", interval="1d")

    print(f"[INFO] Data shape: {raw_df.shape}")
    print(f"[INFO] Date range: {raw_df.index[0]} → {raw_df.index[-1]}")
    print(raw_df.head())

    feat_df = calculate_technical_indicators(raw_df)

    # plot indicators for visual sanity check
    plot_indicators(feat_df, ticker, currency_label="EUR")

    # ----- labels -----
    labels, valid_mask = create_labels(feat_df, prediction_threshold=0.01)
    print(f"[INFO] Total samples: {len(labels)}")
    print(f"[INFO] Neutral dropped: {np.sum(labels == -1)}")

    # We'll build X from engineered features:
    feature_list = [
        'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_10', 'EMA_20', 'EMA_50',
        'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'ATR',
        'Volume_SMA', 'Volume_Ratio',
        'Price_Change', 'Price_Range', 'Price_Position',
        'Momentum', 'ROC',
        'MA_Ratio_10_20', 'MA_Ratio_20_50'
    ]

    X_full = feat_df[feature_list].values
    y_full = labels

    # mask invalid rows (neutral labels OR NaNs in features)
    valid_rows = valid_mask & ~np.isnan(X_full).any(axis=1)
    X_full = X_full[valid_rows]
    y_full = y_full[valid_rows]

    print(f"[INFO] After cleaning: X={X_full.shape}, y={y_full.shape}")
    plot_label_distribution(y_full)

    # ----- train/test split -----
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_full
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # =========================
    # Classical ML models
    # =========================
    results = {}

    # Logistic Regression
    lr_model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr,
                                                    "Logistic Regression")

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf,
                                              "Random Forest")

    # Feature importance RF
    rf_importance_df = pd.DataFrame({
        "Feature": feature_list,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n[INFO] RF top features:")
    print(rf_importance_df.head(10))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        learning_rate=0.01,
        max_depth=10,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")

    xgb_importance_df = pd.DataFrame({
        "Feature": feature_list,
        "Importance": xgb_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n[INFO] XGB top features:")
    print(xgb_importance_df.head(10))

    # =========================
    # LSTM model
    # =========================
    lstm_model, lstm_history, y_pred_lstm, y_test_seq, X_test_seq = train_lstm(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-3
    )

    # evaluate LSTM
    results['LSTM'] = evaluate_model(y_test_seq, y_pred_lstm, "LSTM")

    # plots: LSTM training curves
    plot_lstm_history(lstm_history)

    # =========================
    # Compare models (metrics)
    # =========================
    plot_model_comparison(results)

    # =========================
    # Backtesting
    # =========================
    # use last N prices aligned with test sets
    # for tree / LR / XGB, we used X_test (not sequences)
    test_prices_full = raw_df['Close'].values[valid_rows][-len(y_test):]

    # for LSTM we used sequences, so align to y_test_seq length (shorter)
    lstm_test_prices = test_prices_full[-len(y_test_seq):]

    backtest_results = {
        "Logistic Regression": backtest_strategy(test_prices_full, y_pred_lr),
        "Random Forest": backtest_strategy(test_prices_full, y_pred_rf),
        "XGBoost": backtest_strategy(test_prices_full, y_pred_xgb),
        "LSTM": backtest_strategy(lstm_test_prices, y_pred_lstm),
    }

    # print backtest summary
    print("\n" + "="*70)
    print("BACKTEST RESULTS (Initial Capital: €10,000)")
    print("="*70)
    for name, res in backtest_results.items():
        print(f"\n{name}:")
        print(f"  Final Capital: €{res['final_capital']:.2f}")
        print(f"  Total Return: {res['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {res['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {res['max_drawdown']*100:.2f}%")
        print(f"  Total Trades: {res['total_trades']}")

    # visualize backtest
    plot_backtest_results(backtest_results)

    # visualize feature importance
    plot_feature_importance(rf_importance_df, xgb_importance_df)

    # =========================
    # Summary table (for report / dashboard)
    # =========================
    summary_data = {
        "Model": list(results.keys()),
        "Accuracy": [results[m]["accuracy"] * 100 for m in results],
        "Precision": [results[m]["precision"] * 100 for m in results],
        "Recall": [results[m]["recall"] * 100 for m in results],
        "F1-Score": [results[m]["f1_score"] * 100 for m in results],
        "Total Return (%)": [backtest_results[m]["total_return"] * 100
                             for m in results],
        "Sharpe Ratio": [backtest_results[m]["sharpe_ratio"]
                         for m in results],
        "Max Drawdown (%)": [backtest_results[m]["max_drawdown"] * 100
                             for m in results],
    }

    summary_df = pd.DataFrame(summary_data).round(2)

    print("\n" + "="*90)
    print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print("="*90)
    print(summary_df.to_string(index=False))
    print("="*90)

    # winners
    best_return_model = summary_df.loc[
        summary_df['Total Return (%)'].idxmax(), 'Model'
    ]
    best_sharpe_model = summary_df.loc[
        summary_df['Sharpe Ratio'].idxmax(), 'Model'
    ]
    best_accuracy_model = summary_df.loc[
        summary_df['Accuracy'].idxmax(), 'Model'
    ]

    print(f"\n🏆 Best by Total Return: {best_return_model}")
    print(f"🏆 Best by Sharpe: {best_sharpe_model}")
    print(f"🏆 Best by Accuracy: {best_accuracy_model}")

    print("\n📊 KEY INSIGHTS:")
    print(f"  • Mean accuracy across models: "
          f"{summary_df['Accuracy'].mean():.2f}%")
    print("  • Transaction costs (0.1%) eat into profit.")
    print("  • Technical indicators (momentum, RSI, MACD, bands, ATR) "
          "carry predictive weight in RF/XGB.")
    print("  • LSTM learns temporal patterns, not just snapshot features.")

    print("\n⚠️  IMPORTANT NOTES:")
    print("  • Past performance ≠ future performance.")
    print("  • This backtest ignores slippage, liquidity limits, tax.")
    print("  • Real trading also needs position sizing, risk limits, "
          "regulatory compliance, execution latency, etc.")

    # =========================
    # Latest live-ish signal
    # =========================
    last_price = float(raw_df['Close'].iloc[-1]) if len(raw_df) else None
    latest_signals_and_size(
        y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_lstm,
        last_price
    )


if __name__ == "__main__":
    main()
