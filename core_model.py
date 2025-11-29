import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Additional imports for reinforcement learning
import random

# Global Constants
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.001  # 0.1%
BATCH_SIZE = 32
NUM_EPOCHS = 50
LSTM_HIDDEN = 500
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")




# Data Loading Function
def get_stock_data(ticker, period="5y", interval="1d"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# Feature Engineering Function
def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / ((df['BB_Upper'] - df['BB_Lower']) + 1e-10)

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    df['Price_Position'] = (df['Close'] - df['Low']) / ((df['High'] - df['Low']) + 1e-10)

    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['ROC'] = (df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-10) * 100
    df['MA_Ratio_10_20'] = df['SMA_10'] / (df['SMA_20'] + 1e-10)
    df['MA_Ratio_20_50'] = df['SMA_20'] / (df['SMA_50'] + 1e-10)
    return df


# Label Creation Function
def create_labels(df, prediction_threshold=0.01):
    future_return = df['Close'].shift(-1) / df['Close'] - 1
    labels = np.where(
        future_return > prediction_threshold, 1,
        np.where(future_return < -prediction_threshold, 0, -1)
    )
    valid_mask = labels != -1
    return labels, valid_mask


# Model Evaluation Function
def evaluate_model(y_true, y_pred, model_name="Model"):
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


# Backtest Function
def backtest_strategy(price_data, predictions, initial_capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST):
    capital = initial_capital
    position = 0
    portfolio_value = [initial_capital]
    trades = []

    for i in range(1, len(predictions)):
        px = price_data[i]

        if predictions[i] == 1 and position == 0:
            shares = capital / px
            capital = 0
            position = 1
            trades.append(("BUY", i, px, shares))

        elif predictions[i] == 0 and position == 1:
            shares = trades[-1][3]
            capital = shares * px * (1 - transaction_cost)
            position = 0
            trades.append(("SELL", i, px, shares))

        if position == 1:
            shares = trades[-1][3]
            portfolio_value.append(shares * px)
        else:
            portfolio_value.append(capital)

    if position == 1 and len(trades) > 0:
        shares = trades[-1][3]
        final_capital = shares * px * (1 - transaction_cost)
    else:
        final_capital = portfolio_value[-1]

    total_return = (final_capital - initial_capital) / initial_capital
    portfolio_value = np.array(portfolio_value)
    returns = np.diff(portfolio_value) / (portfolio_value[:-1] + 1e-10)

    sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
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


# ==========================
# Reinforcement Learning (Q-Learning) Utilities
# ==========================

def _get_state_indices(row):
    """
    Discretize momentum (price change) and RSI to derive a compact state index.

    States are a combination of:
      - Momentum sign: -1 (down), 0 (flat), 1 (up)
      - RSI category: 0 (oversold <30), 1 (neutral 30-70), 2 (overbought >70)

    The resulting state index = (momentum_sign + 1) * 3 + rsi_cat, giving 9 possible states.
    If values are NaN, a neutral state (index 4) is returned.
    """
    try:
        momentum = row.get('Price_Change', 0.0)
        if np.isnan(momentum):
            momentum_sign = 0
        elif momentum > 0.01:
            momentum_sign = 1
        elif momentum < -0.01:
            momentum_sign = -1
        else:
            momentum_sign = 0

        rsi = row.get('RSI', 50.0)
        if np.isnan(rsi):
            rsi_cat = 1
        elif rsi < 30:
            rsi_cat = 0
        elif rsi > 70:
            rsi_cat = 2
        else:
            rsi_cat = 1
        return (momentum_sign + 1) * 3 + rsi_cat
    except Exception:
        return 4  # default neutral state


def train_q_learning(feat_df, alpha=0.1, gamma=0.95, epsilon=0.1, episodes=1):
    """
    Train a simple Q-learning agent on price data and engineered features.

    Parameters
    ----------
    feat_df : pd.DataFrame
        DataFrame with engineered features including 'Price_Change' and 'RSI'.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Exploration rate.
    episodes : int
        Number of passes over the data.

    Returns
    -------
    Q : np.ndarray
        Learned Q-table of shape (9 states, 3 actions).
    """
    # Ensure necessary columns are available
    if 'Price_Change' not in feat_df.columns or 'RSI' not in feat_df.columns:
        raise ValueError("Feature DataFrame must contain 'Price_Change' and 'RSI' columns for RL training")

    prices = feat_df['Close'].values
    # Reset index for integer-based access
    df = feat_df.reset_index(drop=True)
    n = len(df) - 1  # because we reference t and t+1
    Q = np.zeros((9, 3), dtype=float)
    for _ in range(max(1, episodes)):
        for t in range(n):
            # skip if next price is NaN
            if np.isnan(prices[t]) or np.isnan(prices[t+1]):
                continue
            state = _get_state_indices(df.loc[t])
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)  # 0=hold,1=buy,2=sell
            else:
                action = int(np.argmax(Q[state]))
            # compute reward as price difference times action direction
            price_diff = prices[t+1] - prices[t]
            if action == 1:       # buy
                reward = price_diff
            elif action == 2:     # sell
                reward = -price_diff
            else:                 # hold
                reward = 0.0
            # next state
            next_state = _get_state_indices(df.loc[t+1])
            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    return Q


def rl_predict_actions(feat_df, Q):
    """
    Derive optimal actions using learned Q-table for each row in the feature DataFrame.

    Returns
    -------
    actions : np.ndarray
        Array of length len(feat_df) containing actions: 0=hold, 1=buy, 2=sell.
    """
    df = feat_df.reset_index(drop=True)
    actions = []
    for _, row in df.iterrows():
        state = _get_state_indices(row)
        # choose the action with maximum Q-value
        action = int(np.argmax(Q[state]))
        actions.append(action)
    return np.array(actions, dtype=int)


def evaluate_rl_actions(y_true, y_pred):
    """
    Evaluate RL predicted actions against true labels using classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels from create_labels (values in {-1,0,1}).
    y_pred : array-like
        RL predicted actions mapped to same label space (-1, 0, 1).

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1 score and confusion matrix.
    """
    # Filter out neutral (-1) targets for evaluation
    mask = y_true != -1
    if mask.sum() == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan, "confusion_matrix": None}
    y_t = y_true[mask]
    y_p = y_pred[mask]
    # Compute metrics for 3-class classification (labels: -1, 0, 1)
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, average='weighted', zero_division=0)
    rec = recall_score(y_t, y_p, average='weighted', zero_division=0)
    f1 = f1_score(y_t, y_p, average='weighted', zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=[-1, 0, 1])
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def backtest_strategy_rl(price_data, actions, initial_capital=INITIAL_CAPITAL, transaction_cost=TRANSACTION_COST):
    """
    Backtest a trading strategy where actions can be hold (0), buy (1), or sell (2).

    The strategy starts with no position. Buying opens a long position; selling closes an existing position. Holds leave position unchanged.

    Parameters
    ----------
    price_data : array-like
        Sequence of closing prices.
    actions : array-like
        Sequence of actions (0=hold, 1=buy, 2=sell) for each timestep.
    initial_capital : float
        Starting capital in euros.
    transaction_cost : float
        Proportional cost incurred when closing a position.

    Returns
    -------
    dict
        Backtest statistics similar to backtest_strategy.
    """
    capital = initial_capital
    position = 0  # 0 = flat, 1 = long
    shares = 0.0
    portfolio_value = [initial_capital]
    trades = []
    prices = np.array(price_data)
    n = len(actions)
    for i in range(1, n):
        px = prices[i]
        action = actions[i]
        if action == 1 and position == 0:  # buy
            # open long
            shares = capital / px
            capital = 0.0
            position = 1
            trades.append(("BUY", i, px, shares))
        elif action == 2 and position == 1:  # sell
            # close long
            capital = shares * px * (1 - transaction_cost)
            position = 0
            trades.append(("SELL", i, px, shares))
            shares = 0.0
        # compute current portfolio value
        if position == 1:
            portfolio_value.append(shares * px)
        else:
            portfolio_value.append(capital)
    # close remaining position at end of series
    final_capital = portfolio_value[-1]
    if position == 1:
        final_capital = shares * prices[-1] * (1 - transaction_cost)
    total_return = (final_capital - initial_capital) / initial_capital
    pv = np.array(portfolio_value)
    if len(pv) > 1:
        returns = np.diff(pv) / (pv[:-1] + 1e-10)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdown)
    else:
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "total_trades": len(trades),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "portfolio_value": pv,
        "trades": trades
    }

def predict_tomorrow_price(feat_df, model, scaler, feature_list):
    """Predict tomorrow's stock price based on last feature row."""
    last_row = feat_df.iloc[-1][feature_list].values.reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)
    predicted_price = model.predict(last_row_scaled)[0]
    return predicted_price



# Main Pipeline to run all steps
def run_pipeline(ticker="AAPL"):
    # Step 1: Data Loading
    raw_df = get_stock_data(ticker, period="5y", interval="1d")

    # Step 2: Feature Engineering
    feat_df = calculate_technical_indicators(raw_df)

    # Step 3: Create Labels
    labels, valid_mask = create_labels(feat_df)

    # Prepare X and y for classification
    feature_list = [
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower',
        'BB_Width', 'BB_Position', 'ATR', 'Volume_SMA', 'Volume_Ratio',
        'Price_Change', 'Price_Range', 'Price_Position', 'Momentum', 'ROC',
        'MA_Ratio_10_20', 'MA_Ratio_20_50'
    ]
    X_full = feat_df[feature_list].values
    y_full = labels
    # select rows where classification labels are valid and no NaN in features
    valid_rows = valid_mask & ~np.isnan(X_full).any(axis=1)
    X_full = X_full[valid_rows]
    y_full = y_full[valid_rows]

    # Step 4: Train/Test Split and Scaling for classification
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train Models (LogReg, RF, XGB, LSTM)
    results = {}

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, "Logistic Regression")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")

    # LSTM Model (Train, Evaluate, and Record Results)

    class lstmDataset(Dataset):
        def __init__(self, X, y, seq_length=SEQUENCE_LENGTH):
            self.X = X
            self.y = y
            self.seq_length = seq_length

        def __len__(self):
            return len(self.X) - self.seq_length

        def __getitem__(self, idx):
            X_seq = self.X[idx:idx + self.seq_length]
            y_label = self.y[idx + self.seq_length]
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)
        
    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super(LSTMModel, self).__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = torch.nn.Linear(hidden_size, 2)

        def forward(self, x):
            h_lstm, _ = self.lstm(x)
            out = self.fc(h_lstm[:, -1, :])
            return out
    # LSTM-specific steps here...
    # LSTM-specific steps here...

    # -------------------------------------------------------
    # Reinforcement Learning: Q-Learning agent
    # -------------------------------------------------------
    try:
        # Train Q-learning agent on full feature DataFrame
        Q_table = train_q_learning(feat_df.dropna(subset=['Price_Change', 'RSI']), alpha=0.1, gamma=0.95, epsilon=0.1, episodes=1)
        rl_actions = rl_predict_actions(feat_df, Q_table)
        # Map RL actions to label space: 0=hold->-1, 1=buy->1, 2=sell->0
        rl_pred_labels = np.array([-1 if a == 0 else (1 if a == 1 else 0) for a in rl_actions])
        rl_eval = evaluate_rl_actions(labels, rl_pred_labels)
    except Exception as e:
        warnings.warn(f"RL training failed: {e}")
        rl_actions = np.zeros(len(feat_df), dtype=int)
        rl_pred_labels = np.array([-1] * len(feat_df))
        rl_eval = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan, "confusion_matrix": None}




    # Step 6: Backtest Models
    backtest_results = {
        "Logistic Regression": backtest_strategy(raw_df['Close'].values, y_pred_lr),
        "Random Forest": backtest_strategy(raw_df['Close'].values, y_pred_rf),
        "XGBoost": backtest_strategy(raw_df['Close'].values, y_pred_xgb)
    }

    # Backtest RL strategy
    backtest_results_rl = backtest_strategy_rl(raw_df['Close'].values, rl_actions)
    # Include RL results in backtest_results dict with a descriptive key
    backtest_results["Q-Learning RL"] = backtest_results_rl



    # Step 7: Build Summary DataFrame With Backtest Metrics
    summary_rows = []
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        row = {
            "Model": model_name,
            "Accuracy": results[model_name]["accuracy"],
            "Total Return (%)": backtest_results[model_name]["total_return"] * 100,
            "Sharpe Ratio": backtest_results[model_name]["sharpe_ratio"],
            "Max Drawdown (%)": backtest_results[model_name]["max_drawdown"] * 100
        }
        summary_rows.append(row)

    # Append RL performance to summary
    summary_rows.append({
        "Model": "Q-Learning RL",
        "Accuracy": rl_eval["accuracy"],
        "Total Return (%)": backtest_results_rl["total_return"] * 100,
        "Sharpe Ratio": backtest_results_rl["sharpe_ratio"],
        "Max Drawdown (%)": backtest_results_rl["max_drawdown"] * 100
    })

    summary_df = pd.DataFrame(summary_rows)
    # -------------------------------------------------------
    # Price Prediction: Train regression model to forecast next-day close
    # -------------------------------------------------------
    # Use the same feature list; target is next day's close price
    y_reg_full = feat_df['Close'].shift(-1).values
    # Use rows with valid features and non-NaN target
    valid_reg_rows = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_reg_full[valid_rows])
    X_reg = X_full[valid_reg_rows]
    y_reg = y_reg_full[valid_rows][valid_reg_rows]
    # Train/test split for regression
    if len(y_reg) > 0:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        scaler_reg = StandardScaler()
        X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
        X_test_r_scaled = scaler_reg.transform(X_test_r)
        reg_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
        reg_model.fit(X_train_r_scaled, y_train_r)
        # For last observation, predict next-day price
        last_feature_row = feat_df.iloc[-1][feature_list].values.reshape(1, -1)
        last_feature_row_scaled = scaler_reg.transform(last_feature_row)
        predicted_next_price_ml = reg_model.predict(last_feature_row_scaled)[0]
    else:
        predicted_next_price_ml = np.nan
    # RL predicted next price: add Q-value of buy action (expected price diff)
    # If RL training succeeded, compute predicted diff as Q[state,buy]
    try:
        last_state = _get_state_indices(feat_df.iloc[-1])
        predicted_diff_rl = Q_table[last_state, 1]  # expected reward for buying
        predicted_next_price_rl = feat_df['Close'].iloc[-1] + predicted_diff_rl
    except Exception:
        predicted_next_price_rl = np.nan
    # RL recommended action for last state
    try:
        rl_action_last = ['Hold', 'Buy', 'Sell'][int(np.argmax(Q_table[last_state]))]
    except Exception:
        rl_action_last = None

    # Consolidate all results
    return {
        "raw_df": raw_df,
        "feat_df": feat_df,
        "results": results,
        "rl_results": rl_eval,
        "backtest_results": backtest_results,
        "summary_df": summary_df,
        "predicted_price_ml": predicted_next_price_ml,
        "predicted_price_rl": predicted_next_price_rl,
        "rl_action_last": rl_action_last,
        "rl_actions": rl_actions
    }

    
