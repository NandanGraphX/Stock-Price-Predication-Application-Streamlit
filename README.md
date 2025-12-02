# 📈 Stock Price Prediction & Backtesting Application

A comprehensive stock market prediction and analysis platform built with **Streamlit**, combining traditional machine learning models, deep learning (LSTM), and reinforcement learning (Q-Learning) to predict stock prices and provide trading recommendations.

## 🎯 What This Application Does

This application provides an end-to-end solution for stock market analysis and prediction:

1. **Real-time Stock Data Retrieval** - Downloads historical stock data from Yahoo Finance
2. **Technical Analysis** - Calculates 24+ technical indicators for comprehensive market analysis
3. **Multi-Model Predictions** - Uses 4 different prediction approaches (Logistic Regression, Random Forest, XGBoost, and Q-Learning RL)
4. **Price Forecasting** - Predicts next-day closing prices using both ML and RL approaches
5. **Backtesting** - Simulates trading strategies with realistic transaction costs
6. **Performance Comparison** - Evaluates models based on returns, Sharpe ratio, and maximum drawdown
7. **Interactive Dashboard** - Visualizes predictions, portfolio performance, and technical indicators

## 🛠️ Technologies Used

### **Core Technologies**
- **Python 3.11+** - Primary programming language
- **Streamlit** - Interactive web application framework

### **Data & Analysis**
- **yfinance** - Real-time stock market data retrieval
- **pandas** - Data manipulation and time series analysis
- **numpy** - Numerical computations

### **Machine Learning**
- **scikit-learn** - Classical ML algorithms and preprocessing
  - Logistic Regression
  - Random Forest Classifier & Regressor
  - StandardScaler for feature normalization
  - Train/test splitting and cross-validation
- **XGBoost** - Gradient boosting framework for classification

### **Deep Learning**
- **PyTorch** - Neural network framework
  - LSTM (Long Short-Term Memory) networks
  - Custom Dataset and DataLoader implementations
  - GPU acceleration support (CUDA)

### **Visualization**
- **matplotlib** - Chart generation and data visualization
- **seaborn** - Statistical data visualization with enhanced aesthetics

### **Reinforcement Learning**
- **Custom Q-Learning Implementation** - State-action-reward based trading agent

## 🧠 Machine Learning & Deep Learning Models

### **1. Logistic Regression**
- **Type**: Binary Classification
- **Purpose**: Predicts whether stock price will rise or fall
- **Features**: Linear decision boundary with L2 regularization
- **Configuration**: 1000 max iterations, balanced class weights

### **2. Random Forest**
- **Type**: Ensemble Classification & Regression
- **Purpose**: Classification for buy/sell signals; Regression for price prediction
- **Features**: 
  - 100-200 decision trees for robust predictions
  - Feature importance ranking
  - Resistant to overfitting
- **Configuration**: 
  - Classification: 100 estimators, max_depth=10
  - Regression: 200 estimators for next-day price prediction

### **3. XGBoost**
- **Type**: Gradient Boosted Trees
- **Purpose**: High-performance classification for trading signals
- **Features**:
  - Sequential tree building with gradient optimization
  - Regularization to prevent overfitting
  - Superior performance on tabular data
- **Configuration**: 200 estimators, learning_rate=0.01, max_depth=10

### **4.  LSTM Neural Network** _(Architecture Defined)_
- **Type**: Deep Recurrent Neural Network
- **Architecture**:
  ```
  Input Layer → LSTM Layer 1 (500 units) → Dropout (0.2) 
  → LSTM Layer 2 (500 units) → Dropout (0.2) 
  → Fully Connected (25 units) → ReLU 
  → Output Layer (1 unit) → Sigmoid
  ```
- **Purpose**: Captures temporal dependencies in stock price movements
- **Features**:
  - Two stacked LSTM layers with 500 hidden units each
  - Dropout layers (20%) to prevent overfitting
  - Processes 30-day rolling windows (sequences)
  - Binary classification output (buy/sell signal)
- **Training Configuration**:
  - Batch Size: 32
  - Epochs: 50
  - Loss Function: Binary Cross-Entropy
  - Optimizer: Adam (lr=1e-3)
  - Device: GPU (CUDA) if available, else CPU

**How LSTM Generates Output:**
1. **Sequence Creation**: Takes 30 consecutive days of technical indicators
2. **Forward Pass**: 
   - First LSTM layer processes sequential data, capturing short-term patterns
   - Dropout prevents overfitting
   - Second LSTM layer learns higher-level temporal features
   - Takes final timestep output from second LSTM
3. **Classification**: Fully connected layers map LSTM output to buy/sell probability
4. **Prediction**: Sigmoid activation produces probability (>0.5 = Buy, ≤0.5 = Sell)

### **5. Q-Learning Reinforcement Agent**
- **Type**: Model-Free Reinforcement Learning
- **Purpose**: Learn optimal trading policy through trial and error
- **State Space**: 9 discrete states based on:
  - **Momentum**: Price change direction (down/flat/up)
  - **RSI**: Relative Strength Index (oversold <30, neutral 30-70, overbought >70)
- **Action Space**: 3 actions (Hold, Buy, Sell)
- **Reward Function**: Price difference weighted by action direction
  - Buy action: Rewarded by positive price change
  - Sell action: Rewarded by negative price change
  - Hold action: Zero reward
- **Q-Table**: 9×3 matrix storing expected rewards for state-action pairs
- **Learning Parameters**:
  - Alpha (learning rate): 0.1
  - Gamma (discount factor): 0.95
  - Epsilon (exploration): 0.1 (ε-greedy policy)

**How Q-Learning Works:**
1. **State Discretization**: Converts continuous features (momentum, RSI) into discrete state index
2. **Action Selection**: ε-greedy policy (10% random exploration, 90% exploit best Q-value)
3. **Reward Calculation**: Measures profit/loss from price movement based on action taken
4. **Q-Update**: `Q(s,a) ← Q(s,a) + α[reward + γ·max(Q(s',a')) - Q(s,a)]`
5. **Policy Extraction**: For each state, select action with maximum Q-value
6. **Price Prediction**: Uses Q-value of "buy" action as expected price differential

## 📊 Technical Indicators & Feature Engineering

The application calculates **24 technical indicators** across 5 categories:

### **Moving Averages** (6 features)
- SMA (Simple Moving Average): 10, 20, 50-day periods
- EMA (Exponential Moving Average): 10, 20, 50-day periods
- **Purpose**: Identify trend direction and momentum

### **Momentum Indicators** (4 features)
- **RSI** (Relative Strength Index, 14-day): Measures overbought/oversold conditions
- **MACD** (Moving Average Convergence Divergence): Trend-following indicator
- **MACD Signal**: 9-day EMA of MACD
- **MACD Histogram**: Difference between MACD and Signal
- **Purpose**: Detect trend reversals and momentum shifts

### **Volatility Indicators** (5 features)
- **Bollinger Bands**: Upper, Middle, Lower bands (20-day, 2σ)
- **BB Width**: Volatility measure
- **BB Position**: Price position within bands (0-1 normalized)
- **ATR** (Average True Range, 14-day): Measures market volatility
- **Purpose**: Assess market volatility and potential breakouts

### **Volume Indicators** (2 features)
- **Volume SMA** (20-day): Average trading volume
- **Volume Ratio**: Current volume vs. average (detects unusual activity)
- **Purpose**: Confirm trend strength and detect potential reversals

### **Price-Based Features** (7 features)
- **Price Change**: Daily percentage return
- **Price Range**: (High-Low)/Close ratio
- **Price Position**: Position within daily range
- **Momentum**: 10-day price momentum
- **ROC** (Rate of Change, 10-day): Velocity of price changes
- **MA Ratios**: SMA_10/SMA_20 and SMA_20/SMA_50 crossover signals
- **Purpose**: Capture price dynamics and trend relationships

## 🔬 Data Analysis & Processing Pipeline

### **Step 1: Data Acquisition**
```python
# Downloads 5 years of daily OHLCV data
get_stock_data(ticker, period="5y", interval="1d")
```
- Retrieves: Open, High, Low, Close, Volume
- Source: Yahoo Finance via yfinance API

### **Step 2: Feature Engineering**
```python
calculate_technical_indicators(raw_df)
```
- Computes all 24 technical indicators
- Handles missing values and edge cases
- Creates derived features (ratios, positions)

### **Step 3: Label Creation**
```python
create_labels(df, prediction_threshold=0.01)
```
- **Binary Classification Labels**:
  - `1` (Buy): Next-day return > +1. 0%
  - `0` (Sell): Next-day return < -1.0%
  - `-1` (Neutral): Between -1.0% and +1.0% (filtered out)
- **Purpose**: Creates actionable trading signals

### **Step 4: Data Preprocessing**
- **Handling Missing Values**: Drops rows with NaN after indicator calculation
- **Feature Scaling**: StandardScaler (mean=0, std=1) for ML models
- **Train/Test Split**: 80/20 split, stratified by labels
- **Sequence Generation** (LSTM only): Creates 30-day rolling windows

### **Step 5: Model Training**
Each model trains on engineered features:
- **Classical ML**: Trained on scaled tabular data
- **LSTM**: Trained on 30-day sequences
- **Q-Learning**: Trained on full dataset with online updates

### **Step 6: Prediction & Evaluation**
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression Metrics** (price prediction): Mean Squared Error
- **Trading Metrics**: Total Return, Sharpe Ratio, Maximum Drawdown

### **Step 7: Backtesting**
Simulates realistic trading:
```python
backtest_strategy(price_data, predictions, 
                  initial_capital=10000, 
                  transaction_cost=0.001)
```
- **Capital**: Starts with €10,000
- **Transaction Cost**: 0.1% per trade (realistic broker fees)
- **Strategy**: 
  - Buy signal (1) → Go all-in if flat
  - Sell signal (0) → Close position if holding
- **Metrics Calculated**:
  - Total Return (%)
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown (worst peak-to-trough loss)
  - Number of trades

### **Step 8: Visualization & Reporting**
- Price charts with technical overlays
- Portfolio value curves for each model
- Performance comparison tables
- Feature importance rankings

## 📥 Installation & Setup

### **Prerequisites**
- Python 3.11 or higher
- pip package manager
- (Optional) CUDA-capable GPU for LSTM acceleration

### **Installation Steps**

1. **Clone the repository**
```bash
git clone https://github.com/NandanGraphX/Stock-Price-Predication-Application-Streamlit.git
cd Stock-Price-Predication-Application-Streamlit
```

2.  **Create a virtual environment** (recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install streamlit pandas numpy yfinance scikit-learn xgboost torch matplotlib seaborn
```

**Or create a `requirements.txt` with:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
scikit-learn>=1. 3.0
xgboost>=2.0.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Application

### **Start the Streamlit App**
```bash
streamlit run app.py
```

### **Using the Application**

1. **Open Browser**: Streamlit will automatically open `http://localhost:8501`

2. **Enter Stock Ticker**: 
   - In the sidebar, enter a stock ticker symbol
   - Examples: `AAPL` (Apple), `MSFT` (Microsoft), `IFX. DE` (Infineon), `RHM. DE` (Rheinmetall)

3. **Run Analysis**: Click the "Run Analysis" button

4. **View Results**:
   - **Current Price**: Last closing price
   - **Predicted Prices**: Next-day predictions from ML and RL models
   - **Recommended Action**: RL agent's trading suggestion (Buy/Hold/Sell)
   - **Performance Table**: Model comparison with accuracy and backtest metrics
   - **Best Models**: Highlighted top performers by return, accuracy, and Sharpe ratio
   - **Price Chart**: Stock price with technical indicators
   - **Portfolio Curves**: Backtest performance over time
   - **Risk Metrics**: Visual comparison of returns, Sharpe ratio, drawdown
   - **Feature Table**: Sample of engineered technical indicators

## 📁 Project Structure

```
Stock-Price-Predication-Application-Streamlit/
│
├── app.py                    # Streamlit frontend application
├── core_model.py             # ML/DL/RL pipeline and model definitions
├── get_stocks_prices.py      # Alternative standalone script
├── research. ipynb            # Jupyter notebook for experimentation
└── __pycache__/              # Python cache files
```

### **File Descriptions**

- **`app.py`**: Main Streamlit dashboard with UI components and visualization
- **`core_model. py`**: Core functionality including:
  - Data loading and technical indicator calculation
  - Label creation and preprocessing
  - ML model training (LogReg, RF, XGBoost)
  - LSTM architecture and training
  - Q-Learning RL agent implementation
  - Backtesting engine
  - Performance evaluation
- **`get_stocks_prices. py`**: Standalone version of the pipeline (can run without Streamlit)
- **`research.ipynb`**: Experimental notebook for model development and analysis

## 🎓 Model Output Interpretation

### **Classification Models** (LogReg, RF, XGBoost, LSTM)
- **Output**: Binary prediction (0 or 1)
- **Meaning**: 
  - `1` = Buy signal (price expected to rise >1%)
  - `0` = Sell signal (price expected to fall >1%)

### **Q-Learning RL Agent**
- **Output**: Action (0, 1, or 2) + Price prediction
- **Meaning**:
  - `0` = Hold (no action)
  - `1` = Buy (open long position)
  - `2` = Sell (close position)
- **Price Prediction**: Current price + Q-value of buy action

### **Regression Model** (Random Forest Regressor)
- **Output**: Continuous value (predicted closing price)
- **Meaning**: Next trading day's expected closing price in currency units

## 📈 Performance Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Accuracy** | % of correct buy/sell predictions | >60% |
| **Total Return** | Profit/loss from backtest strategy | Positive |
| **Sharpe Ratio** | Risk-adjusted return (higher = better) | >1.0 |
| **Max Drawdown** | Largest peak-to-trough loss | <20% |
| **Precision** | % of buy signals that were correct | >0.6 |
| **Recall** | % of actual opportunities captured | >0.6 |
| **F1-Score** | Harmonic mean of precision & recall | >0.6 |

## ⚠️ Important Notes

### **Disclaimer**
- **This application is for educational and research purposes only**
- **Not financial advice**: Do not use these predictions for actual trading without proper risk assessment
- **Past performance**: Historical backtests do not guarantee future results
- **Market risk**: Stock markets are inherently unpredictable and risky

### **Known Limitations**
- Predictions are based solely on technical analysis (no fundamental data)
- 1% threshold for buy/sell signals may not suit all trading styles
- Backtesting assumes perfect execution (no slippage)
- Models may underperform in highly volatile or trending markets
- Requires stable internet connection for data download

### **Computational Requirements**
- **CPU**: Multi-core recommended for Random Forest and XGBoost
- **RAM**: Minimum 4GB (8GB+ recommended for larger datasets)
- **GPU**: Optional but accelerates LSTM training significantly
- **Network**: Required for downloading stock data from Yahoo Finance

## 🔧 Customization Options

You can modify hyperparameters in `core_model.py`:

```python
SEQUENCE_LENGTH = 30        # LSTM lookback window (days)
TEST_SIZE = 0.2             # Train/test split ratio
INITIAL_CAPITAL = 10_000    # Backtest starting capital
TRANSACTION_COST = 0.001    # Trading fees (0.1%)
BATCH_SIZE = 32             # LSTM mini-batch size
NUM_EPOCHS = 50             # LSTM training epochs
LSTM_HIDDEN = 500           # LSTM hidden units
LSTM_DROPOUT = 0.2          # Dropout probability
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more technical indicators (Stochastic, ADX, Ichimoku)
- Implement additional models (Transformer, GRU, ensemble methods)
- Add fundamental analysis features (P/E ratio, earnings, news sentiment)
- Improve RL agent (DQN, PPO, A3C)
- Add options/futures support
- Implement portfolio optimization

## 📧 Contact

**Developer**: NandanGraphX  
**GitHub**: [@NandanGraphX](https://github.com/NandanGraphX)  
**Repository**: [Stock-Price-Predication-Application-Streamlit](https://github.com/NandanGraphX/Stock-Price-Predication-Application-Streamlit)

## 📄 License

This project is available for educational use.  Please check the repository for specific licensing terms.

---

**Happy Trading!  📊💹**  
*Remember: Always do your own research and never invest more than you can afford to lose.*
