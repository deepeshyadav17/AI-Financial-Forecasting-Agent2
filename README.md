<<<<<<< HEAD
# AI-Financial-Forecasting-Agent2
=======
# 📈 AI Agent for Financial Forecasting

> A production-ready, end-to-end stock price forecasting system that combines classical statistics (ARIMA) with deep learning (LSTM), compares both models automatically, forecasts 30 days into the future, and issues an AI-powered Buy/Sell trading signal — all wrapped in a beautiful Streamlit UI.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📊 **ARIMA Forecasting** | Auto-tunes (p, d, q) via `pmdarima.auto_arima`; produces confidence intervals |
| 🤖 **LSTM Forecasting** | Two-layer LSTM with dropout; trained end-to-end with early stopping |
| 🏆 **Model Comparison** | Side-by-side RMSE comparison; declares winning model automatically |
| 🔮 **30-Day Forecast** | Both models project prices 30 business days beyond the last date |
| 💡 **Trading Agent** | Rule-based Buy / Sell signal derived from the best model's forecast |
| 🖥️ **Streamlit UI** | Interactive web app: upload CSV, adjust settings, view charts & signals |
| 📁 **Structured Outputs** | Plots, JSON results, and saved model weights in organised sub-folders |

---

## 📂 Project Structure

```
ai_financial_agent/
│
├── data/
│   ├── raw/               ← put your CSV files here
│   └── processed/         ← reserved for derived data
│
├── src/
│   ├── data/
│   │   └── data_loader.py      ← CSV loading, date parsing, frequency alignment
│   ├── features/
│   │   └── preprocess.py       ← train/test split, normalisation, sequences
│   ├── models/
│   │   ├── arima_model.py      ← ARIMA training & future forecast
│   │   ├── lstm_model.py       ← LSTM training & future forecast
│   │   ├── evaluate.py         ← RMSE comparison between models
│   │   └── predict.py          ← combined 30-day future forecast & plot
│   ├── agent/
│   │   └── trading_agent.py    ← Buy/Sell signal logic
│   └── utils/
│       └── helpers.py          ← shared utilities
│
├── app/
│   └── streamlit_app.py        ← Streamlit web interface
│
├── outputs/
│   ├── models/            ← saved LSTM weights (.keras)
│   ├── predictions/       ← JSON results for each model
│   └── plots/             ← all generated PNG charts
│
├── requirements.txt
├── README.md
└── run.py                 ← single-command pipeline runner
```

---

## ⚡ Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/ai-financial-agent.git
cd ai-financial-agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Replace `tensorflow` with `tensorflow-gpu` in requirements.txt for faster LSTM training.

### 4a. Run the full pipeline from the command line

```bash
# Using the built-in demo data (no CSV needed)
python run.py --demo

# Using your own CSV
python run.py --csv data/raw/INFY.csv --symbol INFY

# More options
python run.py --demo --epochs 30 --forecast 30 --train_ratio 0.80
```

### 4b. Launch the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📋 CSV Format

Your stock CSV should look like this:

```
Date,Open,High,Low,Close,Volume
2021-01-04,1482.10,1510.00,1475.00,1502.35,2345678
2021-01-05,1503.00,1525.50,1498.00,1518.90,1987654
...
```

**Required columns:** `Date`, `Close`  
**Optional columns:** `Open`, `High`, `Low`, `Volume`, `Symbol`

Data sources you can use:
- [Yahoo Finance](https://finance.yahoo.com) → Historical Data → Download
- [NSE India](https://www.nseindia.com) → Historical Data
- [Kaggle Datasets](https://www.kaggle.com/datasets?search=stock+prices)

---

## 🧠 How It Works

```
CSV Data
  │
  ▼
data_loader.py   ─── parse dates, set BusinessDay freq, forward-fill gaps
  │
  ▼
preprocess.py    ─── 80/20 train-test split, MinMax scale, create LSTM sequences
  │
  ├──► arima_model.py  ─── auto_arima (p,d,q) → forecast → RMSE → plot
  │
  └──► lstm_model.py   ─── LSTM(128)→LSTM(64)→Dense(1) → forecast → RMSE → plot
         │
         ▼
      evaluate.py      ─── compare RMSE → pick best model
         │
         ▼
      predict.py       ─── 30-day future forecast (both models) → combined plot
         │
         ▼
   trading_agent.py    ─── avg_forecast > current_price? → BUY : SELL
```

---

## 📊 Outputs

After running the pipeline you will find:

| Path | Description |
|---|---|
| `outputs/plots/arima_forecast.png` | ARIMA test-period chart |
| `outputs/plots/lstm_forecast.png`  | LSTM test-period chart |
| `outputs/plots/future_forecast.png`| 30-day future forecast (both models) |
| `outputs/predictions/arima_results.json` | ARIMA RMSE, predictions, order |
| `outputs/predictions/lstm_results.json`  | LSTM RMSE, predictions |
| `outputs/predictions/model_comparison.json` | Side-by-side RMSE + best model |
| `outputs/predictions/future_predictions.json` | 30-day price arrays |
| `outputs/predictions/agent_recommendation.json` | Buy/Sell signal + stats |
| `outputs/models/lstm_model.keras`  | Saved LSTM weights |

---

## 🖥️ Screenshots

> *(Replace placeholders with actual screenshots after running the app)*

| | |
|---|---|
| ![Dashboard](https://via.placeholder.com/600x300?text=Streamlit+Dashboard) | ![Forecast](https://via.placeholder.com/600x300?text=30-Day+Forecast) |
| **Main Dashboard** | **Future Forecast Chart** |
| ![RMSE](https://via.placeholder.com/600x200?text=RMSE+Comparison) | ![Signal](https://via.placeholder.com/600x200?text=BUY+%2F+SELL+Signal) |
| **RMSE Comparison** | **AI Trading Signal** |

---

## ⚙️ Configuration Reference

| CLI Argument | Default | Description |
|---|---|---|
| `--csv` | None | Path to your stock CSV |
| `--symbol` | None | Filter by stock symbol (optional) |
| `--demo` | False | Use built-in synthetic data |
| `--train_ratio` | 0.80 | Fraction of data for training |
| `--epochs` | 20 | LSTM training epochs |
| `--forecast` | 30 | Days to forecast into the future |
| `--output_dir` | outputs | Root folder for all outputs |

---

## 🧪 Running Tests

```bash
# Quick sanity check (no external data needed)
python run.py --demo --epochs 5 --forecast 7
```

Expected output:
```
[DataLoader] Loaded 756 rows …
[ARIMA] Best order: (2, 1, 2)
[ARIMA] Test RMSE: XX.XX
[LSTM] Training complete. Final val_loss: 0.00XXXX
[LSTM] Test RMSE: XX.XX
✅ Best Model : ARIMA / LSTM
➡  Signal    : 📈 BUY  /  📉 SELL
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and time-series manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Plot generation |
| `scikit-learn` | MinMaxScaler, RMSE metric |
| `statsmodels` | ADF stationarity test |
| `pmdarima` | Automatic ARIMA order selection |
| `tensorflow` | LSTM model (Keras API) |
| `streamlit` | Web application framework |

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.  
Model predictions are **not** financial advice.  
Always consult a qualified financial professional before making investment decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with ❤️ using Python, TensorFlow, statsmodels & Streamlit*
>>>>>>> 0e4beb1 (Initial commit)
