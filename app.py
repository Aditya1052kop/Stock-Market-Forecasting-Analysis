# ===============================================================
# 📊 Streamlit Dashboard: NIFTY 50 vs NIFTY BeES - Time-Series Forecasting
# Author: [Your Name]
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")

# ===============================================================
# 🎨 Dashboard Configuration
# ===============================================================
st.set_page_config(
    page_title="NIFTY 50 vs NIFTY BeES | Forecasting Dashboard",
    layout="wide",
    page_icon="📈"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .reportview-container {
            background: linear-gradient(180deg, #0e1117 0%, #1a1f2b 100%);
        }
        h1, h2, h3, h4 {
            color: #FFD700; /* Gold headings */
        }
        .stButton>button {
            background-color: #1e3d59;
            color: white;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FFD700;
            color: black;
        }
        .metric-container {
            background-color: #1e3d59;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================================================
# 🧩 Dashboard Title
# ===============================================================
st.markdown("<h1 style='text-align: center;'>📊 NIFTY 50 vs NIFTY BeES: Time-Series Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze, Compare, and Forecast the Performance of NIFTY 50 and NIFTY BeES using SARIMAX & Prophet Models</p>", unsafe_allow_html=True)
st.markdown("---")

# ===============================================================
# 📂 Sidebar: File Upload
# ===============================================================
st.sidebar.header("📁 Upload Datasets")
nifty50_file = st.sidebar.file_uploader("Upload NIFTY 50 CSV", type=["csv"])
niftybees_file = st.sidebar.file_uploader("Upload NIFTY BeES CSV", type=["csv"])
forecast_days = st.sidebar.slider("Select Future Forecast Days", 10, 90, 30)
train_ratio = st.sidebar.slider("Training Data Ratio", 0.6, 0.9, 0.8)

# ===============================================================
# 📦 Data Loading Function
# ===============================================================
def load_price_data(file, label):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    df = df[[date_col, close_col]]
    df.columns = ["Date", f"Close_{label}"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").dropna()
    return df

# ===============================================================
# 🧾 When both files are uploaded
# ===============================================================
if nifty50_file and niftybees_file:
    st.success("✅ Both datasets uploaded successfully!")
    
    # Load and merge
    nifty50 = load_price_data(nifty50_file, "NIFTY50")
    niftybees = load_price_data(niftybees_file, "NIFTYBEES")
    df = pd.merge(nifty50, niftybees, on="Date", how="inner").dropna()
    
    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "📊 EDA", "🔮 Forecasting", "💾 Download"])
    
    # ===========================================================
    # 🧭 Tab 1: Overview
    # ===========================================================
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**Columns:** " + ", ".join(df.columns))
        st.write(f"**Total Records:** {len(df)}")

        st.line_chart(df.set_index("Date")[["Close_NIFTY50", "Close_NIFTYBEES"]])

    # ===========================================================
    # 📊 Tab 2: EDA
    # ===========================================================
    with tab2:
        st.subheader("Exploratory Data Analysis (EDA)")

        df["Tracking_Error"] = df["Close_NIFTYBEES"] - df["Close_NIFTY50"]
        df["Tracking_Error_%"] = (df["Tracking_Error"] / df["Close_NIFTY50"]) * 100
        corr = df["Close_NIFTY50"].corr(df["Close_NIFTYBEES"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Correlation", f"{corr:.4f}")
        col2.metric("Mean Tracking Error (%)", f"{df['Tracking_Error_%'].mean():.4f}")
        col3.metric("Data Points", len(df))

        st.markdown("#### Price Comparison")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df["Date"], df["Close_NIFTY50"], label="NIFTY 50", color="#00BFFF")
        ax.plot(df["Date"], df["Close_NIFTYBEES"], label="NIFTY BeES", color="#FFD700")
        ax.set_title("NIFTY 50 vs NIFTY BeES Prices")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### Tracking Error Over Time")
        st.area_chart(df.set_index("Date")["Tracking_Error_%"])

        st.markdown("#### Return Correlation")
        df["Return_NIFTY50"] = df["Close_NIFTY50"].pct_change()
        df["Return_NIFTYBEES"] = df["Close_NIFTYBEES"].pct_change()
        sns.jointplot(x=df["Return_NIFTY50"], y=df["Return_NIFTYBEES"], kind="scatter", color="gold")
        st.pyplot(plt.gcf())

    # ===========================================================
    # 🔮 Tab 3: Forecasting
    # ===========================================================
    with tab3:
        st.subheader("Forecasting Models (SARIMAX & Prophet)")
        df["Log_NIFTY50"] = np.log(df["Close_NIFTY50"])
        df["Log_NIFTYBEES"] = np.log(df["Close_NIFTYBEES"])

        split = int(len(df) * train_ratio)
        train = df.iloc[:split]
        test = df.iloc[split:]

        # --- SARIMAX ---
        st.markdown("### 📈 SARIMAX Forecast for NIFTY 50")
        model = SARIMAX(train["Log_NIFTY50"], order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=len(test))
        pred = np.exp(fc.predicted_mean)
        ci = fc.conf_int()
        lower, upper = np.exp(ci.iloc[:,0]), np.exp(ci.iloc[:,1])

        mae = mean_absolute_error(test["Close_NIFTY50"], pred)
        rmse = math.sqrt(mean_squared_error(test["Close_NIFTY50"], pred))

        col1, col2 = st.columns(2)
        col1.metric("MAE (NIFTY50)", f"{mae:.2f}")
        col2.metric("RMSE (NIFTY50)", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(train["Date"], train["Close_NIFTY50"], label="Train")
        ax.plot(test["Date"], test["Close_NIFTY50"], label="Actual", color="orange")
        ax.plot(test["Date"], pred, label="Forecast", color="lime")
        ax.fill_between(test["Date"], lower, upper, color='gray', alpha=0.2)
        ax.legend(); ax.set_title("SARIMAX Forecast - NIFTY 50")
        st.pyplot(fig)

        # --- Prophet ---
        st.markdown("### 🔮 Prophet Forecast for NIFTY BeES")
        df_prophet = df[["Date", "Close_NIFTYBEES"]].rename(columns={"Date": "ds", "Close_NIFTYBEES": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    # ===========================================================
    # 💾 Tab 4: Download
    # ===========================================================
    with tab4:
        st.subheader("Download Forecast Results")
        forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
        forecast_out.rename(columns={"ds":"Date", "yhat":"Forecast", "yhat_lower":"Lower_CI", "yhat_upper":"Upper_CI"}, inplace=True)

        csv = forecast_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prophet Forecast (CSV)",
            data=csv,
            file_name="NIFTYBEES_Future_Forecast.csv",
            mime="text/csv"
        )

# ===============================================================
# 🧾 Footer
# ===============================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #AAAAAA;'>© 2025 | Developed by <b>[Your Name]</b> | Financial Forecasting Dashboard using Python, Streamlit, SARIMAX & Prophet</p>", unsafe_allow_html=True)
