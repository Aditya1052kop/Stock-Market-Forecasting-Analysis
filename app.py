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

# Custom CSS for dark-gold theme
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
    </style>
""", unsafe_allow_html=True)

# ===============================================================
# 🧩 Dashboard Header
# ===============================================================
st.markdown("<h1 style='text-align: center;'>📊 NIFTY 50 vs NIFTY BeES: Time-Series Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze, Compare, and Forecast the Performance of NIFTY 50 and NIFTY BeES using SARIMAX & Prophet Models</p>", unsafe_allow_html=True)
st.markdown("---")

# ===============================================================
# 📂 Sidebar Inputs
# ===============================================================
st.sidebar.header("📁 Upload Datasets")
nifty50_file = st.sidebar.file_uploader("Upload NIFTY 50 CSV", type=["csv"])
niftybees_file = st.sidebar.file_uploader("Upload NIFTY BeES CSV", type=["csv"])
forecast_days = st.sidebar.slider("Select Future Forecast Days", 10, 90, 30)
train_ratio = st.sidebar.slider("Training Data Ratio", 0.6, 0.9, 0.8)

# ===============================================================
# 📦 Load & Clean Data
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
# 🧾 Main Logic
# ===============================================================
if nifty50_file and niftybees_file:
    st.success("✅ Both datasets uploaded successfully!")

    nifty50 = load_price_data(nifty50_file, "NIFTY50")
    niftybees = load_price_data(niftybees_file, "NIFTYBEES")
    df = pd.merge(nifty50, niftybees, on="Date", how="inner").dropna()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "📊 EDA", "🔮 Forecasting", "💾 Download"])

    # ===========================================================
    # 🧭 Tab 1: Overview
    # ===========================================================
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Total Records:** {len(df)}")
        st.line_chart(df.set_index("Date")[["Close_NIFTY50", "Close_NIFTYBEES"]])

    # ===========================================================
    # 📊 Tab 2: EDA
    # ===========================================================
    with tab2:
        st.subheader("Exploratory Data Analysis")
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
        ax.legend(); ax.set_title("NIFTY 50 vs NIFTY BeES")
        st.pyplot(fig)

        st.markdown("#### Tracking Error (%)")
        st.area_chart(df.set_index("Date")["Tracking_Error_%"])

    # ===========================================================
    # 🔮 Tab 3: Forecasting (Updated)
    # ===========================================================
    with tab3:
        st.subheader("Forecasting Models (SARIMAX & Prophet)")
        df["Log_NIFTY50"] = np.log(df["Close_NIFTY50"])
        df["Log_NIFTYBEES"] = np.log(df["Close_NIFTYBEES"])

        split = int(len(df) * train_ratio)
        train = df.iloc[:split]
        test = df.iloc[split:]

        # --- SARIMAX Forecast for NIFTY 50 ---
        st.markdown("### 📈 SARIMAX Forecast for NIFTY 50")
        model = SARIMAX(train["Log_NIFTY50"], order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=forecast_days)
        mean_log = fc.predicted_mean
        ci = fc.conf_int()
        forecast_price = np.exp(mean_log)
        lower_ci = np.exp(ci.iloc[:, 0])
        upper_ci = np.exp(ci.iloc[:, 1])
        future_index = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

        future_forecast_nifty50 = pd.DataFrame({
            "Date": future_index,
            "Predicted_Price": forecast_price.values,
            "Lower_CI": lower_ci.values,
            "Upper_CI": upper_ci.values
        })

        st.success("✅ NIFTY 50 Future Forecast Generated Successfully!")
        st.dataframe(future_forecast_nifty50.style.format({
            "Predicted_Price": "{:.2f}",
            "Lower_CI": "{:.2f}",
            "Upper_CI": "{:.2f}"
        }), use_container_width=True)

        mae = mean_absolute_error(test["Close_NIFTY50"].iloc[-len(mean_log):], np.exp(mean_log))
        rmse = math.sqrt(mean_squared_error(test["Close_NIFTY50"].iloc[-len(mean_log):], np.exp(mean_log)))
        col1, col2 = st.columns(2)
        col1.metric("MAE (NIFTY50)", f"{mae:.2f}")
        col2.metric("RMSE (NIFTY50)", f"{rmse:.2f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df["Date"], df["Close_NIFTY50"], label="Historical", color="#00BFFF")
        ax.plot(future_forecast_nifty50["Date"], future_forecast_nifty50["Predicted_Price"], label="Forecast", color="#FFD700")
        ax.fill_between(future_forecast_nifty50["Date"], future_forecast_nifty50["Lower_CI"], future_forecast_nifty50["Upper_CI"], color='gray', alpha=0.2)
        ax.legend(); ax.set_title("NIFTY 50 Forecast (SARIMAX)")
        st.pyplot(fig)

        csv_nifty50 = future_forecast_nifty50.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download NIFTY 50 Forecast (CSV)",
            data=csv_nifty50,
            file_name="NIFTY50_Future_Forecast.csv",
            mime="text/csv"
        )

        # --- Prophet Forecast for NIFTY BeES ---
        st.markdown("### 🔮 Prophet Forecast for NIFTY BeES")
        df_prophet = df[["Date", "Close_NIFTYBEES"]].rename(columns={"Date": "ds", "Close_NIFTYBEES": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        future_forecast_bees = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
        future_forecast_bees.columns = ["Date", "Predicted_Price", "Lower_CI", "Upper_CI"]

        st.success("✅ NIFTY BeES Future Forecast Generated Successfully!")
        st.dataframe(future_forecast_bees.style.format({
            "Predicted_Price": "{:.2f}",
            "Lower_CI": "{:.2f}",
            "Upper_CI": "{:.2f}"
        }), use_container_width=True)

        fig1, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(df["Date"], df["Close_NIFTYBEES"], label="Historical", color="#00BFFF")
        ax1.plot(future_forecast_bees["Date"], future_forecast_bees["Predicted_Price"], label="Forecast", color="#FFD700")
        ax1.fill_between(future_forecast_bees["Date"], future_forecast_bees["Lower_CI"], future_forecast_bees["Upper_CI"], color='gray', alpha=0.2)
        ax1.legend(); ax1.set_title("NIFTY BeES Forecast (Prophet)")
        st.pyplot(fig1)

        csv_bees = future_forecast_bees.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download NIFTY BeES Forecast (CSV)",
            data=csv_bees,
            file_name="NIFTYBEES_Future_Forecast.csv",
            mime="text/csv"
        )

    # ===========================================================
    # 💾 Tab 4: Download
    # ===========================================================
    with tab4:
        st.subheader("Download Processed Dataset")
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned & Merged Data (CSV)",
            data=csv_all,
            file_name="Merged_NIFTY_Data.csv",
            mime="text/csv"
        )

# ===============================================================
# 🧾 Footer
# ===============================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #AAAAAA;'>© 2025 | Developed by <b>[Your Name]</b> | Financial Forecasting Dashboard using Python, Streamlit, SARIMAX & Prophet</p>", unsafe_allow_html=True)

