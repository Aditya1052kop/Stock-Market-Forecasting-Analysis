# 📈 NIFTY 50 vs NIFTY BeES: Time-Series Forecasting

---

## 🧩 Introduction
The Indian stock market is one of the most dynamic financial ecosystems, driven by the movements of benchmark indices such as **NIFTY 50**.  
**NIFTY BeES (Benchmark Exchange-Traded Scheme)** is an Exchange Traded Fund (ETF) designed to replicate and track the performance of the NIFTY 50 index.  

This project performs a **comparative time-series analysis and forecasting** of NIFTY 50 and NIFTY BeES to understand how efficiently the ETF mirrors the index and to predict their future price trends using statistical and machine-learning models.

---

## 🎯 Objective
- To analyze the **relationship** between NIFTY 50 and NIFTY BeES over time.  
- To apply **time-series forecasting techniques** (SARIMAX, Prophet, etc.) for predicting future prices.  
- To measure **correlation, volatility, and tracking error** between the index and its ETF.  
- To visualize and interpret market trends for **investment insights**.

---

## ⚙️ Problem Statement
Although NIFTY BeES is intended to replicate NIFTY 50, tracking discrepancies often occur due to management costs, market inefficiencies, and timing differences.  
This project addresses the question:  
> “How closely does NIFTY BeES track NIFTY 50, and can their future price movements be accurately predicted using time-series models?”

---

## 🧠 Task
1. **Data Collection:**  
   Download daily price data for NIFTY 50 and NIFTY BeES (Jan 2024 – Nov 2025).  
2. **Data Pre-processing:**  
   Clean datasets, align dates, remove missing values, and normalize features.  
3. **Exploratory Data Analysis (EDA):**  
   Identify patterns, trends, and correlations between the index and ETF.  
4. **Feature Engineering:**  
   Compute returns, moving averages, volatility, and log transformations.  
5. **Model Building:**  
   - SARIMAX for univariate forecasting  
   - Prophet for trend prediction  
   - VAR for multi-variable (index-ETF) forecasting  
6. **Evaluation:**  
   Compare models using MAE and RMSE.  
7. **Forecasting:**  
   Generate 30-day future forecasts for both NIFTY 50 and NIFTY BeES.  

---

## 📊 Analysis
- **Correlation:** NIFTY 50 and NIFTY BeES show a correlation coefficient close to **0.99**, indicating strong co-movement.  
- **Tracking Error:** Found to be very low, confirming efficient ETF tracking.  
- **SARIMAX Results:** Forecasts capture short-term trends accurately with low error values.  
- **Prophet Forecast:** Displays clear trend continuation with seasonal stability.  
- **Volatility Analysis:** NIFTY 50 shows slightly higher short-term volatility compared to NIFTY BeES.  

📈 **Visualizations include:**  
- Price Comparison Chart  
- Rolling Mean & Volatility Plots  
- Forecast vs Actual Plots (with Confidence Intervals)

---

## 🚀 Future Scope
- Implement **LSTM / GRU** models for deep-learning-based forecasting.  
- Integrate **news sentiment analysis** to capture market sentiment impact.  
- Extend study to include other ETFs like **BANKBEES**, **JUNIORBEES**, etc.  
- Create a **dashboard (Streamlit / Power BI)** for live visualization and forecasting updates.  
- Automate data collection via **NSE / Yahoo Finance APIs**.

---

## 🏁 Conclusion
This project successfully demonstrates how **time-series forecasting** can be used to analyze and predict market movements.  
Findings confirm that **NIFTY BeES efficiently mirrors NIFTY 50** with a minimal tracking error and strong correlation.  
The developed forecasting models provide valuable insights for **investors, analysts, and researchers** interested in index-ETF performance.  

In conclusion, data-driven approaches like **SARIMAX and Prophet** serve as powerful tools for understanding and predicting stock-market dynamics in the Indian financial landscape.

---

## 👨‍💻 Project Details
**Author:** *[ADITYA ADULKAR ]*  
**Project Title:** *NIFTY 50 vs NIFTY BeES: Time-Series Forecasting*  
**Tools Used:** Python, Pandas, Matplotlib, Statsmodels, Prophet, Scikit-Learn  
**Duration:** Jan 2024 – Nov 2025  

---
