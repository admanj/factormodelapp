import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import getfactormodels as gfm

st.set_page_config(page_title="Mega Factor Model", layout="wide")
st.title("🚀 Your Personal Mega Factor Model App")
st.markdown("**12+ public factors** (Fama-French 6, QMJ, BAB, Liquidity, Hou-Xue-Zhang q-factors, etc.) — no Barra needed!")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TSLA, NVDA)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1))
run_button = st.sidebar.button("RUN FACTOR MODEL", type="primary")

if run_button:
    with st.spinner("Downloading data and running the model... (this takes ~15 seconds)"):
        try:
            # Load 12+ factors
            factors_obj = gfm.model(
                model=['ff6', 'qmj', 'bab', 'liq', 'q'],
                frequency='m',
                start_date=str(start_date),
                end_date=datetime.today().strftime("%Y-%m-%d")
            )
            factors = factors_obj.load().to_pandas()

            # Get stock data
            stock = yf.download(ticker, start=start_date, end=datetime.today(), progress=False)['Adj Close']
            stock_monthly = stock.resample('ME').last().pct_change().dropna()
            
            # Combine
            data = pd.concat([stock_monthly, factors], axis=1).dropna()
            data.columns = ['Stock_Ret'] + list(factors.columns)
            data['Excess_Ret'] = data['Stock_Ret'] - data.get('RF', 0)

            # Regression
            X = data.drop(columns=['Stock_Ret', 'Excess_Ret'])
            X = sm.add_constant(X)
            y = data['Excess_Ret']
            model = sm.OLS(y, X).fit()

            # Display results
            st.success(f"✅ Analysis complete for {ticker}!")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📊 Regression Results")
                st.text(model.summary().as_text())
            
            with col2:
                st.subheader("Key Numbers")
                params = model.params.round(4)
                st.write("**Alpha** (extra return):", params.get('const', 'N/A'))
                st.write("**Market Beta**:", params.get('Mkt_RF', 'N/A'))
                st.write("**R²** (how well factors explain the stock):", f"{model.rsquared:.1%}")

            # Correlation heatmap
            st.subheader("Factor Correlations")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(factors.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)

            # Download button
            csv = data.to_csv().encode()
            st.download_button("Download Full Data as CSV", csv, f"{ticker}_factor_data.csv", "text/csv")

        except Exception as e:
            st.error(f"Oops! Error: {e}. Try a different ticker or earlier start date.")

st.caption("Built just for you — powered by public academic data. Change the ticker anytime!")
