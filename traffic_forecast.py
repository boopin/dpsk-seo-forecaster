import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

# Set page config for the browser tab
st.set_page_config(
    page_title="SEO Traffic Forecaster",  # Browser tab title
    page_icon="🚀",  # Favicon (emoji or file path)
    layout="centered"  # Page layout
)

# Streamlit app title and tagline
st.title("🚀 SEO Traffic Forecaster")
st.markdown("**Predict your website's organic traffic for the next 6 months with AI-powered forecasting.**")
st.markdown("*Upload your monthly traffic data and get accurate predictions in seconds!*")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your CSV file with monthly traffic data", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data")
    st.write(df)

    # Prepare data for Prophet
    df = df.rename(columns={'Month': 'ds', 'Organic Traffic': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%b-%y')

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=6, freq='M')  # Forecast for next 6 months
    forecast = model.predict(future)

    # Format the forecasted data
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%b-%y')  # Format date as "Jan-25"
    forecast_df['yhat'] = forecast_df['yhat'].round().astype(int)  # Round off forecasted traffic
    forecast_df['yhat_lower'] = forecast_df['yhat_lower'].round().astype(int)  # Round off lower bound
    forecast_df['yhat_upper'] = forecast_df['yhat_upper'].round().astype(int)  # Round off upper bound

    # Rename columns for better readability
    forecast_df = forecast_df.rename(columns={
        'ds': 'Month',
        'yhat': 'Forecasted Traffic',
        'yhat_lower': 'Minimum Traffic',
        'yhat_upper': 'Maximum Traffic'
    })

    # Display forecast
    st.write("### 🔮 Forecasted Traffic for the Next 6 Months")
    st.write(forecast_df)

    # Plot the forecast with a line graph
    st.write("### 📈 Forecasted Traffic Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Traffic', color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.3, label='Uncertainty Range')
    ax.set_xlabel('Month')
    ax.set_ylabel('Organic Traffic')
    ax.set_title('SEO Traffic Forecast')
    ax.legend()
    st.pyplot(fig)

    # Plot forecast components
    st.write("### 🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
else:
    st.write("👋 Please upload a CSV file to get started.")
