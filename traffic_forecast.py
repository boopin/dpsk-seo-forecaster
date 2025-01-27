import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config for the browser tab
st.set_page_config(
    page_title="SEO Traffic Forecaster",  # Browser tab title
    page_icon="🚀",  # Favicon (emoji or file path)
    layout="centered"  # Page layout
)

# Streamlit app title and tagline
st.title("🚀 SEO Traffic Forecaster")
st.markdown("**Predict your website's organic traffic with AI-powered forecasting.**")
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

    # Let the user choose the forecast duration
    forecast_duration = st.radio(
        "📅 Select Forecast Duration",
        options=["6 Months", "12 Months"],
        index=0  # Default to 6 Months
    )

    # Set the number of periods based on user selection
    if forecast_duration == "6 Months":
        periods = 6
    else:
        periods = 12

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=periods, freq='M')  # Forecast for selected duration
    forecast = model.predict(future)

    # Format the forecasted data
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
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
    st.write(f"### 🔮 Forecasted Traffic for the Next {forecast_duration}")
    st.write(forecast_df)

    # Calculate percentage change for forecasted vs uploaded data
    uploaded_traffic = df['y'].sum()  # Total traffic in uploaded data
    forecasted_traffic = forecast_df['Forecasted Traffic'].sum()  # Total forecasted traffic

    if forecast_duration == "6 Months":
        uploaded_traffic_period = df['y'].tail(6).sum()  # Last 6 months of uploaded data
    else:
        uploaded_traffic_period = df['y'].sum()  # Full 12 months of uploaded data

    percentage_change = ((forecasted_traffic - uploaded_traffic_period) / uploaded_traffic_period) * 100

    st.write(f"### 📈 Percentage Change in Traffic")
    st.write(f"**Forecasted Traffic for Next {forecast_duration}:** {forecasted_traffic:,}")
    st.write(f"**Uploaded Traffic for Last {forecast_duration}:** {uploaded_traffic_period:,}")
    st.write(f"**Percentage Change:** {percentage_change:.2f}%")

    # Plot the forecast with a line graph
    st.write("### 📊 Forecasted Traffic Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Traffic', color='blue', linewidth=2)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.3, label='Uncertainty Range')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Organic Traffic', fontsize=12)
    ax.set_title(f'SEO Traffic Forecast for the Next {forecast_duration}', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Export the graph as an image for PPT
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📥 Download Graph for PPT",
        data=buf.getvalue(),
        file_name="traffic_forecast.png",
        mime="image/png"
    )

    # Plot forecast components
    st.write("### 🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
else:
    st.write("👋 Please upload a CSV file to get started.")
