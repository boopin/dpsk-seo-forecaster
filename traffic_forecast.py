import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config for the browser tab
st.set_page_config(
    page_title="SEO Traffic Forecaster",  # Browser tab title
    page_icon="🚀",  # Favicon (emoji or file path)
    layout="centered",  # Page layout
    initial_sidebar_state="expanded"  # Expand sidebar by default
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stHeader {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stRadio>div {
        flex-direction: row;
        gap: 20px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app title and tagline
st.title("🚀 SEO Traffic Forecaster")
st.markdown("**Predict your website's organic traffic with AI-powered forecasting.**")
st.markdown("*Upload your monthly traffic data and get accurate predictions in seconds!*")

# Upload CSV or Excel file
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()

        st.sidebar.success("File uploaded successfully!")
        st.write("### 📊 Uploaded Data")
        st.write(df)

        # Check if the required columns exist
        if 'Month' not in df.columns or 'Organic Traffic' not in df.columns:
            st.error("The uploaded file must contain 'Month' and 'Organic Traffic' columns.")
            st.stop()

        # Convert 'Month' column to datetime
        try:
            df['ds'] = pd.to_datetime(df['Month'], format='%b-%y')  # Try parsing with expected format
        except ValueError:
            st.warning("The 'Month' column is not in the expected format (e.g., 'Jan-24'). Trying alternative parsing...")
            df['ds'] = pd.to_datetime(df['Month'])  # Fallback to automatic parsing

        # Check for missing or invalid dates
        if df['ds'].isnull().any():
            st.error("The 'Month' column contains invalid or missing dates. Please check your data.")
            st.stop()

        # Prepare data for Prophet
        df = df.rename(columns={'Organic Traffic': 'y'})

        # Let the user choose the forecast duration
        st.sidebar.header("📅 Forecast Settings")
        forecast_duration = st.sidebar.radio(
            "Select Forecast Duration",
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
        st.header("🔮 Forecasted Traffic")
        st.dataframe(forecast_df.style.background_gradient(cmap='Blues'), height=300)

        # Calculate percentage change for forecasted vs uploaded data
        uploaded_traffic = df['y'].sum()  # Total traffic in uploaded data
        forecasted_traffic = forecast_df['Forecasted Traffic'].sum()  # Total forecasted traffic

        if forecast_duration == "6 Months":
            uploaded_traffic_period = df['y'].tail(6).sum()  # Last 6 months of uploaded data
        else:
            uploaded_traffic_period = df['y'].sum()  # Full 12 months of uploaded data

        percentage_change = ((forecasted_traffic - uploaded_traffic_period) / uploaded_traffic_period) * 100
        percentage_change_rounded = round(percentage_change)  # Round to zero decimal places

        st.header("📈 Traffic Growth Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecasted Traffic", f"{forecasted_traffic:,}")
        col2.metric("Uploaded Traffic", f"{uploaded_traffic_period:,}")
        col3.metric("Percentage Change", f"{percentage_change_rounded}%")  # Display rounded percentage

        # Create a DataFrame for percentage change data
        percentage_change_df = pd.DataFrame({
            "Metric": ["Forecasted Traffic", "Uploaded Traffic", "Percentage Change"],
            "Value": [forecasted_traffic, uploaded_traffic_period, percentage_change_rounded]
        })

        # Export percentage change data as CSV
        csv_buffer = BytesIO()
        percentage_change_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Percentage Change Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="percentage_change.csv",
            mime="text/csv"
        )

        # Plot the forecast with a line graph
        st.header("📊 Forecasted Traffic Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Traffic', color='#4CAF50', linewidth=2)
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgreen', alpha=0.3, label='Uncertainty Range')
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
        st.header("🧩 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please check your file and try again.")
else:
    st.info("👋 Please upload a CSV or Excel file to get started.")
