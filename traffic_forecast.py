import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config for the browser tab
st.set_page_config(
    page_title="ForecastEdge - Traffic Forecasting Tool",  # Updated browser tab title
    page_icon="📈",  # Favicon (emoji or file path)
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
st.title("📈 ForecastEdge")
st.markdown("**Your AI-powered forecasting tool for accurate traffic predictions.**")
st.markdown("*Upload your monthly traffic data and get actionable insights in seconds!*")

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

        # Validate required columns
        if 'Month' not in df.columns:
            st.error("The uploaded file must contain a 'Month' column.")
            st.stop()

        traffic_columns = ['Organic Traffic', 'Traffic']
        traffic_column = next((col for col in traffic_columns if col in df.columns), None)
        if not traffic_column:
            st.error(f"The uploaded file must contain one of the following columns: {', '.join(traffic_columns)}.")
            st.stop()
        df = df.rename(columns={traffic_column: 'y'})

        # Parse and validate the 'Month' column
        df['ds'] = pd.to_datetime(df['Month'], errors='coerce')  # Coerce invalid dates to NaT
        if df['ds'].isnull().any():
            st.error("The 'Month' column contains invalid or missing dates. Please check your data.")
            st.stop()
        df = df.sort_values(by='ds')  # Ensure dates are sorted chronologically

        # Let the user choose the forecast duration
        st.sidebar.header("📅 Forecast Settings")
        forecast_duration_options = {"6 Months": 6, "12 Months": 12}
        forecast_duration = st.sidebar.radio(
            "Select Forecast Duration",
            options=list(forecast_duration_options.keys()),
            index=0  # Default to 6 Months
        )
        periods = forecast_duration_options[forecast_duration]

        # Initialize and fit Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dataframe for forecasting
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)

        # Format the forecasted data
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%b-%y')  # Format date as "Jan-25"
        forecast_df['yhat'] = forecast_df['yhat'].round().astype(int)  # Round off forecasted traffic
        forecast_df['yhat_lower'] = forecast_df['yhat_lower'].round().astype(int)  # Round off lower bound
        forecast_df['yhat_upper'] = forecast_df['yhat_upper'].round().astype(int)  # Round off upper bound
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
            mime="text/csv",
            help="Download the percentage change data as a CSV file."
        )

        # Plot the forecast with a line graph
        st.header("📊 Forecasted Traffic Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {
            "forecast_line": "#4CAF50",
            "uncertainty_fill": "lightgreen",
            "grid_lines": "#E0E0E0"
        }
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Traffic', color=colors["forecast_line"], linewidth=2)
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color=colors["uncertainty_fill"], alpha=0.3, label='Uncertainty Range')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Organic Traffic', fontsize=12)
        ax.set_title(f'SEO Traffic Forecast for the Next {forecast_duration}', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', color=colors["grid_lines"], alpha=0.6)

        # Add annotations for key insights
        for i, row in forecast_df.iterrows():
            ax.annotate(
                f"{row['Forecasted Traffic']}",
                (row['Month'], row['Forecasted Traffic']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                color='black'
            )

        st.pyplot(fig)

        # Export the graph as an image for PPT
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="📥 Download Graph for PPT",
            data=buf.getvalue(),
            file_name="traffic_forecast.png",
            mime="image/png",
            help="Download the forecast graph as a PNG image for presentations."
        )

        # Plot forecast components
        st.header("🧩 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check your file and try again.")
else:
    st.info("👋 Please upload a CSV or Excel file to get started.")
