import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image  # Corrected import from 'pil' to 'PIL'

# Set page config
st.set_page_config(page_title="Industrial Equipment Monitoring & RUL Prediction", layout="wide")

# Global variables
df = None
thresholds = {}
thresholds_rul = {}
operating_hours_per_day = 16
estimated_total_lifespan_hours = 5000

# Helper function to create plot as image
def create_plot_image(plot_function):
    buf = io.BytesIO()  # Fixed capitalization
    plt.figure(figsize=(10, 6))
    plot_function()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

# Load dataset function
def load_dataset(uploaded_file):
    global df, thresholds, thresholds_rul
    try:
        df = pd.read_excel(uploaded_file, header=1)
        df.columns = [
            "timestamp", "temperature", "x_rms_vel", "z_rms_vel",
            "x_peak_vel", "z_peak_vel", "x_rms_accel", "z_rms_accel",
            "x_peak_accel", "z_peak_accel"
        ]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna()

        thresholds = {
            'temperature': np.percentile(df['temperature'], 95),
            'x_rms_vel': np.percentile(df['x_rms_vel'], 95),
            'z_rms_vel': np.percentile(df['z_rms_vel'], 95)
        }

        thresholds_rul = {
            'temperature': 100,
            'x_rms_vel': 0.5,
            'z_rms_vel': 0.5
        }

        window_size = 30
        df['temp_mean'] = df['temperature'].rolling(window=window_size).mean()
        df['x_rms_vel_mean'] = df['x_rms_vel'].rolling(window=window_size).mean()
        df['z_rms_vel_mean'] = df['z_rms_vel'].rolling(window=window_size).mean()

        df['temp_alert'] = df['temperature'] > thresholds['temperature']
        df['x_rms_vel_alert'] = df['x_rms_vel'] > thresholds['x_rms_vel']
        df['z_rms_vel_alert'] = df['z_rms_vel'] > thresholds['z_rms_vel']

        def estimate_rul(data, threshold):
            return np.maximum(0, (1 - (data / threshold)) * 100)

        for param in thresholds_rul:
            if param in df.columns:
                df[f'{param}_rul'] = estimate_rul(df[param], thresholds_rul[param])

        st.success(f"Data loaded successfully! Dataset contains {len(df)} samples.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Sidebar layout
st.sidebar.header("Menu")
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your sensor Excel file (.xlsx):", type="xlsx")
    if uploaded_file:
        load_dataset(uploaded_file)

# Main content layout
st.title("Industrial Equipment Monitoring & RUL Prediction")

if df is not None:
    with st.expander("Dataset Overview", expanded=True):
        st.markdown(f"### Dataset Summary")
        st.write(f"- Total Samples: {len(df)}")
        st.write(f"- Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        st.write(f"- Missing Values: {df.isnull().sum().sum()}")

        st.write("### Statistical Summary")
        st.dataframe(df[['temperature', 'x_rms_vel', 'z_rms_vel']].describe())

    # Distributions
    with st.expander("Distributions"):
        def plot_distributions():
            sns.histplot(df['temperature'], bins=50, kde=True, color='r', label='Temperature')
            sns.histplot(df['x_rms_vel'], bins=50, kde=True, color='b', label='X RMS Velocity')
            sns.histplot(df['z_rms_vel'], bins=50, kde=True, color='g', label='Z RMS Velocity')
            plt.legend()
            plt.title("Sensor Data Distributions")
        st.image(create_plot_image(plot_distributions))

    # Correlation Heatmap
    with st.expander("Correlation Heatmap"):
        def plot_correlation():
            sns.heatmap(df.drop(columns=["timestamp"]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Feature Correlation")
        st.image(create_plot_image(plot_correlation))

    # Degradation Trends
    with st.expander("Degradation Trends"):
        def plot_degradation():
            plt.plot(df['temperature'], label='Temperature', color='r', alpha=0.6)
            plt.axhline(thresholds['temperature'], color='r', linestyle='--')
            plt.plot(df['x_rms_vel'], label='X RMS Velocity', color='b', alpha=0.6)
            plt.axhline(thresholds['x_rms_vel'], color='b', linestyle='--')
            plt.plot(df['z_rms_vel'], label='Z RMS Velocity', color='g', alpha=0.6)
            plt.axhline(thresholds['z_rms_vel'], color='g', linestyle='--')
            plt.legend()
            plt.title("Sensor Degradation Over Time")
        st.image(create_plot_image(plot_degradation))

    # RUL Charts
    with st.expander("RUL Charts"):
        col1, col2, col3 = st.columns(3)

        def temp_rul():
            plt.plot(df['temperature_rul'], label='Temperature RUL', color='r')
            plt.title('Temperature RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL (%)')
            plt.legend()

        def x_rul():
            plt.plot(df['x_rms_vel_rul'], label='X RMS Velocity RUL', color='b')
            plt.title('X RMS Velocity RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL (%)')
            plt.legend()

        def z_rul():
            plt.plot(df['z_rms_vel_rul'], label='Z RMS Velocity RUL', color='g')
            plt.title('Z RMS Velocity RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL (%)')
            plt.legend()

        col1.image(create_plot_image(temp_rul))
        col2.image(create_plot_image(x_rul))
        col3.image(create_plot_image(z_rul))

    # RUL Prediction
    with st.expander("Predict RUL from Input"):
        temp = st.slider("Temperature", min_value=0.0, max_value=150.0, step=0.1)
        x_vel = st.slider("X RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)
        z_vel = st.slider("Z RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)

        if st.button("Predict RUL"):
            pred_temp = max(0, 100 - (temp * 0.5) + np.random.uniform(-5, 5))
            pred_x = max(0, 100 - (x_vel * 0.7) + np.random.uniform(-5, 5))
            pred_z = max(0, 100 - (z_vel * 0.6) + np.random.uniform(-5, 5))
            cum_rul = min(pred_temp, pred_x, pred_z)

            def plot_rul():
                labels = ['Temperature', 'X RMS Velocity', 'Z RMS Velocity', 'Cumulative']
                values = [pred_temp, pred_x, pred_z, cum_rul]
                colors = ['red', 'blue', 'green', 'purple']
                bars = plt.bar(labels, values, color=colors)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center')
                plt.ylim(0, 100)
                plt.title("Predicted RUL Comparison")
                plt.ylabel("RUL (%)")
            st.image(create_plot_image(plot_rul))

            alert = ""
            if cum_rul < 10:
                alert = "⚠️ Critical: Immediate maintenance required!"
            elif cum_rul < 25:
                alert = "⚠️ Warning: Schedule maintenance soon."
            elif cum_rul < 50:
                alert = "⚠️ Caution: Equipment showing wear."
            else:
                alert = "✅ Equipment is in good condition."

            st.markdown(f"### RUL Results\n- Temperature: {pred_temp:.1f}%\n- X RMS Velocity: {pred_x:.1f}%\n- Z RMS Velocity: {pred_z:.1f}%\n- Cumulative: {cum_rul:.1f}%")
            st.warning(alert)

    # Threshold Values
    with st.expander("Threshold Values"):
        if thresholds:
            st.markdown(f"""
            - **Temperature Threshold:** {thresholds['temperature']:.2f}
            - **X RMS Velocity Threshold:** {thresholds['x_rms_vel']:.2f}
            - **Z RMS Velocity Threshold:** {thresholds['z_rms_vel']:.2f}
            """)
        else:
            st.info("Load data to view thresholds.")
else:
    st.info("Upload data to get started.")