import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

st.set_page_config(page_title="Industrial Equipment Monitoring & RUL Prediction", layout="wide")

# Global Variables
df = None
thresholds = {}
thresholds_rul = {}
OPERATING_HOURS_PER_DAY = 16
ESTIMATED_TOTAL_LIFESPAN_HOURS = 5000

# Helper function to create plot as image
def create_plot_image(plot_function):
    buf = io.BytesIO()
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
            "TimeStamp", "Temperature", "X_RMS_Vel", "Z_RMS_Vel", "X_Peak_Vel",
            "Z_Peak_Vel", "X_RMS_Accel", "Z_RMS_Accel", "X_Peak_Accel", "Z_Peak_Accel"
        ]
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df = df.dropna()

        thresholds = {
            'Temperature': np.percentile(df['Temperature'], 95),
            'X_RMS_Vel': np.percentile(df['X_RMS_Vel'], 95),
            'Z_RMS_Vel': np.percentile(df['Z_RMS_Vel'], 95)
        }

        thresholds_rul = {
            'Temperature': 100,
            'X_RMS_Vel': 0.5,
            'Z_RMS_Vel': 0.5
        }

        window_size = 30
        df['Temp_mean'] = df['Temperature'].rolling(window=window_size).mean()
        df['X_RMS_Vel_mean'] = df['X_RMS_Vel'].rolling(window=window_size).mean()
        df['Z_RMS_Vel_mean'] = df['Z_RMS_Vel'].rolling(window=window_size).mean()

        df['Temp_alert'] = df['Temperature'] > thresholds['Temperature']
        df['X_RMS_Vel_alert'] = df['X_RMS_Vel'] > thresholds['X_RMS_Vel']
        df['Z_RMS_Vel_alert'] = df['Z_RMS_Vel'] > thresholds['Z_RMS_Vel']

        def estimate_rul(data, threshold):
            return np.maximum(0, (1 - (data / threshold)) * 100)

        for param in thresholds_rul:
            if param in df.columns:
                df[f'{param}_rul'] = estimate_rul(df[param], thresholds_rul[param])

        st.success(f"Data loaded successfully! Dataset contains {len(df)} samples.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# UI Layout
st.title("Industrial Equipment Monitoring & RUL Prediction")

with st.expander("Upload Dataset"):
    uploaded_file = st.file_uploader("Upload your sensor Excel file (.xlsx):", type="xlsx")
    if uploaded_file:
        load_dataset(uploaded_file)

if df is not None:
    with st.expander("Dataset Overview"):
        st.write("### Dataset Summary")
        st.write(f"- Total Samples: {len(df)}")
        st.write(f"- Time Range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")
        st.write(f"- Missing Values: {df.isnull().sum().sum()}")

        st.write("### Statistical Summary")
        st.write(df[['Temperature', 'X_RMS_Vel', 'Z_RMS_Vel']].describe())

    with st.expander("Distributions"):
        def plot_distributions():
            sns.histplot(df['Temperature'], bins=50, kde=True, color='r', label='Temperature')
            sns.histplot(df['X_RMS_Vel'], bins=50, kde=True, color='b', label='X_RMS_Vel')
            sns.histplot(df['Z_RMS_Vel'], bins=50, kde=True, color='g', label='Z_RMS_Vel')
            plt.legend()
            plt.title("Sensor Data Distributions")
        st.image(create_plot_image(plot_distributions))

    with st.expander("Correlation Heatmap"):
        def plot_correlation():
            sns.heatmap(df.drop(columns=["TimeStamp"]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Feature Correlation")
        st.image(create_plot_image(plot_correlation))

    with st.expander("Degradation Trends"):
        def plot_degradation():
            plt.plot(df['Temperature'], label='Temperature', color='r', alpha=0.6)
            plt.axhline(thresholds['Temperature'], color='r', linestyle='--')
            plt.plot(df['X_RMS_Vel'], label='X_RMS_Vel', color='b', alpha=0.6)
            plt.axhline(thresholds['X_RMS_Vel'], color='b', linestyle='--')
            plt.plot(df['Z_RMS_Vel'], label='Z_RMS_Vel', color='g', alpha=0.6)
            plt.axhline(thresholds['Z_RMS_Vel'], color='g', linestyle='--')
            plt.legend()
            plt.title("Sensor Degradation Over Time")
        st.image(create_plot_image(plot_degradation))

    with st.expander("RUL Charts"):
        col1, col2, col3 = st.columns(3)

        def temp_rul():
            plt.plot(df['Temperature_rul'], label='Temp RUL', color='r')
            plt.title('Temperature RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL %')
            plt.legend()

        def x_rul():
            plt.plot(df['X_RMS_Vel_rul'], label='X_RMS_Vel RUL', color='b')
            plt.title('X_RMS_Vel RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL %')
            plt.legend()

        def z_rul():
            plt.plot(df['Z_RMS_Vel_rul'], label='Z_RMS_Vel RUL', color='g')
            plt.title('Z_RMS_Vel RUL')
            plt.xlabel('Index')
            plt.ylabel('RUL %')
            plt.legend()

        col1.image(create_plot_image(temp_rul))
        col2.image(create_plot_image(x_rul))
        col3.image(create_plot_image(z_rul))

    with st.expander("Predict RUL From Input"):
        temp = st.slider("Temperature", min_value=0.0, max_value=150.0, step=0.1)
        x_vel = st.slider("X_RMS_Vel", min_value=0.0, max_value=1.5, step=0.01)
        z_vel = st.slider("Z_RMS_Vel", min_value=0.0, max_value=1.5, step=0.01)

        if st.button("Predict RUL"):
            pred_temp = max(0, 100 - (temp * 0.5) + np.random.uniform(-5, 5))
            pred_x = max(0, 100 - (x_vel * 0.7) + np.random.uniform(-5, 5))
            pred_z = max(0, 100 - (z_vel * 0.6) + np.random.uniform(-5, 5))
            cum_rul = min(pred_temp, pred_x, pred_z)

            def plot_rul():
                labels = ['Temperature', 'X_RMS_Vel', 'Z_RMS_Vel', 'Cumulative']
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
                alert = "⚠️ CRITICAL: Immediate maintenance required!"
            elif cum_rul < 25:
                alert = "⚠️ WARNING: Schedule maintenance soon."
            elif cum_rul < 50:
                alert = "⚠️ CAUTION: Equipment showing wear."
            else:
                alert = "✅ Equipment is in good condition."

            st.markdown(f"### RUL Results\n- Temperature: {pred_temp:.1f}%\n- X_RMS_Vel: {pred_x:.1f}%\n- Z_RMS_Vel: {pred_z:.1f}%\n- Cumulative: {cum_rul:.1f}%")
            st.warning(alert)

    with st.expander("Threshold Values"):
        if thresholds:
            st.markdown(f"""
            - **Temperature Threshold:** {thresholds['Temperature']:.2f}
            - **X_RMS_Vel Threshold:** {thresholds['X_RMS_Vel']:.2f}
            - **Z_RMS_Vel Threshold:** {thresholds['Z_RMS_Vel']:.2f}
            """)
        else:
            st.info("Load data to view thresholds.")
else:
    st.info("Upload data to get started.")
