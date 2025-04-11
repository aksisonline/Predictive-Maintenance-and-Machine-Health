import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="Industrial Equipment Monitoring & RUL Prediction", layout="wide")

st.title("🛠️ Industrial Equipment Monitoring Dashboard")
st.markdown("""
Upload your sensor dataset to monitor key metrics and predict Remaining Useful Life (RUL) based on sensor data.
""")

uploaded_file = st.file_uploader("📁 Upload Excel File", type="xlsx")

# Helper to render seaborn/matplotlib plots in Streamlit
@st.cache_data(show_spinner=False)
def create_plot_image(plot_function):
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plot_function()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=1)
        df.columns = [
            "TimeStamp", "Temperature", "X_RMS_Vel", "Z_RMS_Vel", "X_Peak_Vel",
            "Z_Peak_Vel", "X_RMS_Accel", "Z_RMS_Accel", "X_Peak_Accel", "Z_Peak_Accel"
        ]
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df = df.dropna()

        st.success("✅ Data loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Rows", f"{df.shape[0]:,}")
        col2.metric("📊 Columns", f"{df.shape[1]:,}")
        col3.metric("🕒 Time Range", f"{df['TimeStamp'].min().date()} → {df['TimeStamp'].max().date()}")
        col4.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")

        st.markdown("---")

        # Show dataset
        with st.expander("🔍 Preview Data"):
            st.dataframe(df, use_container_width=True)

        # Statistical summary
        with st.expander("📊 Summary Statistics"):
            st.dataframe(df.describe(), use_container_width=True)

        # Distributions
        with st.expander("📌 Distributions"):
            def plot_dist():
                sns.histplot(df['Temperature'], kde=True, color='red', label='Temperature')
                sns.histplot(df['X_RMS_Vel'], kde=True, color='blue', label='X_RMS_Vel')
                sns.histplot(df['Z_RMS_Vel'], kde=True, color='green', label='Z_RMS_Vel')
                plt.legend()
                plt.title("Distribution of Sensor Readings")
            st.image(create_plot_image(plot_dist))

        # Correlation heatmap
        with st.expander("📌 Correlation Heatmap"):
            def plot_corr():
                sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Feature Correlation Matrix")
            st.image(create_plot_image(plot_corr))

        # Degradation over time
        with st.expander("📉 Degradation Trends"):
            def plot_trend():
                plt.plot(df['TimeStamp'], df['Temperature'], label='Temperature', color='red')
                plt.plot(df['TimeStamp'], df['X_RMS_Vel'], label='X_RMS_Vel', color='blue')
                plt.plot(df['TimeStamp'], df['Z_RMS_Vel'], label='Z_RMS_Vel', color='green')
                plt.title("Sensor Trends Over Time")
                plt.xlabel("Time")
                plt.ylabel("Values")
                plt.legend()
            st.image(create_plot_image(plot_trend))

        # Predictive sliders
        with st.expander("🔮 Predict RUL"):
            temp = st.slider("Temperature", 0.0, 150.0, 80.0, 0.5)
            x_vel = st.slider("X_RMS_Vel", 0.0, 2.0, 0.8, 0.01)
            z_vel = st.slider("Z_RMS_Vel", 0.0, 2.0, 0.9, 0.01)

            if st.button("Predict RUL"):
                pred_temp = max(0, 100 - (temp * 0.5) + np.random.uniform(-5, 5))
                pred_x = max(0, 100 - (x_vel * 0.7) + np.random.uniform(-5, 5))
                pred_z = max(0, 100 - (z_vel * 0.6) + np.random.uniform(-5, 5))
                cum_rul = min(pred_temp, pred_x, pred_z)

                def plot_rul():
                    labels = ['Temperature', 'X_RMS_Vel', 'Z_RMS_Vel', 'Overall']
                    values = [pred_temp, pred_x, pred_z, cum_rul]
                    colors = ['red', 'blue', 'green', 'purple']
                    bars = plt.bar(labels, values, color=colors)
                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center')
                    plt.ylim(0, 100)
                    plt.ylabel("Predicted RUL (%)")
                    plt.title("Predicted Remaining Useful Life")
                st.image(create_plot_image(plot_rul))

                if cum_rul < 10:
                    st.error("🚨 CRITICAL: Immediate maintenance required!")
                elif cum_rul < 25:
                    st.warning("⚠️ WARNING: Schedule maintenance soon.")
                elif cum_rul < 50:
                    st.info("🔧 CAUTION: Monitor equipment.")
                else:
                    st.success("✅ Equipment condition is good.")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("📤 Please upload a valid Excel file to get started.")
