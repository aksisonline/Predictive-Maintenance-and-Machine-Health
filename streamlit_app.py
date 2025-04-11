import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Industrial Equipment Monitoring & RUL Prediction", layout="wide")

# Global variables
df = None
thresholds = {}
thresholds_rul = {}
operating_hours_per_day = 16
estimated_total_lifespan_hours = 5000

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

    # Distributions (Interactive with Plotly)
    with st.expander("Distributions"):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['temperature'], name='Temperature', nbinsx=50, marker_color='red', opacity=0.6))
        fig.add_trace(go.Histogram(x=df['x_rms_vel'], name='X RMS Velocity', nbinsx=50, marker_color='blue', opacity=0.6))
        fig.add_trace(go.Histogram(x=df['z_rms_vel'], name='Z RMS Velocity', nbinsx=50, marker_color='green', opacity=0.6))
        fig.update_layout(
            title="Sensor Data Distributions",
            xaxis_title="Value",
            yaxis_title="Count",
            barmode='overlay',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap (Retained as static for simplicity, but could use Plotly)
    with st.expander("Correlation Heatmap"):
        import seaborn as sns
        import matplotlib.pyplot as plt
        def plot_correlation():
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.drop(columns=["timestamp"]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Feature Correlation")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return Image.open(buf)
        st.image(plot_correlation())

    # Degradation Trends (Interactive with Streamlit line_chart)
    with st.expander("Degradation Trends"):
        trend_df = df[['timestamp', 'temperature', 'x_rms_vel', 'z_rms_vel']].set_index('timestamp')
        st.line_chart(trend_df, use_container_width=True)
        st.markdown("**Thresholds:**")
        st.write(f"- Temperature: {thresholds['temperature']:.2f}")
        st.write(f"- X RMS Velocity: {thresholds['x_rms_vel']:.2f}")
        st.write(f"- Z RMS Velocity: {thresholds['z_rms_vel']:.2f}")

    # RUL Charts (Interactive with Plotly)
    with st.expander("RUL Charts"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_temp = px.line(df, x=df.index, y='temperature_rul', title='Temperature RUL',
                             labels={'temperature_rul': 'RUL (%)', 'index': 'Index'},
                             color_discrete_sequence=['red'])
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            fig_x = px.line(df, x=df.index, y='x_rms_vel_rul', title='X RMS Velocity RUL',
                           labels={'x_rms_vel_rul': 'RUL (%)', 'index': 'Index'},
                           color_discrete_sequence=['blue'])
            st.plotly_chart(fig_x, use_container_width=True)
        
        with col3:
            fig_z = px.line(df, x=df.index, y='z_rms_vel_rul', title='Z RMS Velocity RUL',
                           labels={'z_rms_vel_rul': 'RUL (%)', 'index': 'Index'},
                           color_discrete_sequence=['green'])
            st.plotly_chart(fig_z, use_container_width=True)

    # RUL Prediction (Interactive Bar Chart with Plotly)
    with st.expander("Predict RUL from Input"):
        temp = st.slider("Temperature", min_value=0.0, max_value=150.0, step=0.1)
        x_vel = st.slider("X RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)
        z_vel = st.slider("Z RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)

        if st.button("Predict RUL"):
            pred_temp = max(0, 100 - (temp * 0.5) + np.random.uniform(-5, 5))
            pred_x = max(0, 100 - (x_vel * 0.7) + np.random.uniform(-5, 5))
            pred_z = max(0, 100 - (z_vel * 0.6) + np.random.uniform(-5, 5))
            cum_rul = min(pred_temp, pred_x, pred_z)

            # Interactive bar chart
            fig_rul = go.Figure(data=[
                go.Bar(x=['Temperature', 'X RMS Velocity', 'Z RMS Velocity', 'Cumulative'],
                       y=[pred_temp, pred_x, pred_z, cum_rul],
                       marker_color=['red', 'blue', 'green', 'purple'],
                       text=[f'{val:.1f}%' for val in [pred_temp, pred_x, pred_z, cum_rul]],
                       textposition='auto')
            ])
            fig_rul.update_layout(
                title="Predicted RUL Comparison",
                yaxis_title="RUL (%)",
                yaxis_range=[0, 100],
                showlegend=False
            )
            st.plotly_chart(fig_rul, use_container_width=True)

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