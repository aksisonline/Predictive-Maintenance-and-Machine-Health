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
maintenance_threshold = 25  # RUL percentage threshold for maintenance requirement

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

    # Distributions (Vega-Lite Histogram)
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

    # Correlation Heatmap (Vega-Lite Heatmap)
    with st.expander("Correlation Heatmap"):
        # Compute correlation matrix and prepare for Vega-Lite
        corr_matrix = df.drop(columns=["timestamp"]).corr().reset_index().melt(id_vars='index', var_name='variable', value_name='correlation')
        
        vega_spec_heatmap = {
            "mark": "rect",
            "encoding": {
                "x": {
                    "field": "index",
                    "type": "nominal",
                    "title": "Variable"
                },
                "y": {
                    "field": "variable",
                    "type": "nominal",
                    "title": "Variable"
                },
                "color": {
                    "field": "correlation",
                    "type": "quantitative",
                    "scale": {"scheme": "redblue", "domain": [-1, 1]},
                    "legend": {"title": "Correlation"}
                },
                "tooltip": [
                    {"field": "index", "type": "nominal", "title": "Variable X"},
                    {"field": "variable", "type": "nominal", "title": "Variable Y"},
                    {"field": "correlation", "type": "quantitative", "title": "Correlation", "format": ".2f"}
                ]
            },
            "title": "Feature Correlation",
            "data": {"values": corr_matrix.to_dict('records')}
        }
        st.vega_lite_chart(vega_spec_heatmap, use_container_width=True)

    # Degradation Trends (Streamlit Line Chart)
    with st.expander("Degradation Trends"):
        trend_df = df[['timestamp', 'temperature', 'x_rms_vel', 'z_rms_vel']].set_index('timestamp')
        st.line_chart(trend_df, use_container_width=True)
        st.markdown("**Thresholds:**")
        st.write(f"- Temperature: {thresholds['temperature']:.2f}")
        st.write(f"- X RMS Velocity: {thresholds['x_rms_vel']:.2f}")
        st.write(f"- Z RMS Velocity: {thresholds['z_rms_vel']:.2f}")

    # RUL Charts (Plotly Line Charts)
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

    # RUL Prediction (Vega-Lite Bar Chart)
    with st.expander("Predict RUL from Input"):
        temp = st.slider("Temperature", min_value=0.0, max_value=150.0, step=0.1)
        x_vel = st.slider("X RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)
        z_vel = st.slider("Z RMS Velocity", min_value=0.0, max_value=1.5, step=0.01)

        if st.button("Predict RUL"):
            pred_temp = max(0, 100 - (temp * 0.5) + np.random.uniform(-5, 5))
            pred_x = max(0, 100 - (x_vel * 0.7) + np.random.uniform(-5, 5))
            pred_z = max(0, 100 - (z_vel * 0.6) + np.random.uniform(-5, 5))
            cum_rul = min(pred_temp, pred_x, pred_z)

            # Prepare data for Vega-Lite bar chart
            rul_data = [
                {"metric": "Temperature", "rul": pred_temp, "color": "red"},
                {"metric": "X RMS Velocity", "rul": pred_x, "color": "blue"},
                {"metric": "Z RMS Velocity", "rul": pred_z, "color": "green"},
                {"metric": "Cumulative", "rul": cum_rul, "color": "purple"}
            ]

            vega_spec_bar = {
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "title": "Metric"
                    },
                    "y": {
                        "field": "rul",
                        "type": "quantitative",
                        "title": "RUL (%)",
                        "scale": {"domain": [0, 100]}
                    },
                    "color": {
                        "field": "color",
                        "type": "nominal",
                        "scale": {"range": ["red", "blue", "green", "purple"]},
                        "legend": None
                    },
                    "tooltip": [
                        {"field": "metric", "type": "nominal", "title": "Metric"},
                        {"field": "rul", "type": "quantitative", "title": "RUL (%)", "format": ".1f"}
                    ]
                },
                "title": "Predicted RUL Comparison",
                "data": {"values": rul_data}
            }
            st.vega_lite_chart(vega_spec_bar, use_container_width=True)

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
    
    # Maintenance Period Predictor
    with st.expander("Maintenance Period Predictor", expanded=True):
        st.markdown("### Predict Days Until Next Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Configurable parameters
            user_operating_hours = st.slider("Daily Operating Hours", 
                                            min_value=1, 
                                            max_value=24, 
                                            value=operating_hours_per_day,
                                            help="Average number of hours the equipment operates per day")
            
            user_maint_threshold = st.slider("Maintenance Threshold (%)", 
                                           min_value=5, 
                                           max_value=50, 
                                           value=maintenance_threshold,
                                           help="RUL percentage at which maintenance is recommended")
        
        with col2:
            current_rul_temp = None
            current_rul_x = None
            current_rul_z = None
            
            if df is not None and 'temperature_rul' in df.columns:
                # Get latest RUL values (or averages of last few readings)
                last_n = min(10, len(df))
                current_rul_temp = df['temperature_rul'].tail(last_n).mean()
                current_rul_x = df['x_rms_vel_rul'].tail(last_n).mean()
                current_rul_z = df['z_rms_vel_rul'].tail(last_n).mean()
                
                st.markdown("### Current RUL Values")
                st.markdown(f"- Temperature: {current_rul_temp:.1f}%")
                st.markdown(f"- X RMS Velocity: {current_rul_x:.1f}%")
                st.markdown(f"- Z RMS Velocity: {current_rul_z:.1f}%")
            else:
                st.info("Load data or predict RUL to estimate maintenance period")
        
        # Calculate maintenance prediction when user clicks the button
        if st.button("Calculate Maintenance Period"):
            if df is not None and 'temperature_rul' in df.columns:
                # Use trend analysis to determine degradation rate
                if len(df) > 30:
                    # Calculate degradation rates (% per day)
                    time_diff_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (60 * 60 * 24)
                    if time_diff_days > 0:
                        temp_degradation_rate = abs((df['temperature_rul'].iloc[-1] - df['temperature_rul'].iloc[0]) / time_diff_days)
                        x_degradation_rate = abs((df['x_rms_vel_rul'].iloc[-1] - df['x_rms_vel_rul'].iloc[0]) / time_diff_days)
                        z_degradation_rate = abs((df['z_rms_vel_rul'].iloc[-1] - df['z_rms_vel_rul'].iloc[0]) / time_diff_days)
                        
                        # Use highest degradation rate to be conservative
                        max_degradation_rate = max(temp_degradation_rate, x_degradation_rate, z_degradation_rate)
                        
                        # Add a small safety factor to account for accelerated degradation
                        max_degradation_rate = max_degradation_rate * 1.1
                        
                        # Calculate days until maintenance threshold
                        current_min_rul = min(current_rul_temp, current_rul_x, current_rul_z)
                        
                        if max_degradation_rate > 0:
                            days_until_maintenance = (current_min_rul - user_maint_threshold) / max_degradation_rate
                            days_until_maintenance = max(0, days_until_maintenance)
                            
                            # Create timeline visualization
                            timeline_data = pd.DataFrame({
                                'Event': ['Today', 'Maintenance Due'],
                                'Days': [0, round(days_until_maintenance)],
                                'RUL': [current_min_rul, user_maint_threshold]
                            })
                            
                            fig = px.line(timeline_data, x='Days', y='RUL', markers=True,
                                         labels={'Days': 'Days from Now', 'RUL': 'Remaining Useful Life (%)'},
                                         title="Maintenance Timeline Forecast")
                            
                            fig.update_layout(
                                annotations=[
                                    dict(
                                        x=x,
                                        y=y,
                                        text=event,
                                        showarrow=True,
                                        arrowhead=1,
                                        ax=0,
                                        ay=-40
                                    ) for event, x, y in zip(timeline_data['Event'], timeline_data['Days'], timeline_data['RUL'])
                                ]
                            )
                            
                            # Add threshold reference line
                            fig.add_shape(
                                type="line",
                                x0=0,
                                y0=user_maint_threshold,
                                x1=days_until_maintenance,
                                y1=user_maint_threshold,
                                line=dict(color="red", width=2, dash="dash"),
                            )
                            
                            fig.update_layout(
                                yaxis_range=[0, 100],
                                xaxis_range=[-1, days_until_maintenance + 5],
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show maintenance prediction
                            maintenance_date = pd.Timestamp.now() + pd.Timedelta(days=days_until_maintenance)
                            
                            st.markdown(f"""
                            ### Maintenance Prediction Results
                            
                            Based on the current degradation trends:
                            
                            - **Days until maintenance required:** {days_until_maintenance:.1f} days
                            - **Estimated maintenance date:** {maintenance_date.strftime('%B %d, %Y')}
                            - **Current minimum RUL:** {current_min_rul:.1f}%
                            - **Maintenance threshold:** {user_maint_threshold}%
                            - **Daily RUL degradation rate:** {max_degradation_rate:.2f}% per day
                            
                            """)
                            
                            # Add maintenance urgency indicator
                            if days_until_maintenance < 7:
                                st.error("⚠️ URGENT: Maintenance required within one week!")
                            elif days_until_maintenance < 30:
                                st.warning("⚠️ Plan for maintenance within the next month")
                            else:
                                st.success("✅ No immediate maintenance required")
                        else:
                            st.info("No significant degradation detected in the data")
                    else:
                        st.info("Insufficient time span in data to calculate degradation rate")
                else:
                    # If not enough data points, use manual input and simplified calculation
                    st.info("Not enough data points for trend analysis. Using simplified estimation.")
                    
                    # Get current minimum RUL
                    current_min_rul = min(current_rul_temp, current_rul_x, current_rul_z)
                    
                    # Estimate days based on a fixed degradation rate
                    est_degradation_rate = 0.5  # % per day, conservative estimate
                    days_until_maintenance = (current_min_rul - user_maint_threshold) / est_degradation_rate
                    days_until_maintenance = max(0, days_until_maintenance)
                    
                    st.markdown(f"""
                    ### Estimated Maintenance Period
                    
                    Based on conservative estimates:
                    
                    - **Days until maintenance required:** {days_until_maintenance:.1f} days
                    - **Current minimum RUL:** {current_min_rul:.1f}%
                    - **Maintenance threshold:** {user_maint_threshold}%
                    - **Assumed daily degradation:** {est_degradation_rate}% per day
                    
                    Note: This is a simplified estimate. More data is needed for accurate prediction.
                    """)
            elif 'pred_temp' in locals() and 'pred_x' in locals() and 'pred_z' in locals():
                # Use predicted values from manual input
                current_min_rul = min(pred_temp, pred_x, pred_z)
                
                # Estimate days based on a fixed degradation rate
                est_degradation_rate = 0.5  # % per day, conservative estimate
                days_until_maintenance = (current_min_rul - user_maint_threshold) / est_degradation_rate
                days_until_maintenance = max(0, days_until_maintenance)
                
                st.markdown(f"""
                ### Estimated Maintenance Period
                
                Based on your manually input values:
                
                - **Days until maintenance required:** {days_until_maintenance:.1f} days
                - **Current minimum RUL:** {current_min_rul:.1f}%
                - **Maintenance threshold:** {user_maint_threshold}%
                - **Assumed daily degradation:** {est_degradation_rate}% per day
                """)
            else:
                st.error("Please predict RUL values first using the 'Predict RUL from Input' section")
else:
    st.info("Upload data to get started.")