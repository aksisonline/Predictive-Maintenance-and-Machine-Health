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
        # Read the Excel file directly with column names from first row
        df = pd.read_excel(uploaded_file)
        
        # Rename columns to standardized format for internal processing
        column_mapping = {
            "TimeStamp": "timestamp",
            "Temperature": "temperature", 
            "X_RMS_Vel": "x_rms_vel",
            "Z_RMS_Vel": "z_rms_vel",
            "X_Peak_Vel": "x_peak_vel", 
            "Z_Peak_Vel": "z_peak_vel",
            "X_RMS_Accel": "x_rms_accel",
            "Z_RMS_Accel": "z_rms_accel",
            "X_Peak_Accel": "x_peak_accel",
            "Z_Peak_Accel": "z_peak_accel"
        }
        
        df.rename(columns=column_mapping, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna()

        # Calculate thresholds for all relevant parameters (95th percentile)
        thresholds = {
            'temperature': np.percentile(df['temperature'], 95),
            'x_rms_vel': np.percentile(df['x_rms_vel'], 95),
            'z_rms_vel': np.percentile(df['z_rms_vel'], 95),
            'x_peak_vel': np.percentile(df['x_peak_vel'], 95),
            'z_peak_vel': np.percentile(df['z_peak_vel'], 95),
            'x_rms_accel': np.percentile(df['x_rms_accel'], 95),
            'z_rms_accel': np.percentile(df['z_rms_accel'], 95),
            'x_peak_accel': np.percentile(df['x_peak_accel'], 95),
            'z_peak_accel': np.percentile(df['z_peak_accel'], 95)
        }

        # Define critical thresholds for RUL calculation
        thresholds_rul = {
            'temperature': 100,  # Critical temperature
            'x_rms_vel': 0.5,    # mm/s (ISO 10816 standard)
            'z_rms_vel': 0.5,    # mm/s
            'x_peak_vel': 0.8,   # mm/s
            'z_peak_vel': 0.8,   # mm/s
            'x_rms_accel': 4.0,  # m/s² 
            'z_rms_accel': 4.0,  # m/s²
            'x_peak_accel': 6.0, # m/s²
            'z_peak_accel': 6.0  # m/s²
        }

        window_size = 30
        # Calculate rolling means for trend analysis
        df['temp_mean'] = df['temperature'].rolling(window=window_size).mean()
        df['x_rms_vel_mean'] = df['x_rms_vel'].rolling(window=window_size).mean()
        df['z_rms_vel_mean'] = df['z_rms_vel'].rolling(window=window_size).mean()
        df['x_rms_accel_mean'] = df['x_rms_accel'].rolling(window=window_size).mean()
        df['z_rms_accel_mean'] = df['z_rms_accel'].rolling(window=window_size).mean()

        # Set alert flags based on thresholds
        df['temp_alert'] = df['temperature'] > thresholds['temperature']
        df['x_rms_vel_alert'] = df['x_rms_vel'] > thresholds['x_rms_vel']
        df['z_rms_vel_alert'] = df['z_rms_vel'] > thresholds['z_rms_vel']
        df['x_rms_accel_alert'] = df['x_rms_accel'] > thresholds['x_rms_accel']
        df['z_rms_accel_alert'] = df['z_rms_accel'] > thresholds['z_rms_accel']

        def estimate_rul(data, threshold):
            return np.maximum(0, (1 - (data / threshold)) * 100)

        # Calculate RUL for each parameter
        for param in thresholds_rul:
            if param in df.columns:
                df[f'{param}_rul'] = estimate_rul(df[param], thresholds_rul[param])

        # Calculate a combined health index (weighted average of all parameters)
        # Higher weights for acceleration parameters as they often indicate developing issues
        weights = {
            'temperature_rul': 0.1,
            'x_rms_vel_rul': 0.1, 
            'z_rms_vel_rul': 0.1,
            'x_peak_vel_rul': 0.1,
            'z_peak_vel_rul': 0.1,
            'x_rms_accel_rul': 0.15,
            'z_rms_accel_rul': 0.15,
            'x_peak_accel_rul': 0.15,
            'z_peak_accel_rul': 0.15
        }
        
        # Make sure all columns exist before calculating weighted health index
        valid_columns = [col for col in weights.keys() if col in df.columns]
        if valid_columns:
            df['health_index'] = sum(df[col] * weights[col] for col in valid_columns) / sum(weights[col] for col in valid_columns)

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

    # Advanced Analytics Section
    with st.expander("Advanced RUL & Maintenance Analytics", expanded=True):
        st.markdown("### Advanced Remaining Useful Life Analysis")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Comprehensive Health Analysis", "Trend Forecasting", "Maintenance Scheduling"])
        
        with tab1:
            st.markdown("#### Machine Health Dashboard")
            
            # Create a visual health indicator using all parameters
            if 'health_index' in df.columns:
                # Get latest health index
                latest_health = df['health_index'].iloc[-1]
                
                # Create gauge chart for health index
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest_health,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Machine Health (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 50], 'color': "orange"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': maintenance_threshold
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create component health table
                component_health = {
                    'Component': ['Temperature', 'X-axis Velocity', 'Z-axis Velocity', 
                                 'X-axis Acceleration', 'Z-axis Acceleration'],
                    'RUL %': [
                        df['temperature_rul'].iloc[-1],
                        df['x_rms_vel_rul'].iloc[-1],
                        df['z_rms_vel_rul'].iloc[-1],
                        df['x_rms_accel_rul'].iloc[-1],
                        df['z_rms_accel_rul'].iloc[-1]
                    ],
                    'Status': [''] * 5
                }
                
                # Assign status based on RUL values
                for i, rul in enumerate(component_health['RUL %']):
                    if rul < 25:
                        component_health['Status'][i] = "⚠️ Critical"
                    elif rul < 50:
                        component_health['Status'][i] = "⚠️ Warning"
                    elif rul < 75:
                        component_health['Status'][i] = "ℹ️ Monitor"
                    else:
                        component_health['Status'][i] = "✅ Good"
                
                # Create a DataFrame for display
                health_df = pd.DataFrame(component_health)
                st.table(health_df)
            
            else:
                st.info("Health index not available. Please load data with all required columns.")
        
        with tab2:
            st.markdown("#### Long-term Trend Analysis")
            
            # Create forecasting options
            forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
            
            if st.button("Generate Forecast"):
                # Check if we have enough data for forecasting
                if len(df) > 30:
                    # For demonstration, we'll use a simple linear extrapolation
                    # In a real app, you'd use something like ARIMA, Prophet, or other forecasting models
                    
                    # Get the current health parameters and their trends
                    current_date = df['timestamp'].max()
                    
                    # Calculate days between measurements to get trend
                    days_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (60 * 60 * 24)
                    if days_span > 0:
                        # Calculate daily degradation rates for each parameter
                        params = ['temperature_rul', 'x_rms_vel_rul', 'z_rms_vel_rul', 
                                 'x_rms_accel_rul', 'z_rms_accel_rul']
                        
                        # Create forecast dataframe
                        forecast_dates = pd.date_range(
                            start=current_date, 
                            periods=forecast_days+1, 
                            freq='D'
                        )
                        
                        forecast_df = pd.DataFrame({'Date': forecast_dates})
                        
                        # For each parameter, calculate forecasted values
                        for param in params:
                            if param in df.columns:
                                # Get first and last values
                                first_val = df[param].iloc[0]
                                last_val = df[param].iloc[-1]
                                
                                # Calculate daily rate of change
                                daily_change = (last_val - first_val) / days_span
                                
                                # Generate forecast values
                                forecast_df[param] = [
                                    max(0, last_val + (daily_change * day))
                                    for day in range(forecast_days+1)
                                ]
                                
                        # Create a visualization of the forecast
                        fig = go.Figure()
                        
                        # Add historical data
                        hist_data = df[['timestamp'] + params].dropna()
                        
                        for param in params:
                            if param in hist_data.columns:
                                # Add historical data with dotted line
                                fig.add_trace(go.Scatter(
                                    x=hist_data['timestamp'],
                                    y=hist_data[param],
                                    mode='lines',
                                    line=dict(dash='dot'),
                                    name=f"Historical {param.replace('_rul', '')}"
                                ))
                                
                                # Add forecast data
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['Date'],
                                    y=forecast_df[param],
                                    mode='lines',
                                    name=f"Forecast {param.replace('_rul', '')}"
                                ))
                        
                        # Add threshold line
                        fig.add_shape(
                            type="line",
                            x0=forecast_df['Date'].min(),
                            y0=maintenance_threshold,
                            x1=forecast_df['Date'].max(),
                            y1=maintenance_threshold,
                            line=dict(color="red", width=2, dash="dash"),
                        )
                        
                        # Add annotation for threshold
                        fig.add_annotation(
                            x=forecast_df['Date'].max(),
                            y=maintenance_threshold,
                            text="Maintenance Threshold",
                            showarrow=False,
                            yshift=10
                        )
                        
                        fig.update_layout(
                            title="RUL Parameter Forecast",
                            xaxis_title="Date",
                            yaxis_title="RUL (%)",
                            legend_title="Parameters",
                            yaxis_range=[0, 100],
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find when each parameter crosses the maintenance threshold
                        threshold_crossings = {}
                        
                        for param in params:
                            if param in forecast_df.columns:
                                # Find the first day where value falls below threshold
                                below_threshold = forecast_df[forecast_df[param] < maintenance_threshold]
                                
                                if not below_threshold.empty:
                                    cross_day = (below_threshold['Date'].min() - current_date).days
                                    threshold_crossings[param] = cross_day
                                else:
                                    threshold_crossings[param] = ">90"  # Beyond forecast window
                        
                        # Display the threshold crossing table
                        st.markdown("#### Maintenance Threshold Crossings")
                        
                        cross_data = {
                            "Parameter": [p.replace("_rul", "") for p in threshold_crossings.keys()],
                            "Days Until Maintenance": list(threshold_crossings.values()),
                            "Projected Date": [
                                current_date + pd.Timedelta(days=d) if isinstance(d, int) else "Beyond forecast window"
                                for d in threshold_crossings.values()
                            ]
                        }
                        
                        cross_df = pd.DataFrame(cross_data)
                        st.table(cross_df)
                        
                        # Identify the earliest maintenance need
                        numeric_days = [d for d in threshold_crossings.values() if isinstance(d, int)]
                        
                        if numeric_days:
                            earliest_maintenance = min(numeric_days)
                            maintenance_date = current_date + pd.Timedelta(days=earliest_maintenance)
                            
                            st.markdown(f"""
                            #### Recommended Maintenance Schedule
                            
                            Based on the forecast:
                            - **Earliest maintenance needed**: In {earliest_maintenance} days
                            - **Maintenance date**: {maintenance_date.strftime('%B %d, %Y')}
                            """)
                            
                            if earliest_maintenance < 7:
                                st.error("⚠️ URGENT: Schedule maintenance immediately!")
                            elif earliest_maintenance < 30:
                                st.warning("⚠️ Plan for maintenance within the next month")
                            else:
                                st.success("✅ No immediate maintenance required")
                        else:
                            st.success("✅ No maintenance needed within the forecast window")
                    
                    else:
                        st.info("Insufficient time span in data to calculate trends")
                else:
                    st.info("Not enough data points for trend forecasting. At least 30 data points are recommended.")
        
        with tab3:
            st.markdown("#### Optimized Maintenance Schedule")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Equipment configuration
                machine_id = st.text_input("Machine ID/Name", "Motor-01")
                machine_cost = st.number_input("Machine Replacement Cost ($)", value=10000, min_value=0)
                downtime_cost = st.number_input("Downtime Cost per Day ($)", value=2000, min_value=0)
            
            with col2:
                # Maintenance configuration
                maintenance_cost = st.number_input("Maintenance Cost ($)", value=500, min_value=0)
                maintenance_days = st.number_input("Maintenance Duration (Days)", value=1, min_value=1)
                maintenance_efficiency = st.slider("Maintenance Efficiency (%)", 70, 100, 85, 
                                                  help="How much of the equipment's health is restored after maintenance")
            
            # Calculate optimal maintenance interval
            if st.button("Calculate Optimal Schedule"):
                if 'health_index' in df.columns:
                    # Get current health
                    current_health = df['health_index'].iloc[-1]
                    
                    # Calculate degradation rate (% per day)
                    if len(df) > 30:
                        time_diff_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (60 * 60 * 24)
                        if time_diff_days > 0:
                            health_degradation = (df['health_index'].iloc[0] - df['health_index'].iloc[-1]) / time_diff_days
                            health_degradation = max(0.1, health_degradation)  # Ensure a minimum degradation rate
                            
                            # Calculate days until maintenance threshold
                            days_to_maintenance = (current_health - maintenance_threshold) / health_degradation
                            days_to_maintenance = max(0, days_to_maintenance)
                            
                            # Calculate days until failure (health = 0)
                            days_to_failure = current_health / health_degradation if health_degradation > 0 else 999
                            
                            # Calculate costs
                            cost_of_maintenance = maintenance_cost + (maintenance_days * downtime_cost)
                            cost_of_failure = machine_cost + (5 * downtime_cost)  # Assume 5 days downtime for replacement
                            
                            # Calculate optimal maintenance date
                            optimal_maintenance_day = min(days_to_maintenance, days_to_failure * 0.8)
                            optimal_maintenance_date = df['timestamp'].max() + pd.Timedelta(days=optimal_maintenance_day)
                            
                            # Create visualization of maintenance optimization
                            timeline_df = pd.DataFrame({
                                'Day': range(0, int(days_to_failure) + 15),
                                'Health': [max(0, current_health - (d * health_degradation)) for d in range(0, int(days_to_failure) + 15)],
                                'Status': ['Current'] + ['Future'] * (int(days_to_failure) + 14)
                            })
                            
                            # Add maintenance threshold
                            timeline_df['Maintenance_Threshold'] = maintenance_threshold
                            
                            fig = go.Figure()
                            
                            # Plot health over time
                            fig.add_trace(go.Scatter(
                                x=timeline_df['Day'],
                                y=timeline_df['Health'],
                                mode='lines',
                                name='Equipment Health'
                            ))
                            
                            # Add maintenance threshold line
                            fig.add_shape(
                                type="line",
                                x0=0,
                                y0=maintenance_threshold,
                                x1=timeline_df['Day'].max(),
                                y1=maintenance_threshold,
                                line=dict(color="orange", width=2, dash="dash"),
                                name='Maintenance Threshold'
                            )
                            
                            # Add annotation for optimal maintenance
                            fig.add_annotation(
                                x=optimal_maintenance_day,
                                y=timeline_df['Health'].iloc[int(optimal_maintenance_day)],
                                text="Optimal Maintenance Point",
                                showarrow=True,
                                arrowhead=1,
                            )
                            
                            # Add failure point annotation
                            fig.add_annotation(
                                x=days_to_failure,
                                y=0,
                                text="Projected Failure Point",
                                showarrow=True,
                                arrowhead=1,
                            )
                            
                            fig.update_layout(
                                title="Equipment Health Projection & Optimal Maintenance Point",
                                xaxis_title="Days from Now",
                                yaxis_title="Health Index (%)",
                                yaxis_range=[0, 100],
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display maintenance recommendations
                            st.markdown(f"""
                            ### Maintenance Recommendations
                            
                            Based on current health ({current_health:.1f}%) and degradation rate ({health_degradation:.2f}% per day):
                            
                            - **Maintenance threshold crossed in**: {days_to_maintenance:.1f} days
                            - **Projected failure in**: {days_to_failure:.1f} days
                            - **Optimal maintenance point**: {optimal_maintenance_day:.1f} days from now
                            - **Recommended maintenance date**: {optimal_maintenance_date.strftime('%B %d, %Y')}
                            
                            **Cost Analysis:**
                            - Cost of preventive maintenance: ${cost_of_maintenance:,.2f}
                            - Cost of machine failure: ${cost_of_failure:,.2f}
                            - **Potential savings**: ${(cost_of_failure - cost_of_maintenance):,.2f}
                            """)
                            
                            # Create a downloadable maintenance report
                            report_data = {
                                'Machine ID': [machine_id],
                                'Current Health (%)': [current_health],
                                'Degradation Rate (%/day)': [health_degradation],
                                'Days to Maintenance Threshold': [days_to_maintenance],
                                'Days to Failure': [days_to_failure],
                                'Optimal Maintenance Day': [optimal_maintenance_day],
                                'Recommended Maintenance Date': [optimal_maintenance_date.strftime('%Y-%m-%d')],
                                'Maintenance Cost ($)': [maintenance_cost],
                                'Downtime Cost per Day ($)': [downtime_cost],
                                'Total Preventive Maintenance Cost ($)': [cost_of_maintenance],
                                'Failure Cost ($)': [cost_of_failure],
                                'Potential Savings ($)': [(cost_of_failure - cost_of_maintenance)]
                            }
                            
                            report_df = pd.DataFrame(report_data)
                            
                            # Create a CSV for download
                            csv = report_df.to_csv(index=False)
                            st.download_button(
                                label="Download Maintenance Report",
                                data=csv,
                                file_name=f"maintenance_report_{machine_id}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("Insufficient time span in data to calculate degradation rate")
                    else:
                        st.info("Not enough data points for accurate degradation analysis. Using assumed values.")
                        
                        # Use assumed values for demonstration
                        assumed_degradation = 0.5  # % per day
                        days_to_maintenance = (current_health - maintenance_threshold) / assumed_degradation
                        days_to_maintenance = max(0, days_to_maintenance)
                        
                        st.markdown(f"""
                        ### Estimated Maintenance Schedule
                        
                        Based on current health ({current_health:.1f}%) and assumed degradation rate (0.5% per day):
                        
                        - **Estimated maintenance needed in**: {days_to_maintenance:.1f} days
                        - **Recommended maintenance date**: {(df['timestamp'].max() + pd.Timedelta(days=days_to_maintenance)).strftime('%B %d, %Y')}
                        
                        Note: This is a simplified estimate. More data is needed for accurate prediction.
                        """)
                else:
                    st.error("Health index not available. Please load data with all required columns.")
else:
    st.info("Upload data to get started.")
</copilot-edited-file>
```