# Predictive Maintenance and Machine Health Monitoring System

## Overview

This application provides a comprehensive dashboard for monitoring industrial equipment health and predicting maintenance needs using sensor data. It visualizes vibration patterns, temperature trends, and calculates remaining useful life (RUL) to prevent unplanned downtime.

## Data Processing Approach

### Input Data Structure

The system processes sensor data from Excel files with the following parameters:

- Timestamp (date/time of measurement)
- Temperature readings
- Vibration measurements:
  - X and Z axis RMS velocity (mm/s)
  - X and Z axis peak velocity (mm/s)
  - X and Z axis RMS acceleration (m/s²)
  - X and Z axis peak acceleration (m/s²)

### Data Processing Pipeline

1. **Data Loading and Preparation**:

   - Reads Excel file with sensor time-series data
   - Standardizes column names for internal processing
   - Handles missing values through forward/backward filling
   - Validates data structure and required columns
2. **Threshold Calculation**:

   - Calculates statistical thresholds (95th percentile) for all parameters
   - Sets critical thresholds for RUL calculation based on industry standards
   - Marks values exceeding thresholds for alert generation
3. **Trend Analysis**:

   - Calculates rolling means over 30-day windows for each parameter
   - Identifies long-term trends in vibration and temperature patterns
   - Captures cyclical patterns in the data (e.g., weekly operational cycles)

## Mathematical Models for Prediction

### Remaining Useful Life (RUL) Calculation

The system calculates RUL for each monitored parameter using the following approach:

1. **Parameter-Specific RUL**:

   - Formula: `RUL = max(0, (1 - (current_value / threshold)) * 100)`
   - This represents the percentage of remaining life based on how close the parameter is to its critical threshold
   - Values closer to the threshold result in lower RUL percentages
2. **Weighted Health Index**:

   - Combines individual parameter RULs into a single health metric
   - Uses weighted averaging with higher weights for acceleration parameters:
     - Temperature: 10% weight
     - Velocity parameters: 10% weight each
     - Acceleration parameters: 15% weight each
   - Formula: `health_index = sum(parameter_RUL * weight) / sum(weights)`

### Degradation Rate and Failure Prediction

1. **Historical Degradation Analysis**:

   - Calculates rate of health deterioration over the observed period
   - Formula: `degradation_rate = (initial_health - current_health) / time_period`
2. **Time-to-Failure Projection**:

   - Projects the health index into the future based on the observed degradation rate
   - Formula: `days_to_failure = current_health / degradation_rate`
   - Formula: `days_to_maintenance = (current_health - maintenance_threshold) / degradation_rate`

## Visualization and Interpretation

The dashboard presents multiple visualization layers:

1. **Time-Series Analysis**:

   - Raw sensor data trends over time
   - Rolling averages and threshold visualizations
   - Parameter correlations
2. **Health Assessment**:

   - Current health gauge visualization
   - Individual parameter RUL trends
   - Combined health index trajectory
3. **Predictive Timeline**:

   - Health projection based on historical degradation
   - Key event markers (degradation acceleration, maintenance due)
   - Maintenance scheduling based on projected degradation

## Using Custom Data

To use the dashboard with your own equipment sensor data:

1. Prepare an Excel file with columns for timestamp and sensor measurements
2. Upload the file using the sidebar uploader
3. Ensure your data contains at least the required columns (timestamp, temperature, vibration measurements)
4. The system will automatically calculate thresholds, health indices, and maintenance projections

## Mathematical Limitations

- The linear degradation model assumes consistent deterioration rates
- Real-world equipment may exhibit non-linear degradation patterns
- Accuracy improves with longer historical datasets and consistent measurement intervals
- External factors (operating conditions, load variations) may impact actual failure timing
