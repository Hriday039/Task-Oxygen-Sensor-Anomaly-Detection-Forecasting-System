# Oxygen Sensor Anomaly Detection & Forecasting System


## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Solution Architecture](#solution-architecture)
- [Addressing Anomaly Types](#addressing-anomaly-types)
- [System Design & Lifecycle](#system-design--lifecycle)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Project Setup](#project-setup)
- [Execution Guide](#execution-guide)
- [Input Data Format](#input-data-format)
- [Limitations & Improvements](#limitations--improvements)
- [Contributing & Support](#contributing--support)

---



## Key Features

✨ **Unsupervised Anomaly Detection** using Isolation Forests on engineered time-series features

🔮 **Multi-Algorithm Forecasting** combining Prophet and ARIMA for robust predictions

🎯 **Comprehensive Anomaly Classification** detecting point, collective, contextual, and sensor fault anomalies

📊 **Severity Scoring** with actionable insights based on anomaly magnitude

🔧 **Enterprise-Grade Preprocessing** with intelligent handling of missing data and sensor gaps

✅ **Automated Evaluation** with synthetic anomaly injection for validation

---

## Solution Architecture

### Anomaly Detection

**Algorithm:** Isolation Forest (Unsupervised)

Isolation Forests are highly effective at identifying anomalies in high-dimensional feature spaces without requiring labeled training data. They are computationally efficient and handle "swamping" and "masking" better than distance-based methods.

- **Input:** A vector of 40+ engineered features per 60-minute window
- **Output:** Anomaly scores where scores < 0 indicate anomalies, with magnitude representing severity

### Forecasting

**Algorithm:** Ensemble (Prophet + ARIMA)

- **Prophet** captures seasonality (daily/weekly cycles of oxygen levels) and naturally handles missing data and outliers
- **ARIMA** models short-term residual dependencies that Prophet might miss
- **Strategy:** A weighted ensemble (60% Prophet / 40% ARIMA) ensures the model captures both macro trends and micro dynamics

---

## Addressing Anomaly Types

The core of this solution is the **RobustFeatureExtractor** module. Since Isolation Forests treat points as independent vectors, we embed temporal context directly into the features to capture different anomaly patterns.

### Point Anomalies

**Detection Strategy:** Statistical deviation from global distribution

**Implemented Features:**
- Z-scores (max/mean)
- IQR outliers
- Extreme values relative to median

### Collective Anomalies

**Detection Strategy:** Unusual sequences or patterns over time

**Implemented Features:**
- Autocorrelation (Lag 1 & 5)
- Trend strength
- Monotonicity checks
- Rate of Change (RoC) volatility

### Contextual Anomalies

**Detection Strategy:** Values that are normal globally but abnormal for the immediate context

**Implemented Features:**
- Deviation from 5-min & 10-min rolling means
- Local Outlier Factor (LOF) approximation
- Skewness & Kurtosis of the window

### Sensor Faults

**Detection Strategy:** Mechanical or electrical failures

**Implemented Features:**
- **Stuck Sensors:** consecutive_constants, zero_variance_ratio
- **Spikes:** n_spikes, max_abs_change
- **Noise:** High-frequency noise estimation (RMS of 2nd derivative)

---

## System Design & Lifecycle

### A. Data Pipeline (Preprocessing)

**Ingestion**
- Loads raw CSV data via the `DataPreprocessor` class
- Automatically detects and validates input format

**Tag-Agnostic Routing**
- Creates composite entity_id (Equipment + System) to treat all streams uniformly
- Enables seamless scaling across multiple customers and equipment types

**Cleaning & Validation**
- Does not interpolate blindly
- Gaps > 10 minutes are flagged as segment breaks to prevent "hallucinating" data during sensor downtime
- Preserves missing value information for statistical analysis

**Imputation**
- Uses an Ensemble Imputation strategy (KNN + Spline interpolation) inside the feature extraction window
- Handles minor packet loss before feature calculation without distorting the underlying signal

### B. Training Pipeline

**Feature Store Generation**
- Loops through all entities and extracts sliding window features
- Window size: 60 minutes | Slide interval: 1 minute
- Features are cached to CSV for reproducibility and faster iteration

**Global Training**
- Concatenates features from all entities into a massive training set
- Ensures the model learns from diverse equipment patterns

**Model Fitting**
- Fits one Isolation Forest on the combined dataset
- Fits Prophet models per entity or globally (configurable)
- Automatically serializes models for production deployment

**Validation**
- Uses a Synthetic Injection Engine (`create_synthetic_anomalies`) to inject known faults
- Injects spikes, dips, stuck sensors into the test set
- Calculates Precision/Recall and other classification metrics

### C. Inference & Scoring

**Anomaly Scoring**
- Outputs an Anomaly Score based on the Isolation Forest `decision_function`
- Score < 0: Anomaly detected
- Magnitude indicates severity (lower negative numbers = higher severity)

**Forecasting**
- Generates a 7-day lookahead prediction
- Provides 95% confidence intervals (yhat_lower, yhat_upper)
- Suitable for operational planning and alert generation

---

## Model Performance Evaluation

The system includes an automated evaluation module that calculates comprehensive metrics on a hold-out test set (last 7 days) with injected synthetic anomalies.

### Anomaly Detection Metrics

**ROC-AUC**
- Measures the model's ability to distinguish between normal and anomalous windows
- Ranges from 0 to 1, where 1 is perfect separation

**Precision/Recall**
- Tuned via the `contamination` hyperparameter (default 0.07)
- Precision: Minimize false positives
- Recall: Minimize missed anomalies

**Visualizations**
- **Confusion Matrix:** Shows False Positives vs False Negatives
- **Score Distribution:** Histograms demonstrating separation between normal and anomalous scores

### Forecasting Metrics

**MAE (Mean Absolute Error)**
- Average magnitude of prediction errors in absolute units

**MAPE (Mean Absolute Percentage Error)**
- Error relative to true value (useful for business context and cross-equipment comparison)

**RMSE (Root Mean Square Error)**
- Penalizes large forecasting errors (e.g., missing a sudden drop in oxygen concentration)

---

## Project Setup

### Prerequisites

- **Python:** 3.8 or higher
- **Core Libraries:** pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, joblib
- **Forecasting:** prophet
- **Optional:** pmdarima (for advanced ARIMA configurations)

### Directory Structure

```
.
├── data/
│   └── oxygen.csv                 # Input file
├── output/
│   ├── features/                  # Cached extracted features
│   ├── models/                    # Saved .pkl models
│   └── visualizations/            # PNG plots
├── data_preprocessing.py
├── feature_extraction.py
├── model_training.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Installation

```bash
# Clone or download the repository
cd oxygen-anomaly-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

---

## Running Configurations & Scripts

### Step 1: Training Pipeline (main.py)

#### Basic Usage

Run the complete training pipeline on your oxygen sensor data:

```bash
python main.py data/oxygen_sensor_data.csv
```

This will:
1. Load and validate the CSV file
2. Preprocess the data (handle timestamps, create entity IDs, train/test split)
3. Extract anomaly-detection-specific features from time series
4. Train a global Isolation Forest model for anomaly detection
5. Train Prophet + ARIMA ensemble for forecasting
6. Evaluate models and generate visualizations
7. Save all models and results to `./outputs`

#### Advanced Configuration

```bash
python main.py data/oxygen_sensor_data.csv \
    --output ./production_models \
    --contamination 0.10 \
    --test-days 14 \
    --max-gap-minutes 5 \
    --window-size 120 \
    --window-overlap 118 \
    --imputation ensemble \
    --log-file ./training.log
```

#### Command-Line Arguments

**Required Arguments**
- `csv_file` - Path to CSV file with oxygen sensor data

**Output Configuration**
- `--output, -o` - Output directory for models/visualizations (default: `./outputs`)
- `--log-file` - Path to log file for detailed execution logs (optional)

**Preprocessing Parameters**
- `--test-days` - Number of days for test split (default: 7)
- `--max-gap-minutes` - Maximum allowed gap in data continuity (default: 10)

**Feature Extraction Parameters**
- `--window-size` - Sliding window size in minutes (default: 60)
- `--window-overlap` - Window overlap in minutes (default: 59, creates 1-minute slides)
- `--imputation` - Method for imputing missing values:
  - `ensemble` - Combine KNN and spline (default, most robust)
  - `knn` - K-nearest neighbors
  - `spline` - Cubic spline interpolation
  - `forward_fill` - Simple forward fill

**Model Training Parameters**
- `--contamination, -c` - Contamination parameter for Isolation Forest (default: 0.07)
  - Lower values (0.01-0.05): More conservative, fewer false positives
  - Higher values (0.10-0.20): More aggressive, catches more anomalies

#### Input CSV Format

Your CSV file must contain these columns:

```csv
time,EquipmentUnit,System,Oxygen[%sat]
2025-01-15T10:00:00Z,Unit001,Sys_A,97.5
2025-01-15T10:01:00Z,Unit001,Sys_A,97.4
2025-01-15T10:02:00Z,Unit001,Sys_A,97.6
```

**Column Requirements:**
- `time` - ISO8601 or mixed format timestamps (supports nanosecond precision)
- `EquipmentUnit` - Equipment identifier
- `System` - System identifier
- `Oxygen[%sat]` - Oxygen saturation percentage (numeric)

#### Pipeline Execution Flow

```
┌─────────────────────────────┐
│    Load CSV File            │
│    Validate Data            │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│  STEP 1: PREPROCESSING      │
│  • Parse timestamps         │
│  • Create entity IDs        │
│  • Segment by entity        │
│  • Handle missing values    │
│  • Train/test split         │
│  • Create synthetic labels  │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│  STEP 2: FEATURE EXTRACT    │
│  • Sliding windows          │
│  • 40+ anomaly features     │
│  • Point anomalies          │
│  • Collective anomalies     │
│  • Contextual anomalies     │
│  • Sensor faults            │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│ STEP 3: ANOMALY DETECT      │
│ • Train Isolation Forest    │
│ • Global model (all data)   │
│ • Evaluate & metrics        │
│ • Visualizations            │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│ STEP 4: FORECASTING         │
│ • Train Prophet             │
│ • Train ARIMA               │
│ • Ensemble forecast         │
│ • Evaluate metrics          │
└─────────────────────────────┘
```

#### Output Files and Directories

After training, the `outputs/` directory contains:

**Models** (`outputs/models/`):
- `global_isolation_forest.pkl` - Trained anomaly detection model with scaler
- `forecasting/prophet_model.pkl` - Prophet forecasting model
- `forecasting/arima_model.pkl` - ARIMA forecasting model

**Features** (`outputs/features/`):
- `{entity_id}_train_features.csv` - Training features for each entity
- `{entity_id}_test_features.csv` - Test features with synthetic labels

**Visualizations** (`outputs/visualizations/`):
- `confusion_matrix.png` - Anomaly detection confusion matrix
- `roc_curve.png` - ROC curve for model evaluation
- `anomaly_scores.png` - Anomaly score distributions
- `forecast.png` - Forecast visualization

**Data** (`outputs/processed_data/`):
- `entity_{id}_clean.csv` - Preprocessed data for each entity

**Logs** (`outputs/pipeline.log`):
- Complete execution logs with debug information

---

### Step 2: Inference Pipeline (inference.py)

#### Basic Usage

Run inference on new data using trained models:

```bash
python inference.py \
    --model-dir ./outputs/models \
    --data data/new_oxygen_data.csv \
    --output ./predictions
```

This will:
1. Load trained models
2. Preprocess new data with same pipeline
3. Extract features from new data
4. Run anomaly detection
5. Generate forecasts
6. Save predictions with severity scores
7. Create visualizations

#### Advanced Usage

```bash
python inference.py \
    --model-dir ./outputs/models \
    --data data/new_data.csv \
    --output ./my_predictions \
    --features-dir ./outputs/features \
    --log-file ./inference.log
```

#### Command-Line Arguments

**Required Arguments**
- `--model-dir, -m` - Directory containing trained models
- `--data, -d` - Path to CSV file with new oxygen sensor data

**Optional Arguments**
- `--output, -o` - Output directory for predictions (default: `./predictions`)
- `--features-dir` - Directory with feature extraction config (optional)
- `--log-file` - Path to log file (optional)

#### Inference Output

The inference process generates:

**Predictions** (`predictions/anomaly_predictions.csv`):
```csv
window_start,window_center,entity_id,anomaly_prediction,anomaly_score,severity_score
2025-01-20T10:00:00Z,2025-01-20T10:00:00Z,Unit001_Sys_A,0,-0.123,25
2025-01-20T10:01:00Z,2025-01-20T10:01:00Z,Unit001_Sys_A,1,0.854,92
```

**Forecast** (`predictions/forecast.csv`):
```csv
timestamp,forecast_point,forecast_lower,forecast_upper
2025-01-20T10:00:00Z,97.5,95.2,99.8
2025-01-20T10:01:00Z,97.4,95.1,99.7
```

**Visualization** (`predictions/predictions_visualization.png`):
- Anomaly scores over time with detected anomalies highlighted
- Severity score distribution

---

## Anomaly Type Detection

The system detects four types of anomalies:

### 1. Point Anomalies (Outliers)

Single isolated data points that deviate significantly from normal
- Detected using Z-score and IQR methods
- Example: Single spike in oxygen saturation

### 2. Collective Anomalies (Pattern/Sequence)

Multiple consecutive points forming an unusual pattern
- Detected using rate-of-change and trend analysis
- Examples: Sustained high or low oxygen levels, unusual monotonic trends

### 3. Contextual Anomalies

Values that are outliers in a specific context
- Detected using rolling statistics and local outlier factors
- Example: Normal value in unusual pattern context

### 4. Sensor Fault Anomalies

Indicates equipment malfunction
- Stuck sensor (constant values)
- High noise (rapid fluctuations)
- Stuck zeros/ones (sensor fails)

---

## Hyperparameter Tuning

### Contamination Parameter (Anomaly Detection)

Controls the expected fraction of anomalies in data:

```bash
# Conservative (fewer false positives)
python main.py data/oxygen.csv --contamination 0.01

# Balanced (default)
python main.py data/oxygen.csv --contamination 0.07

# Aggressive (more sensitivity)
python main.py data/oxygen.csv --contamination 0.15
```

### Window Size and Overlap (Feature Extraction)

Controls temporal resolution of anomaly detection:

```bash
# Fast anomaly detection (larger window)
python main.py data/oxygen.csv --window-size 120 --window-overlap 118

# Balanced (default)
python main.py data/oxygen.csv --window-size 60 --window-overlap 59

# Sensitive detection (smaller window)
python main.py data/oxygen.csv --window-size 30 --window-overlap 28
```

### Imputation Method (Missing Values)

Choose strategy for handling missing data:

```bash
# Ensemble: Combine KNN + spline (most robust, recommended)
python main.py data/oxygen.csv --imputation ensemble

# KNN: K-nearest neighbors (good for sparse data)
python main.py data/oxygen.csv --imputation knn

# Spline: Cubic spline (good for smooth trends)
python main.py data/oxygen.csv --imputation spline

# Forward Fill: Simple forward fill (fast but may miss patterns)
python main.py data/oxygen.csv --imputation forward_fill
```

---

## Example Workflows

### Example 1: Quick Test

```bash
# Train on sample data
python main.py data/sample_oxygen.csv --output ./test_output

# Make predictions on new data
python inference.py \
    --model-dir ./test_output/models \
    --data data/new_sample.csv \
    --output ./test_predictions
```

### Example 2: Production Deployment

```bash
# Train with conservative parameters (fewer false positives)
python main.py data/production_data.csv \
    --output ./production_models \
    --contamination 0.05 \
    --test-days 30 \
    --imputation ensemble \
    --log-file ./production_training.log

# Use trained models for continuous inference
python inference.py \
    --model-dir ./production_models/models \
    --data data/daily_readings.csv \
    --output ./daily_predictions
```

### Example 3: Tuning for High Sensitivity

```bash
# Train with aggressive anomaly detection
python main.py data/oxygen_data.csv \
    --output ./sensitive_models \
    --contamination 0.15 \
    --window-size 30 \
    --window-overlap 28

# Inference with sensitive model
python inference.py \
    --model-dir ./sensitive_models/models \
    --data data/test_data.csv
```

---

## Troubleshooting

### Issue: "CSV file not found"

**Solution:** Verify the file path is correct and file exists
```bash
ls -la data/oxygen_data.csv
```

### Issue: "No training features available"

**Solution:** Check that CSV has required columns
```bash
python -c "import pandas as pd; df = pd.read_csv('data/oxygen.csv'); print(df.columns)"
```

### Issue: "Timestamp parsing failed"

**Solution:** Ensure timestamps are in ISO8601 or similar parseable format
```bash
# Example valid formats:
# 2025-01-15T10:00:00Z           (ISO8601 with Z)
# 2025-01-15T10:00:00+00:00      (ISO8601 with offset)
# 2025-01-15 10:00:00            (Space-separated)
```

### Issue: "Memory error during feature extraction"

**Solution:** Increase window size or use fewer entities
```bash
python main.py data/oxygen.csv --window-size 120 --window-overlap 118
```

### Issue: "Models not loading in inference"

**Solution:** Verify models directory structure
```bash
ls -la ./outputs/models/
ls -la ./outputs/models/forecasting/
```

---

## Performance Considerations

### Typical Performance Metrics

- **Training Time:** ~5-30 minutes depending on data size and hardware
- **Memory:** ~2-4GB for 100K+ records
- **Inference Time:** ~30 seconds for 1000 records on CPU

### Optimization Tips

- Larger windows (120-180 min) reduce computation but lose temporal detail
- Ensemble imputation is most robust but slower than KNN
- GPU acceleration available through TensorFlow (if installed)
- Feature caching speeds up pipeline on subsequent runs

---

## Input Data Format

The system expects a CSV file with the following structure:

```csv
timestamp,equipment_id,system_id,oxygen_saturation
2024-01-01T00:00:00Z,EQ001,SYS_A,95.4
2024-01-01T00:01:00Z,EQ001,SYS_A,95.3
2024-01-01T00:02:00Z,EQ001,SYS_A,95.2
```

| Column | Type | Description |
|--------|------|-------------|
| timestamp | String | ISO8601 format (supports nanosecond precision) |
| equipment_id | String | Unique identifier for the equipment unit |
| system_id | String | System classification or category |
| oxygen_saturation | Float | Numeric sensor reading |

---

## Limitations & Improvements

### Current Limitations

#### Cold Start Problem

- Prophet requires at least a few days of data to establish accurate seasonality
- New equipment will have higher forecast error rates initially
- **Mitigation:** Use transfer learning from similar equipment during early deployment

#### Compute Intensity

- Sliding window feature extraction (computing autocorrelations and LOF for every minute) is CPU intensive
- Feature calculation is O(n × window_size) for n data points
- **Mitigation:** Parallelize feature extraction or cache features on historical data

#### Global Threshold

- While the model is global, a fixed contamination rate (0.07) assumes all equipment has similar fault rates
- May require tuning per customer or equipment class
- **Mitigation:** Implement dynamic thresholding (see improvements below)

### Proposed Improvements

#### Deep Learning (LSTM/Transformer)

- Replace Feature Engineering + Isolation Forest with an LSTM-Autoencoder
- The reconstruction error serves as the anomaly score
- Automatically learns temporal dependencies without manual feature engineering
- **Advantages:** Better capture of complex sequential patterns, reduced feature engineering effort

#### Dynamic Thresholding

- Instead of fixed cut-off (Score < 0), use Extreme Value Theory (EVT) on anomaly scores
- Dynamically set thresholds per equipment unit based on their historical distribution
- Adapts to equipment aging and changing operational patterns

#### Real-time Streaming

- Convert `feature_extraction.py` to use a stateful stream processor (Apache Flink or Kafka Streams)
- Calculate rolling windows incrementally rather than re-computing the entire window every minute
- Enables sub-second latency for critical applications
- Reduces memory footprint in production environments

#### Advanced Ensemble Methods

- Incorporate additional forecasting algorithms (XGBoost, LightGBM)
- Implement stacking or blending for improved accuracy
- Use automatic model selection based on equipment characteristics

#### Confidence-Based Alerting

- Combine anomaly scores with forecast confidence intervals
- Generate context-aware alerts based on severity and uncertainty
- Reduce alert fatigue while maintaining detection sensitivity

