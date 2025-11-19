# Oxygen Sensor Anomaly Detection & Forecasting System


## Table of Contents

- [Overview](#overview)
- [Solution Architecture](#solution-architecture)
  - [Anomaly Detection](#anomaly-detection)
  - [Forecasting](#forecasting)
- [Addressing Anomaly Types](#addressing-anomaly-types)
- [System Design & Lifecycle](#system-design--lifecycle)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Project Setup](#project-setup)
- [Execution Guide](#execution-guide)
- [Limitations & Improvements](#limitations--improvements)

---

## Overview

This solution implements a **Tag-Agnostic Global Model** approach. Instead of training individual models for every specific equipment unit (which scales poorly), we extract robust statistical features from time-series windows and train a single, powerful anomaly detector that generalizes across different equipment and customers.

### Key Features

- **Unsupervised Anomaly Detection** using Isolation Forests on engineered time-series features
- **Multi-Algorithm Forecasting** combining Prophet and ARIMA for robust predictions
- **Comprehensive Anomaly Classification** detecting point, collective, contextual, and sensor fault anomalies
- **Severity Scoring** with actionable insights based on anomaly magnitude
- **Enterprise-Grade Preprocessing** with intelligent handling of missing data and sensor gaps
- **Automated Evaluation** with synthetic anomaly injection for validation

---

## Solution Architecture

### Anomaly Detection

**Algorithm:** Isolation Forest (Unsupervised)

Isolation Forests are highly effective at identifying anomalies in high-dimensional feature spaces without requiring labeled training data. They are computationally efficient and handle "swamping" and "masking" better than distance-based methods.

**Input:** A vector of 40+ engineered features per 60-minute window

**Output:** Anomaly scores where scores < 0 indicate anomalies, with magnitude representing severity

### Forecasting

**Algorithm:** Ensemble (Prophet + ARIMA)

**Prophet** captures seasonality (daily/weekly cycles of oxygen levels) and naturally handles missing data and outliers.

**ARIMA** models short-term residual dependencies that Prophet might miss.

**Strategy:** A weighted ensemble (60% Prophet / 40% ARIMA) ensures the model captures both macro trends and micro dynamics.

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
└── README.md
```

### Installation

```bash
# Clone or download the repository
cd oxygen-anomaly-detection

# Install dependencies
pip install pandas numpy scikit-learn prophet scipy matplotlib seaborn joblib

# Optional: Install pmdarima for enhanced ARIMA support
pip install pmdarima
```

---

## Execution Guide

### Quick Start

1. **Configure Data Path**
   
   Open `model_training.py` and update the `DATA_PATH` variable to point to your input CSV file.

2. **Run the Pipeline**

   ```bash
   python model_training.py
   ```

   This single command triggers the entire workflow:

   - Preprocessing & Synthetic Data Injection
   - Feature Extraction (cached to CSV for reproducibility)
   - Global Isolation Forest Training
   - Ensemble Forecasting (Prophet + ARIMA)
   - Evaluation & Visualization Generation

3. **Review Results**

   Check the `output/` directory for:
   - `models/` — Serialized Isolation Forest and Prophet models
   - `features/` — Extracted feature cache for auditing
   - `visualizations/` — Performance plots and diagnostic charts

### Input Data Format

The system expects a CSV with the following structure:

```
timestamp,equipment_id,system_id,oxygen_saturation
2024-01-01T00:00:00Z,EQ001,SYS_A,95.4
2024-01-01T00:01:00Z,EQ001,SYS_A,95.3
...
```

- **timestamp:** ISO8601 format (supports nanosecond precision)
- **equipment_id:** Unique identifier for the equipment unit
- **system_id:** System classification or category
- **oxygen_saturation:** Numeric sensor reading

---

## Limitations & Improvements

### Current Limitations

**Cold Start Problem**
- Prophet requires at least a few days of data to establish accurate seasonality
- New equipment will have higher forecast error rates initially
- Mitigation: Use transfer learning from similar equipment during early deployment

**Compute Intensity**
- Sliding window feature extraction (computing autocorrelations and LOF for every minute) is CPU intensive
- Feature calculation is O(n × window_size) for n data points
- Mitigation: Parallelize feature extraction or cache features on historical data

**Global Threshold**
- While the model is global, a fixed contamination rate (0.07) assumes all equipment has similar fault rates
- May require tuning per customer or equipment class
- Mitigation: Implement dynamic thresholding (see improvements below)

### Proposed Improvements

**Deep Learning (LSTM/Transformer)**
- Replace Feature Engineering + Isolation Forest with an LSTM-Autoencoder
- The reconstruction error serves as the anomaly score
- Automatically learns temporal dependencies without manual feature engineering
- Advantages: Better capture of complex sequential patterns, reduced feature engineering effort

**Dynamic Thresholding**
- Instead of fixed cut-off (Score < 0), use Extreme Value Theory (EVT) on anomaly scores
- Dynamically set thresholds per equipment unit based on their historical distribution
- Adapts to equipment aging and changing operational patterns

**Real-time Streaming**
- Convert `feature_extraction.py` to use a stateful stream processor (Apache Flink or Kafka Streams)
- Calculate rolling windows incrementally rather than re-computing the entire window every minute
- Enables sub-second latency for critical applications
- Reduces memory footprint in production environments

**Advanced Ensemble Methods**
- Incorporate additional forecasting algorithms (XGBoost, LightGBM)
- Implement stacking or blending for improved accuracy
- Use automatic model selection based on equipment characteristics

**Confidence-Based Alerting**
- Combine anomaly scores with forecast confidence intervals
- Generate context-aware alerts based on severity and uncertainty
- Reduce alert fatigue while maintaining detection sensitivity


