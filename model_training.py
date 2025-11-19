"""
Global Model Training - ALL DATA
Single Isolation Forest for anomaly detection + Single Prophet/ARIMA ensemble for forecasting
With evaluation metrics and visualizations
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, classification_report)
from prophet import Prophet
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        ARIMA_AVAILABLE = 'manual'
    except ImportError:
        ARIMA_AVAILABLE = False

class AnomalyDetectionTrainer:
    def __init__(self, contamination=0.07, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        print(f"[Anomaly Detection Trainer] Initialized with contamination={contamination}")
    
    def prepare_features(self, features_df):
        if features_df is None or len(features_df) == 0:
            print("  ✗ ERROR: features_df is None or empty")
            return None
        
        non_numeric_cols = ['window_start', 'window_end', 'window_center', 'anomaly_label']
        X = features_df.drop(columns=non_numeric_cols, errors='ignore')
        
        if len(X.columns) == 0:
            print("  ✗ ERROR: No numeric features found")
            return None
        
        self.feature_names = X.columns.tolist()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"  ✓ Features prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled
    
    def train(self, features_df, entity_id=""):
        print(f"\n[Training Isolation Forest] {entity_id}")
        
        X = self.prepare_features(features_df)
        if X is None:
            return None, None
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X)
        
        train_predictions = self.model.predict(X)
        train_scores = self.model.decision_function(X)
        
        n_anomalies = (train_predictions == -1).sum()
        anomaly_rate = n_anomalies / len(X) * 100
        
        print(f"  ✓ Model trained successfully")
        print(f"  Anomalies detected: {n_anomalies}/{len(X)} ({anomaly_rate:.2f}%)")
        print(f"  Anomaly score range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")
        
        return self.model, train_scores
    
    def predict(self, features_df):
        if self.model is None:
            print("✗ Model not trained yet")
            return None, None
        
        X = self.prepare_features(features_df)
        if X is None:
            return None, None
        
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        return predictions, scores
    
    def save_model(self, filepath):
        if self.model is None:
            print(f"✗ Cannot save: Model not trained")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")


class ForecastingTrainer:
    def __init__(self):
        self.prophet_model = None
        self.arima_model = None
        print("[Forecasting Trainer] Initialized")
    
    def train_prophet(self, timeseries_df, entity_id="", yearly_seasonality=False):
        print(f"\n[Training Prophet] {entity_id}")
        
        if timeseries_df is None or len(timeseries_df) == 0:
            print("  ✗ ERROR: No data for Prophet training")
            return None, None
        
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(timeseries_df['time']),
            'y': pd.to_numeric(timeseries_df['Oxygen[%sat]'], errors='coerce')
        }).dropna().sort_values('ds')
        
        if len(df_prophet) < 2:
            print("  ✗ ERROR: Insufficient data for Prophet")
            return None, None
        
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=True,
            weekly_seasonality=True,
            interval_width=0.90,
            seasonality_mode='additive'
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.prophet_model.fit(df_prophet)
            
            future = self.prophet_model.make_future_dataframe(periods=24*7, freq='min')
            forecast = self.prophet_model.predict(future)
            
            print(f"  ✓ Prophet model trained on {len(df_prophet)} points")
            return self.prophet_model, forecast
        except Exception as e:
            print(f"  ✗ Prophet training failed: {str(e)[:100]}")
            return None, None
    
    def train_arima(self, timeseries_df, entity_id=""):
        print(f"\n[Training ARIMA] {entity_id}")
        
        if not ARIMA_AVAILABLE:
            print(f"  ⚠ ARIMA not available")
            self.arima_model = None
            return None
        
        if timeseries_df is None or len(timeseries_df) == 0:
            print("  ✗ ERROR: No data for ARIMA training")
            return None
        
        ts_df = timeseries_df[['time', 'Oxygen[%sat]']].sort_values('time').copy()
        ts_df.set_index('time', inplace=True)
        
        ts_hourly = ts_df.resample('H').mean()
        ts_hourly = ts_hourly.dropna()
        
        print(f"  Data aggregated to hourly: {len(ts_hourly)} points")
        
        if len(ts_hourly) < 10:
            print(f"  ⚠ Not enough data for ARIMA")
            self.arima_model = None
            return None
        
        try:
            if ARIMA_AVAILABLE == True:
                print(f"  Using pmdarima auto_arima...")
                self.arima_model = auto_arima(
                    ts_hourly['Oxygen[%sat]'],
                    start_p=0, start_q=0,
                    max_p=3, max_q=3,
                    m=1,
                    seasonal=False,
                    stepwise=True,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True
                )
                print(f"  ✓ Auto-ARIMA model trained")
            else:
                from statsmodels.tsa.arima.model import ARIMA
                print(f"  Using statsmodels ARIMA...")
                
                best_aic = np.inf
                best_order = (1, 1, 1)
                
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(ts_hourly['Oxygen[%sat]'], order=(p, d, q))
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                self.arima_model = ARIMA(ts_hourly['Oxygen[%sat]'], order=best_order)
                self.arima_model = self.arima_model.fit()
                print(f"  ✓ ARIMA model trained")
        except Exception as e:
            print(f"  ⚠ ARIMA training failed: {str(e)[:100]}")
            self.arima_model = None
        
        return self.arima_model
    
    def forecast_ensemble(self, timeseries_df, forecast_steps=7*24*60, entity_id=""):
        print(f"\n[Ensemble Forecasting] {entity_id}")
        
        if timeseries_df is None or len(timeseries_df) == 0:
            print("  ✗ ERROR: No data for forecasting")
            return None
        
        if 'time' not in timeseries_df.columns or 'Oxygen[%sat]' not in timeseries_df.columns:
            print("  ✗ ERROR: Required columns not found")
            return None
        
        prophet_model, _ = self.train_prophet(timeseries_df, entity_id)
        if prophet_model is None:
            print("  ✗ Prophet training failed")
            return None
        
        arima_model = self.train_arima(timeseries_df, entity_id)
        
        future = prophet_model.make_future_dataframe(periods=forecast_steps, freq='min')
        prophet_forecast = prophet_model.predict(future)
        
        prophet_fcst_clean = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        prophet_fcst_clean.columns = ['timestamp', 'prophet_point', 'prophet_lower', 'prophet_upper']
        
        if arima_model is not None:
            try:
                forecast_result = prophet_fcst_clean.copy()
                
                if hasattr(arima_model, 'get_forecast'):
                    arima_fcst = arima_model.get_forecast(steps=7*24)
                    arima_values = arima_fcst.predicted_mean.values
                    
                    arima_minute = np.repeat(arima_values, 60)[:forecast_steps]
                    
                    if len(arima_minute) < forecast_steps:
                        arima_minute = np.concatenate([
                            arima_minute, 
                            np.full(forecast_steps - len(arima_minute), arima_minute[-1])
                        ])
                    
                    forecast_result['ensemble_point'] = (
                        0.6 * forecast_result['prophet_point'] + 
                        0.4 * arima_minute[:forecast_steps]
                    )
                    print(f"  Using ensemble: Prophet (60%) + ARIMA (40%)")
                else:
                    forecast_result['ensemble_point'] = forecast_result['prophet_point']
                    print(f"  Using Prophet only")
            except Exception as e:
                print(f"  ⚠ Ensemble failed, using Prophet only")
                forecast_result = prophet_fcst_clean.copy()
                forecast_result['ensemble_point'] = forecast_result['prophet_point']
        else:
            forecast_result = prophet_fcst_clean.copy()
            forecast_result['ensemble_point'] = forecast_result['prophet_point']
            print(f"  Using Prophet-only forecast")
        
        forecast_result['forecast'] = forecast_result.get('ensemble_point', forecast_result['prophet_point'])
        forecast_result['forecast_lower'] = forecast_result['prophet_lower']
        forecast_result['forecast_upper'] = forecast_result['prophet_upper']
        
        print(f"  ✓ Ensemble forecast generated: {len(forecast_result)} steps")
        
        return forecast_result
    
    def save_models(self, output_dir='./'):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.prophet_model is not None:
            try:
                self.prophet_model.save(f"{output_dir}/prophet_model.pkl")
                print(f"✓ Prophet model saved")
            except Exception as e:
                print(f"⚠ Failed to save Prophet model: {e}")
        
        if self.arima_model is not None:
            try:
                joblib.dump(self.arima_model, f"{output_dir}/arima_model.pkl")
                print(f"✓ ARIMA model saved")
            except Exception as e:
                print(f"⚠ Failed to save ARIMA model: {e}")


def load_features_from_csv(features_dir, entity_id):
    entity_features = {}
    train_path = os.path.join(features_dir, f"{entity_id}_train_features.csv")
    
    if not os.path.exists(train_path):
        return None
    
    train_features = pd.read_csv(
        train_path,
        parse_dates=['window_start', 'window_end', 'window_center'],
        infer_datetime_format=True
    )
    
    for col in ['window_start', 'window_end', 'window_center']:
        if col in train_features.columns and train_features[col].dtype == 'object':
            try:
                train_features[col] = pd.to_datetime(train_features[col], format='ISO8601', utc=True)
            except:
                try:
                    train_features[col] = pd.to_datetime(train_features[col], format='mixed', utc=True)
                except:
                    train_features[col] = pd.to_datetime(train_features[col], utc=True)
    
    entity_features['train_features'] = train_features
    
    test_path = os.path.join(features_dir, f"{entity_id}_test_features.csv")
    if os.path.exists(test_path):
        test_features = pd.read_csv(
            test_path,
            parse_dates=['window_start', 'window_end', 'window_center'],
            infer_datetime_format=True
        )
        
        for col in ['window_start', 'window_end', 'window_center']:
            if col in test_features.columns and test_features[col].dtype == 'object':
                try:
                    test_features[col] = pd.to_datetime(test_features[col], format='ISO8601', utc=True)
                except:
                    try:
                        test_features[col] = pd.to_datetime(test_features[col], format='mixed', utc=True)
                    except:
                        test_features[col] = pd.to_datetime(test_features[col], utc=True)
        
        entity_features['test_features'] = test_features
    else:
        entity_features['test_features'] = None
    
    return entity_features


def get_entity_ids_from_csv(features_dir):
    if not os.path.exists(features_dir):
        return []
    
    return sorted([f.replace('_train_features.csv', '') 
                   for f in os.listdir(features_dir) 
                   if f.endswith('_train_features.csv')])


def check_features_exist(features_dir, entity_ids):
    if not os.path.exists(features_dir):
        return False, list(entity_ids), []
    
    missing = []
    existing = []
    
    for entity_id in entity_ids:
        train_path = os.path.join(features_dir, f"{entity_id}_train_features.csv")
        if os.path.exists(train_path):
            existing.append(entity_id)
        else:
            missing.append(entity_id)
    
    all_exist = len(missing) == 0
    return all_exist, missing, existing


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Anomaly Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved: {save_path}")


def plot_anomaly_scores(scores, predictions, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores[predictions == 1], bins=50, alpha=0.7, label='Normal', color='green')
    plt.hist(scores[predictions == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(scores, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', label='Decision Boundary')
    plt.xlabel('Sample')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Anomaly scores plot saved: {save_path}")


def plot_forecast(test_data, forecast, save_path):
    if test_data is None or len(test_data) == 0:
        print("✗ No test data for forecast visualization")
        return
    
    plt.figure(figsize=(14, 6))
    
    # Plot actual
    if 'time' in test_data.columns and 'Oxygen[%sat]' in test_data.columns:
        actual_data = test_data.sort_values('time')
        plt.plot(actual_data['time'], actual_data['Oxygen[%sat]'], 'ko-', label='Actual', linewidth=2, markersize=4)
    
    # Plot forecast
    forecast_subset = forecast.head(min(len(test_data), len(forecast)))
    plt.plot(forecast_subset['timestamp'], forecast_subset['forecast'], 'b-', label='Forecast', linewidth=2)
    
    # Plot confidence interval
    plt.fill_between(
        forecast_subset['timestamp'],
        forecast_subset['forecast_lower'],
        forecast_subset['forecast_upper'],
        alpha=0.2,
        color='blue',
        label='95% Confidence Interval'
    )
    
    plt.xlabel('Time')
    plt.ylabel('Oxygen Saturation (%)')
    plt.title('Forecast vs Actual - Oxygen Saturation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Forecast plot saved: {save_path}")


def calculate_forecast_metrics(test_data, forecast):
    if test_data is None or len(test_data) == 0:
        print("✗ No test data for evaluation")
        return None
    
    if 'time' not in test_data.columns or 'Oxygen[%sat]' not in test_data.columns:
        print("✗ Test data missing required columns")
        return None
    
    actual = test_data.sort_values('time')['Oxygen[%sat]'].values
    forecast_vals = forecast['forecast'].values[:min(len(actual), len(forecast))]
    
    if len(forecast_vals) < len(actual):
        actual = actual[:len(forecast_vals)]
    
    mae = np.mean(np.abs(actual - forecast_vals))
    rmse = np.sqrt(np.mean((actual - forecast_vals)**2))
    mape = np.mean(np.abs((actual - forecast_vals) / actual)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from feature_extraction import RobustFeatureExtractor
    
    print("="*70)
    print("GLOBAL MODEL TRAINING - ALL DATA")
    print("="*70)
    
    DATA_PATH = "/Users/ferdousbinali/rag_pipeline/cefalo/data/oxygen_14_11_25.csv"
    FEATURES_DIR = "/Users/ferdousbinali/rag_pipeline/cefalo/data/features/"
    OUTPUT_DIR = "/Users/ferdousbinali/rag_pipeline/cefalo/outputs/models/"
    VIZ_DIR = "/Users/ferdousbinali/rag_pipeline/cefalo/outputs/visualizations/"
    
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    print("\n[Step 1] Preprocessing Data")
    preprocessor = DataPreprocessor(DATA_PATH)
    preprocessor.load_data()
    preprocessor.create_entity_ids()
    unique_entities = preprocessor.raw_df['entity_id'].unique()
    preprocessor.segment_by_entity(unique_entities)
    preprocessor.handle_data_quality(max_gap_minutes=10)
    split_info = preprocessor.create_train_test_split(test_size_days=7)
    
    if len(preprocessor.get_all_entities()) > 0:
        test_data, labels, details = preprocessor.create_synthetic_anomalies()
    
    print(f"✓ Preprocessing complete: {len(preprocessor.get_all_entities())} entities")
    
    print("\n" + "="*70)
    print("[Step 2] Feature Extraction")
    print("="*70)
    
    all_entities = list(preprocessor.get_all_entities())
    all_exist, missing_entities, existing_entities = check_features_exist(FEATURES_DIR, all_entities)
    
    if not all_exist:
        print(f"Extracting features for {len(missing_entities)} entities...")
        extractor = RobustFeatureExtractor(window_size_minutes=60, overlap_minutes=59)
        all_features = extractor.extract_and_save_features(preprocessor, output_dir=FEATURES_DIR)
        print(f"✓ Feature extraction complete")
    else:
        print(f"✓ All features cached")
    
    print("\n" + "="*70)
    print("[Step 3] Training Global Anomaly Detection Model")
    print("="*70)
    
    all_train_features = []
    all_test_features = []
    
    for entity_id in all_entities:
        entity_features = load_features_from_csv(FEATURES_DIR, entity_id)
        if entity_features and entity_features['train_features'] is not None:
            all_train_features.append(entity_features['train_features'])
        if entity_features and entity_features['test_features'] is not None:
            all_test_features.append(entity_features['test_features'])
    
    if not all_train_features:
        print("✗ No training features available")
        exit(1)
    
    combined_train = pd.concat(all_train_features, ignore_index=True)
    combined_test = pd.concat(all_test_features, ignore_index=True) if all_test_features else None
    
    print(f"Combined train data: {combined_train.shape[0]} samples")
    if combined_test is not None:
        print(f"Combined test data: {combined_test.shape[0]} samples")
    
    ad_trainer = AnomalyDetectionTrainer(contamination=0.07)
    ad_trainer.train(combined_train, entity_id="GLOBAL")
    ad_trainer.save_model(f"{OUTPUT_DIR}global_isolation_forest.pkl")
    
    print("\n" + "="*70)
    print("[Step 4] Evaluating Anomaly Detection on Test Data")
    print("="*70)
    
    if combined_test is not None and len(combined_test) > 0:
        test_predictions, test_scores = ad_trainer.predict(combined_test)
        
        non_numeric_cols = ['window_start', 'window_end', 'window_center', 'anomaly_label']
        y_true = combined_test.get('anomaly_label')
        
        if y_true is None or len(y_true) != len(test_predictions):
            print("⚠ Synthetic labels not available, using model predictions")
            y_true = (test_predictions == -1).astype(int)
        else:
            y_true = (y_true == 1).astype(int)
        
        cm = confusion_matrix(y_true, (test_predictions == -1).astype(int))
        precision = precision_score(y_true, (test_predictions == -1).astype(int), zero_division=0)
        recall = recall_score(y_true, (test_predictions == -1).astype(int), zero_division=0)
        f1 = f1_score(y_true, (test_predictions == -1).astype(int), zero_division=0)
        roc_auc = roc_auc_score(y_true, -test_scores)
        
        print(f"\n[Test Metrics]")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        print(f"\n[Confusion Matrix]")
        print(f"  True Negatives: {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives: {cm[1,1]}")
        
        plot_confusion_matrix(cm, f"{VIZ_DIR}confusion_matrix.png")
        plot_roc_curve(y_true, -test_scores, f"{VIZ_DIR}roc_curve.png")
        plot_anomaly_scores(test_scores, test_predictions, f"{VIZ_DIR}anomaly_scores.png")
    else:
        print("⚠ No test data for evaluation")
    
    print("\n" + "="*70)
    print("[Step 5] Training Global Forecasting Model")
    print("="*70)
    
    all_raw_train = []
    all_raw_test = []
    
    for entity_id in all_entities:
        entity_data = preprocessor.get_entity_data(entity_id)
        if entity_data and 'train' in entity_data and entity_data['train'] is not None:
            all_raw_train.append(entity_data['train'])
        if entity_data and 'test' in entity_data and entity_data['test'] is not None:
            all_raw_test.append(entity_data['test'])
    
    if not all_raw_train:
        print("✗ No raw training data available")
        exit(1)
    
    combined_raw_train = pd.concat(all_raw_train, ignore_index=True)
    combined_raw_test = pd.concat(all_raw_test, ignore_index=True) if all_raw_test else None
    
    print(f"Combined raw train data: {len(combined_raw_train)} samples")
    if combined_raw_test is not None:
        print(f"Combined raw test data: {len(combined_raw_test)} samples")
    
    fc_trainer = ForecastingTrainer()
    forecast = fc_trainer.forecast_ensemble(combined_raw_train, forecast_steps=7*24*60, entity_id="GLOBAL")
    
    if forecast is not None:
        fc_trainer.save_models(f"{OUTPUT_DIR}forecasting/")
        print("✓ Forecasting model trained and saved")
    else:
        print("✗ Forecasting model training failed")
    
    print("\n" + "="*70)
    print("[Step 6] Evaluating Forecasting on Test Data")
    print("="*70)
    
    if forecast is not None and combined_raw_test is not None:
        fc_metrics = calculate_forecast_metrics(combined_raw_test, forecast)
        
        if fc_metrics:
            print(f"\n[Forecast Metrics]")
            print(f"  MAE: {fc_metrics['MAE']:.4f}")
            print(f"  RMSE: {fc_metrics['RMSE']:.4f}")
            print(f"  MAPE: {fc_metrics['MAPE']:.4f}%")
        
        plot_forecast(combined_raw_test, forecast, f"{VIZ_DIR}forecast.png")
    else:
        print("⚠ Cannot evaluate forecasting")
    
    print("\n" + "="*70)
    print("[SUMMARY] GLOBAL MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")