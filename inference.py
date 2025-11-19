"""
Inference Script for Oxygen Sensor Anomaly Detection
Makes predictions on new data using trained models

This script loads pre-trained models and runs anomaly detection and forecasting
on new oxygen sensor data without retraining.

Usage:
    # Inference on new data
    python inference.py --model-dir ./outputs/models --data data/new_data.csv
    
    # With custom output
    python inference.py --model-dir ./outputs/models --data data/new_data.csv --output ./predictions
    
    # With feature extraction from existing features
    python inference.py --features-dir ./outputs/features --model-dir ./outputs/models --data data/new_data.csv
"""

import argparse
import os
import sys
import logging
import joblib
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from feature_extraction import RobustFeatureExtractor
from data_preprocessing import DataPreprocessor
from model_training import AnomalyDetectionTrainer, ForecastingTrainer


class InferenceLogger:
    """Logger for inference execution"""
    
    def __init__(self, log_file=None):
        self.logger = logging.getLogger('AnomalyDetectionInference')
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)


class AnomalyDetectionInference:
    """Run inference using trained models on new data"""
    
    def __init__(self, model_dir, output_dir='./predictions', logger=None):
        """
        Initialize inference engine
        
        Args:
            model_dir: Directory containing trained models
            output_dir: Directory for saving predictions and visualizations
            logger: Logger instance
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.logger = logger or InferenceLogger()
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.ad_model = None
        self.prophet_model = None
        self.arima_model = None
        self.extractor = None
        
        self.logger.info(f"Inference engine initialized")
        self.logger.info(f"  Model directory: {model_dir}")
        self.logger.info(f"  Output directory: {output_dir}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        self.logger.info("\nLoading trained models...")
        
        try:
            # Load anomaly detection model
            ad_model_path = os.path.join(self.model_dir, 'global_isolation_forest.pkl')
            if os.path.exists(ad_model_path):
                model_data = joblib.load(ad_model_path)
                self.ad_model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.logger.info(f"✓ Anomaly detection model loaded")
            else:
                self.logger.warning(f"Anomaly detection model not found at {ad_model_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load anomaly detection model: {str(e)}")
            return False
        
        try:
            # Load Prophet model
            prophet_path = os.path.join(self.model_dir, 'forecasting', 'prophet_model.pkl')
            if os.path.exists(prophet_path):
                from prophet import Prophet
                self.prophet_model = joblib.load(prophet_path)
                self.logger.info(f"✓ Prophet forecasting model loaded")
            else:
                self.logger.warning(f"Prophet model not found")
        except Exception as e:
            self.logger.warning(f"Could not load Prophet model: {str(e)}")
        
        try:
            # Load ARIMA model
            arima_path = os.path.join(self.model_dir, 'forecasting', 'arima_model.pkl')
            if os.path.exists(arima_path):
                self.arima_model = joblib.load(arima_path)
                self.logger.info(f"✓ ARIMA forecasting model loaded")
            else:
                self.logger.warning(f"ARIMA model not found")
        except Exception as e:
            self.logger.warning(f"Could not load ARIMA model: {str(e)}")
        
        if self.ad_model is None:
            self.logger.error("No models could be loaded")
            return False
        
        return True
    
    def preprocess_new_data(self, csv_path):
        """
        Preprocess new data with same pipeline as training
        
        Args:
            csv_path: Path to new CSV file
            
        Returns:
            Preprocessed dataframe
        """
        self.logger.info(f"\nPreprocessing new data: {csv_path}")
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Parse timestamps
            try:
                df['time'] = pd.to_datetime(df['time'], format='ISO8601')
            except:
                try:
                    df['time'] = pd.to_datetime(df['time'], format='mixed')
                except:
                    df['time'] = pd.to_datetime(df['time'])
            
            # Create entity IDs
            if 'EquipmentUnit' in df.columns and 'System' in df.columns:
                df['entity_id'] = (
                    df['EquipmentUnit'].astype(str) + "_" + 
                    df['System'].astype(str)
                )
            else:
                df['entity_id'] = 'global'
            
            # Ensure required columns exist
            if 'Oxygen[%sat]' not in df.columns:
                self.logger.error("Column 'Oxygen[%sat]' not found in CSV")
                return None
            
            self.logger.info(f"✓ Data loaded: {len(df)} records, {df['entity_id'].nunique()} entities")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            return None
    
    def extract_features(self, df, window_size_minutes=60, overlap_minutes=59):
        """
        Extract features from new data using same settings as training
        
        Args:
            df: Input dataframe with oxygen sensor data
            window_size_minutes: Window size for feature extraction
            overlap_minutes: Overlap between windows
            
        Returns:
            Features dataframe
        """
        self.logger.info(f"\nExtracting features from new data...")
        
        try:
            self.extractor = RobustFeatureExtractor(
                window_size_minutes=window_size_minutes,
                overlap_minutes=overlap_minutes,
                imputation_method='ensemble'
            )
            
            # Extract features by entity
            all_features = []
            
            for entity_id in df['entity_id'].unique():
                entity_df = df[df['entity_id'] == entity_id].copy()
                entity_df = entity_df.sort_values('time')
                
                if len(entity_df) < window_size_minutes:
                    self.logger.warning(f"Entity {entity_id}: insufficient data ({len(entity_df)} < {window_size_minutes})")
                    continue
                
                # Extract features
                features = self.extractor.extract_features_for_timeseries(
                    entity_df['time'].values,
                    entity_df['Oxygen[%sat]'].values
                )
                
                if features is not None:
                    features['entity_id'] = entity_id
                    all_features.append(features)
            
            if not all_features:
                self.logger.error("No features extracted")
                return None
            
            combined_features = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"✓ Features extracted: {combined_features.shape[0]} windows, {combined_features.shape[1]} features")
            
            return combined_features
        
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_anomalies(self, features_df):
        """
        Run anomaly detection on features
        
        Args:
            features_df: Features dataframe
            
        Returns:
            Dataframe with predictions and anomaly scores
        """
        if self.ad_model is None:
            self.logger.error("Anomaly detection model not loaded")
            return None
        
        self.logger.info(f"\nRunning anomaly detection...")
        
        try:
            predictions, scores = self._prepare_and_predict(features_df)
            
            # Create results dataframe
            results = features_df[['window_start', 'window_center', 'window_end', 'entity_id']].copy()
            results['anomaly_prediction'] = (predictions == -1).astype(int)
            results['anomaly_score'] = scores
            results['is_anomaly'] = results['anomaly_prediction'] == 1
            
            # Severity scoring (normalize scores to 0-100)
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                results['severity_score'] = ((scores - min_score) / (max_score - min_score) * 100).astype(int)
            else:
                results['severity_score'] = 50
            
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            self.logger.info(f"✓ Anomaly detection complete")
            self.logger.info(f"  Anomalies detected: {n_anomalies}/{len(predictions)} ({anomaly_rate:.2f}%)")
            self.logger.info(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prepare_and_predict(self, features_df):
        """Prepare features and run model prediction"""
        # Select only features used in training
        X = features_df[self.feature_names].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.ad_model.predict(X_scaled)
        scores = self.ad_model.decision_function(X_scaled)
        
        return predictions, scores
    
    def forecast_timeseries(self, df, forecast_periods=7*24*60):
        """
        Generate forecast for time series using trained Prophet/ARIMA
        
        Args:
            df: Input dataframe with oxygen sensor data
            forecast_periods: Number of periods to forecast (default: 7 days in minutes)
            
        Returns:
            Forecast dataframe
        """
        if self.prophet_model is None:
            self.logger.warning("Prophet model not available, skipping forecasting")
            return None
        
        self.logger.info(f"\nGenerating forecasts...")
        
        try:
            # Prepare time series
            df_sorted = df.sort_values('time').copy()
            
            df_prophet = pd.DataFrame({
                'ds': pd.to_datetime(df_sorted['time']),
                'y': pd.to_numeric(df_sorted['Oxygen[%sat]'], errors='coerce')
            }).dropna()
            
            if len(df_prophet) < 2:
                self.logger.warning("Insufficient data for forecasting")
                return None
            
            # Generate forecast
            future = self.prophet_model.make_future_dataframe(periods=forecast_periods, freq='min')
            forecast = self.prophet_model.predict(future)
            
            # Format output
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_result.columns = ['timestamp', 'forecast_point', 'forecast_lower', 'forecast_upper']
            
            self.logger.info(f"✓ Forecast generated: {len(forecast_result)} periods")
            
            return forecast_result
        
        except Exception as e:
            self.logger.error(f"Forecasting failed: {str(e)}")
            return None
    
    def save_predictions(self, predictions_df, forecast_df=None):
        """Save predictions to CSV"""
        try:
            predictions_df.to_csv(
                os.path.join(self.output_dir, 'anomaly_predictions.csv'),
                index=False
            )
            self.logger.info(f"✓ Predictions saved to anomaly_predictions.csv")
            
            if forecast_df is not None:
                forecast_df.to_csv(
                    os.path.join(self.output_dir, 'forecast.csv'),
                    index=False
                )
                self.logger.info(f"✓ Forecast saved to forecast.csv")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {str(e)}")
            return False
    
    def visualize_predictions(self, predictions_df, df_original=None):
        """Generate visualization of anomaly predictions"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot 1: Anomaly scores over time
            ax = axes[0]
            ax.plot(predictions_df['window_center'], predictions_df['anomaly_score'], 
                   'b-', alpha=0.7, linewidth=1)
            ax.scatter(
                predictions_df[predictions_df['is_anomaly']]['window_center'],
                predictions_df[predictions_df['is_anomaly']]['anomaly_score'],
                color='red', s=100, marker='o', label='Anomalies', zorder=5
            )
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Scores Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Severity distribution
            ax = axes[1]
            severity_normal = predictions_df[~predictions_df['is_anomaly']]['severity_score']
            severity_anomaly = predictions_df[predictions_df['is_anomaly']]['severity_score']
            
            ax.hist(severity_normal, bins=30, alpha=0.7, label='Normal', color='green')
            if len(severity_anomaly) > 0:
                ax.hist(severity_anomaly, bins=30, alpha=0.7, label='Anomaly', color='red')
            
            ax.set_xlabel('Severity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Severity Scores')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'predictions_visualization.png'), dpi=300)
            plt.close()
            
            self.logger.info(f"✓ Visualization saved to predictions_visualization.png")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
    
    def run_inference(self, csv_path):
        """
        Execute complete inference pipeline
        
        Args:
            csv_path: Path to new data CSV
            
        Returns:
            Tuple of (predictions_df, forecast_df)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("ANOMALY DETECTION INFERENCE")
        self.logger.info("="*80)
        
        # Load models
        if not self.load_models():
            return None, None
        
        # Preprocess data
        df = self.preprocess_new_data(csv_path)
        if df is None:
            return None, None
        
        # Extract features
        features = self.extract_features(df)
        if features is None:
            return None, None
        
        # Predict anomalies
        predictions = self.predict_anomalies(features)
        if predictions is None:
            return None, None
        
        # Generate forecast
        forecast = self.forecast_timeseries(df)
        
        # Save results
        self.save_predictions(predictions, forecast)
        
        # Visualize
        self.visualize_predictions(predictions, df)
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("INFERENCE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"\nResults saved to: {self.output_dir}")
        
        return predictions, forecast


def main():
    """Main entry point for inference"""
    parser = argparse.ArgumentParser(
        description='Anomaly Detection Inference - Run predictions on new data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Basic inference
  python inference.py --model-dir ./outputs/models --data data/new_data.csv
  
  # With custom output directory
  python inference.py --model-dir ./outputs/models --data data/new_data.csv --output ./my_predictions
  
  # With feature extraction directory
  python inference.py \\
    --model-dir ./outputs/models \\
    --features-dir ./outputs/features \\
    --data data/new_data.csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model-dir', '-m',
        required=True,
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to CSV file with new oxygen sensor data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        default='./predictions',
        help='Output directory for predictions (default: ./predictions)'
    )
    
    parser.add_argument(
        '--features-dir',
        help='Directory with feature extraction configuration (optional)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_dir):
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)
    
    # Initialize logger
    log_file = args.log_file or os.path.join(args.output, 'inference.log')
    logger = InferenceLogger(log_file=log_file)
    
    # Run inference
    inference_engine = AnomalyDetectionInference(
        model_dir=args.model_dir,
        output_dir=args.output,
        logger=logger
    )
    
    predictions, forecast = inference_engine.run_inference(args.data)
    
    sys.exit(0 if predictions is not None else 1)


if __name__ == "__main__":
    main()