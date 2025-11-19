"""
Main Orchestration Script for Oxygen Sensor Anomaly Detection System
Complete pipeline: Data Preprocessing → Feature Extraction → Model Training → Evaluation

This script handles the end-to-end workflow for building production-ready anomaly detection
and forecasting models from raw oxygen sensor data.

Usage:
    # Basic usage
    python main.py data/oxygen_data.csv
    
    # With custom output directory
    python main.py data/oxygen_data.csv --output ./my_outputs
    
    # With custom hyperparameters
    python main.py data/oxygen_data.csv --contamination 0.1 --test-days 14 --window-size 120
"""

import argparse
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Import pipeline modules
from data_preprocessing import DataPreprocessor
from feature_extraction import RobustFeatureExtractor
from model_training import (
    AnomalyDetectionTrainer,
    ForecastingTrainer,
    load_features_from_csv,
    check_features_exist,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_anomaly_scores,
    plot_forecast,
    calculate_forecast_metrics
)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


class PipelineLogger:
    """Custom logger for pipeline execution with both console and file output"""
    
    def __init__(self, log_file=None):
        self.logger = logging.getLogger('AnomalyDetectionPipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (if log file specified)
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


class AnomalyDetectionPipeline:
    """Main orchestration class for complete ML pipeline"""
    
    def __init__(self, csv_path, output_dir='./outputs', logger=None):
        """
        Initialize pipeline with input data and output configuration
        
        Args:
            csv_path: Path to CSV file with oxygen sensor data
            output_dir: Base output directory for models and visualizations
            logger: Logger instance for tracking execution
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.logger = logger or PipelineLogger()
        
        # Create directory structure
        self.models_dir = os.path.join(output_dir, 'models')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.features_dir = os.path.join(output_dir, 'features')
        self.data_dir = os.path.join(output_dir, 'processed_data')
        
        for directory in [self.models_dir, self.viz_dir, self.features_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Pipeline initialized")
        self.logger.info(f"  Input CSV: {csv_path}")
        self.logger.info(f"  Output directory: {output_dir}")
        
        # Pipeline components
        self.preprocessor = None
        self.extractor = None
        self.ad_trainer = None
        self.fc_trainer = None
        self.all_entities = []
    
    def validate_input(self):
        """Validate that input CSV file exists and is readable"""
        if not os.path.exists(self.csv_path):
            self.logger.error(f"CSV file not found: {self.csv_path}")
            return False
        
        if not os.path.isfile(self.csv_path):
            self.logger.error(f"Path is not a file: {self.csv_path}")
            return False
        
        self.logger.info(f"✓ Input file validated: {self.csv_path}")
        return True
    
    def run_preprocessing(self, test_size_days=7, max_gap_minutes=10, create_synthetic_anomalies=True):
        """
        Step 1: Data Preprocessing
        - Load raw data from CSV
        - Parse timestamps with nanosecond precision
        - Create entity IDs for tag-agnostic routing
        - Segment data by entity
        - Handle missing values and data quality
        - Create train/test splits
        - Optionally create synthetic anomalies for validation
        
        Args:
            test_size_days: Number of days for test split
            max_gap_minutes: Maximum allowed gap in minutes
            create_synthetic_anomalies: Whether to inject synthetic anomalies
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 1: DATA PREPROCESSING")
        self.logger.info("="*80)
        
        try:
            self.preprocessor = DataPreprocessor(self.csv_path)
            
            # Load and validate data
            self.preprocessor.load_data()
            
            # Create entity IDs for tag-agnostic routing
            self.preprocessor.create_entity_ids()
            unique_entities = self.preprocessor.raw_df['entity_id'].unique()
            
            # Segment data by entity
            self.preprocessor.segment_by_entity(unique_entities)
            
            # Handle data quality issues
            self.preprocessor.handle_data_quality(max_gap_minutes=max_gap_minutes)
            
            # Create train/test split
            self.preprocessor.create_train_test_split(test_size_days=test_size_days)
            
            # Create synthetic anomalies for validation
            if create_synthetic_anomalies and len(self.preprocessor.get_all_entities()) > 0:
                self.preprocessor.create_synthetic_anomalies()
                self.logger.info("✓ Synthetic anomalies injected for validation")
            
            # Save preprocessed data
            self.preprocessor.save_preprocessed_data(self.data_dir)
            
            self.all_entities = self.preprocessor.get_all_entities()
            self.logger.info(f"✓ Preprocessing complete: {len(self.all_entities)} entities processed")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_feature_extraction(self, window_size_minutes=60, overlap_minutes=59, 
                             imputation_method='ensemble', force_recompute=False):
        """
        Step 2: Feature Extraction
        - Create sliding windows over time series
        - Extract 40+ anomaly-detection-specific features
        - Handle missing values with ensemble imputation
        - Detect four types of anomalies: point, collective, contextual, sensor fault
        
        Args:
            window_size_minutes: Size of sliding window
            overlap_minutes: Overlap between windows
            imputation_method: Method for imputing missing values (ensemble/knn/spline/forward_fill)
            force_recompute: Force recomputation even if features exist
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 2: FEATURE EXTRACTION")
        self.logger.info("="*80)
        
        try:
            # Check if features already exist
            all_exist, missing_entities, existing_entities = check_features_exist(
                self.features_dir, self.all_entities
            )
            
            if all_exist and not force_recompute:
                self.logger.info(f"✓ All features cached ({len(existing_entities)} entities)")
                return True
            
            if not force_recompute and len(existing_entities) > 0:
                self.logger.info(f"Extracting features for {len(missing_entities)} new entities...")
            else:
                self.logger.info(f"Extracting features for {len(self.all_entities)} entities...")
            
            self.extractor = RobustFeatureExtractor(
                window_size_minutes=window_size_minutes,
                overlap_minutes=overlap_minutes,
                imputation_method=imputation_method
            )
            
            all_features = self.extractor.extract_and_save_features(
                self.preprocessor,
                output_dir=self.features_dir
            )
            
            self.logger.info(f"✓ Feature extraction complete")
            return True
        
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_anomaly_detection_training(self, contamination=0.07):
        """
        Step 3: Train Anomaly Detection Model
        - Load all extracted features
        - Combine data from all entities (global model)
        - Train Isolation Forest with contamination parameter
        - Evaluate on test data with synthetic labels
        - Generate evaluation metrics and visualizations
        
        Args:
            contamination: Contamination parameter for Isolation Forest
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 3: ANOMALY DETECTION MODEL TRAINING")
        self.logger.info("="*80)
        
        try:
            # Load all features
            all_train_features = []
            all_test_features = []
            
            for entity_id in self.all_entities:
                entity_features = load_features_from_csv(self.features_dir, entity_id)
                if entity_features and entity_features['train_features'] is not None:
                    all_train_features.append(entity_features['train_features'])
                if entity_features and entity_features['test_features'] is not None:
                    all_test_features.append(entity_features['test_features'])
            
            if not all_train_features:
                self.logger.error("No training features available")
                return False
            
            combined_train = pd.concat(all_train_features, ignore_index=True)
            combined_test = pd.concat(all_test_features, ignore_index=True) if all_test_features else None
            
            self.logger.info(f"Combined train data: {combined_train.shape[0]} samples, {combined_train.shape[1]} features")
            if combined_test is not None:
                self.logger.info(f"Combined test data: {combined_test.shape[0]} samples")
            
            # Train global anomaly detection model
            self.ad_trainer = AnomalyDetectionTrainer(contamination=contamination)
            self.ad_trainer.train(combined_train, entity_id="GLOBAL")
            self.ad_trainer.save_model(os.path.join(self.models_dir, 'global_isolation_forest.pkl'))
            
            self.logger.info(f"✓ Anomaly detection model trained and saved")
            
            # Evaluate on test data
            if combined_test is not None and len(combined_test) > 0:
                self._evaluate_anomaly_detection(combined_test)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Anomaly detection training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _evaluate_anomaly_detection(self, test_features):
        """Evaluate anomaly detection model and generate visualizations"""
        self.logger.info("\n" + "-"*80)
        self.logger.info("EVALUATING ANOMALY DETECTION")
        self.logger.info("-"*80)
        
        try:
            test_predictions, test_scores = self.ad_trainer.predict(test_features)
            
            y_true = test_features.get('anomaly_label')
            
            if y_true is None or len(y_true) != len(test_predictions):
                self.logger.warning("Synthetic labels not available for evaluation")
                return
            
            y_true = (y_true == 1).astype(int)
            y_pred = (test_predictions == -1).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, -test_scores)
            
            self.logger.info(f"\n[Test Metrics]")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1-Score: {f1:.4f}")
            self.logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            
            self.logger.info(f"\n[Confusion Matrix]")
            self.logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
            self.logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            # Generate visualizations
            plot_confusion_matrix(cm, os.path.join(self.viz_dir, 'confusion_matrix.png'))
            plot_roc_curve(y_true, -test_scores, os.path.join(self.viz_dir, 'roc_curve.png'))
            plot_anomaly_scores(test_scores, test_predictions, os.path.join(self.viz_dir, 'anomaly_scores.png'))
            
            self.logger.info(f"✓ Evaluation visualizations saved")
        
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
    
    def run_forecasting_training(self):
        """
        Step 4: Train Forecasting Model
        - Load raw time series data for all entities
        - Train Prophet model with seasonality
        - Train ARIMA model for ensemble
        - Create ensemble forecast (Prophet 60% + ARIMA 40%)
        - Generate forecast visualization
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: FORECASTING MODEL TRAINING")
        self.logger.info("="*80)
        
        try:
            # Load raw time series data
            all_raw_train = []
            all_raw_test = []
            
            for entity_id in self.all_entities:
                entity_data = self.preprocessor.get_entity_data(entity_id)
                if entity_data and 'train' in entity_data and entity_data['train'] is not None:
                    all_raw_train.append(entity_data['train'])
                if entity_data and 'test' in entity_data and entity_data['test'] is not None:
                    all_raw_test.append(entity_data['test'])
            
            if not all_raw_train:
                self.logger.error("No raw training data available")
                return False
            
            combined_raw_train = pd.concat(all_raw_train, ignore_index=True)
            combined_raw_test = pd.concat(all_raw_test, ignore_index=True) if all_raw_test else None
            
            self.logger.info(f"Combined raw train data: {len(combined_raw_train)} samples")
            if combined_raw_test is not None:
                self.logger.info(f"Combined raw test data: {len(combined_raw_test)} samples")
            
            # Train forecasting ensemble
            self.fc_trainer = ForecastingTrainer()
            forecast = self.fc_trainer.forecast_ensemble(
                combined_raw_train,
                forecast_steps=7*24*60,
                entity_id="GLOBAL"
            )
            
            if forecast is not None:
                forecasting_models_dir = os.path.join(self.models_dir, 'forecasting')
                self.fc_trainer.save_models(forecasting_models_dir)
                self.logger.info(f"✓ Forecasting model trained and saved")
            else:
                self.logger.warning("Forecasting model training failed")
                return False
            
            # Evaluate forecasting
            if forecast is not None and combined_raw_test is not None:
                self._evaluate_forecasting(combined_raw_test, forecast)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Forecasting training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _evaluate_forecasting(self, test_data, forecast):
        """Evaluate forecasting model and generate visualizations"""
        self.logger.info("\n" + "-"*80)
        self.logger.info("EVALUATING FORECASTING")
        self.logger.info("-"*80)
        
        try:
            fc_metrics = calculate_forecast_metrics(test_data, forecast)
            
            if fc_metrics:
                self.logger.info(f"\n[Forecast Metrics]")
                self.logger.info(f"  MAE: {fc_metrics['MAE']:.4f}")
                self.logger.info(f"  RMSE: {fc_metrics['RMSE']:.4f}")
                self.logger.info(f"  MAPE: {fc_metrics['MAPE']:.4f}%")
            
            plot_forecast(test_data, forecast, os.path.join(self.viz_dir, 'forecast.png'))
            self.logger.info(f"✓ Forecast visualization saved")
        
        except Exception as e:
            self.logger.error(f"Forecast evaluation failed: {str(e)}")
    
    def run_full_pipeline(self, contamination=0.07, test_size_days=7, max_gap_minutes=10,
                         window_size_minutes=60, overlap_minutes=59, imputation_method='ensemble'):
        """
        Execute the complete pipeline from raw data to trained models
        
        Args:
            contamination: Contamination parameter for anomaly detection
            test_size_days: Number of days for test split
            max_gap_minutes: Maximum allowed gap in minutes
            window_size_minutes: Size of sliding window for feature extraction
            overlap_minutes: Overlap between windows
            imputation_method: Method for imputing missing values
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        start_time = datetime.now()
        
        # Print header
        self.logger.info("\n\n")
        self.logger.info("█"*80)
        self.logger.info("█" + " "*78 + "█")
        self.logger.info("█" + "  OXYGEN SENSOR ANOMALY DETECTION PIPELINE".center(78) + "█")
        self.logger.info("█" + f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "█")
        self.logger.info("█" + " "*78 + "█")
        self.logger.info("█"*80)
        
        # Step 1: Preprocessing
        if not self.validate_input():
            return False
        
        if not self.run_preprocessing(
            test_size_days=test_size_days,
            max_gap_minutes=max_gap_minutes,
            create_synthetic_anomalies=True
        ):
            return False
        
        # Step 2: Feature Extraction
        if not self.run_feature_extraction(
            window_size_minutes=window_size_minutes,
            overlap_minutes=overlap_minutes,
            imputation_method=imputation_method
        ):
            return False
        
        # Step 3: Anomaly Detection Training
        if not self.run_anomaly_detection_training(contamination=contamination):
            return False
        
        # Step 4: Forecasting Training
        if not self.run_forecasting_training():
            self.logger.warning("Forecasting training failed, but pipeline continues...")
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info("\n" + "█"*80)
        self.logger.info("█" + " "*78 + "█")
        self.logger.info("█" + "  PIPELINE EXECUTION COMPLETE".center(78) + "█")
        self.logger.info("█" + " "*78 + "█")
        self.logger.info("█"*80)
        
        self.logger.info(f"\nExecution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        self.logger.info(f"\nOutput directories:")
        self.logger.info(f"  Models: {self.models_dir}")
        self.logger.info(f"  Visualizations: {self.viz_dir}")
        self.logger.info(f"  Features: {self.features_dir}")
        self.logger.info(f"  Data: {self.data_dir}")
        
        return True


def main():
    """Main entry point with argparse configuration"""
    parser = argparse.ArgumentParser(
        description='Oxygen Sensor Anomaly Detection - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Basic usage
  python main.py data/oxygen_data.csv
  
  # With custom output directory
  python main.py data/oxygen_data.csv --output ./my_models
  
  # With custom hyperparameters
  python main.py data/oxygen_data.csv --contamination 0.1 --test-days 14
  
  # Advanced configuration
  python main.py data/oxygen_data.csv \\
    --output ./production_models \\
    --contamination 0.05 \\
    --test-days 21 \\
    --window-size 120 \\
    --window-overlap 118 \\
    --imputation ensemble \\
    --log-file ./pipeline.log
        """
    )
    
    # Required arguments
    parser.add_argument(
        'csv_file',
        help='Path to CSV file with oxygen sensor data'
    )
    
    # Optional arguments - output configuration
    parser.add_argument(
        '--output', '-o',
        default='./outputs',
        help='Output directory for models, visualizations, and features (default: ./outputs)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path for detailed execution logs (optional)'
    )
    
    # Preprocessing parameters
    parser.add_argument(
        '--test-days',
        type=int,
        default=7,
        help='Number of days for test split (default: 7)'
    )
    
    parser.add_argument(
        '--max-gap-minutes',
        type=int,
        default=10,
        help='Maximum allowed gap in minutes for data continuity (default: 10)'
    )
    
    # Feature extraction parameters
    parser.add_argument(
        '--window-size',
        type=int,
        default=60,
        help='Sliding window size in minutes (default: 60)'
    )
    
    parser.add_argument(
        '--window-overlap',
        type=int,
        default=59,
        help='Window overlap in minutes (default: 59, means 1-minute slide)'
    )
    
    parser.add_argument(
        '--imputation',
        choices=['ensemble', 'knn', 'spline', 'forward_fill'],
        default='ensemble',
        help='Method for imputing missing values (default: ensemble)'
    )
    
    # Model training parameters
    parser.add_argument(
        '--contamination', '-c',
        type=float,
        default=0.07,
        help='Contamination parameter for Isolation Forest (default: 0.07)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate CSV path
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Initialize logger
    log_file = args.log_file or os.path.join(args.output, 'pipeline.log')
    logger = PipelineLogger(log_file=log_file)
    
    # Initialize and run pipeline
    pipeline = AnomalyDetectionPipeline(
        csv_path=args.csv_file,
        output_dir=args.output,
        logger=logger
    )
    
    success = pipeline.run_full_pipeline(
        contamination=args.contamination,
        test_size_days=args.test_days,
        max_gap_minutes=args.max_gap_minutes,
        window_size_minutes=args.window_size,
        overlap_minutes=args.window_overlap,
        imputation_method=args.imputation
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()