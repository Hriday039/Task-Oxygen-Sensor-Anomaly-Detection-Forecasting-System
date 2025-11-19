"""
Advanced Feature Extraction Module
Extracts anomaly-detection-specific features from oxygen sensor data
with intelligent handling of missing values and multiple anomaly types
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class RobustFeatureExtractor:
    """
    Extracts features specifically designed to detect:
    - Point anomalies (outliers)
    - Collective anomalies (patterns/sequences)
    - Contextual anomalies (conditional outliers)
    - Sensor fault anomalies (stuck sensor, spikes, high noise)
    """
    
    def __init__(self, window_size_minutes=60, overlap_minutes=59, imputation_method='ensemble'):
        """
        Initialize feature extractor
        
        Args:
            window_size_minutes: Size of sliding window (default: 60 min)
            overlap_minutes: Overlap between windows (default: 59 min, 1 min slide)
            imputation_method: 'ensemble' (KNN+spline), 'knn', 'spline', or 'forward_fill'
        """
        self.window_size_minutes = window_size_minutes
        self.overlap_minutes = overlap_minutes
        self.slide_minutes = window_size_minutes - overlap_minutes
        self.imputation_method = imputation_method
        
        print("[Feature Extractor] Initialized")
        print(f"  Window size: {window_size_minutes} minutes")
        print(f"  Overlap: {overlap_minutes} minutes")
        print(f"  Slide: {self.slide_minutes} minute(s)")
        print(f"  Imputation method: {imputation_method}")
        print(f"  Features: Custom anomaly-detection suite")
    
    def _impute_missing_values(self, timestamps, oxygen_values):
        """
        Intelligently impute missing values using ensemble methods
        
        Args:
            timestamps: Time values
            oxygen_values: Oxygen saturation values with potential NaNs
            
        Returns:
            Imputed oxygen values
        """
        oxygen_values = np.array(oxygen_values, dtype=float)
        nan_mask = np.isnan(oxygen_values)
        
        # If no missing values, return as-is
        if not nan_mask.any():
            return oxygen_values
        
        if self.imputation_method == 'ensemble':
            return self._ensemble_imputation(oxygen_values, nan_mask)
        elif self.imputation_method == 'knn':
            return self._knn_imputation(oxygen_values)
        elif self.imputation_method == 'spline':
            return self._spline_imputation(oxygen_values, nan_mask)
        else:  # forward_fill
            return self._forward_fill_imputation(oxygen_values)
    
    def _ensemble_imputation(self, oxygen_values, nan_mask):
        """Combine KNN and spline imputation"""
        # Try KNN first
        try:
            knn_imputed = self._knn_imputation(oxygen_values)
        except:
            knn_imputed = oxygen_values.copy()
        
        # Try spline for remaining NaNs
        spline_imputed = self._spline_imputation(knn_imputed, np.isnan(knn_imputed))
        
        return spline_imputed
    
    def _knn_imputation(self, oxygen_values, n_neighbors=5):
        """KNN-based imputation"""
        try:
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(oxygen_values)-1))
            return imputer.fit_transform(oxygen_values.reshape(-1, 1)).flatten()
        except:
            return oxygen_values
    
    def _spline_imputation(self, oxygen_values, nan_mask):
        """Spline-based imputation"""
        oxygen_values = oxygen_values.copy()
        
        if not np.isnan(oxygen_values).any():
            return oxygen_values
        
        try:
            valid_idx = ~np.isnan(oxygen_values)
            if valid_idx.sum() < 2:
                return oxygen_values  # Not enough valid points for spline
            
            f = interp1d(
                np.where(valid_idx)[0],
                oxygen_values[valid_idx],
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            all_idx = np.arange(len(oxygen_values))
            oxygen_values[~valid_idx] = f(all_idx[~valid_idx])
            return oxygen_values
        except:
            # Fallback to forward fill
            return self._forward_fill_imputation(oxygen_values)
    
    def _forward_fill_imputation(self, oxygen_values):
        """Simple forward fill imputation"""
        oxygen_values = oxygen_values.copy()
        mask = np.isnan(oxygen_values)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        idx = np.maximum.accumulate(idx)
        return oxygen_values[idx]
    
    def extract_features_for_timeseries(self, timestamps, oxygen_values):
        """
        Extract anomaly-detection features for complete time series
        
        Args:
            timestamps: Pandas DatetimeIndex or array of timestamps
            oxygen_values: Array of oxygen saturation values
            
        Returns:
            DataFrame with windows and their extracted features
        """
        
        if len(oxygen_values) < self.window_size_minutes:
            print(f"⚠ Warning: Series length ({len(oxygen_values)}) < window size ({self.window_size_minutes})")
            return None
        
        # Convert to numpy arrays for consistent handling
        oxygen_values = np.asarray(oxygen_values, dtype=float)
        timestamps = np.asarray(timestamps)
        
        # Impute missing values before feature extraction
        print(f"  Imputing missing values...")
        nan_count_before = np.isnan(oxygen_values).sum()
        if nan_count_before > 0:
            print(f"    Found {nan_count_before} missing values, imputing with {self.imputation_method}...")
            oxygen_values = self._impute_missing_values(timestamps, oxygen_values)
            nan_count_after = np.isnan(oxygen_values).sum()
            print(f"    After imputation: {nan_count_after} NaN values remaining")
        
        features_list = []
        window_starts = []
        window_ends = []
        window_centers = []
        n_points_list = []
        mean_oxygen_list = []
        std_oxygen_list = []
        min_oxygen_list = []
        max_oxygen_list = []
        
        # Create sliding windows
        n_windows_total = 0
        n_windows_skipped = 0
        n_windows_processed = 0
        
        for start_idx in range(0, len(oxygen_values) - self.window_size_minutes + 1, self.slide_minutes):
            end_idx = start_idx + self.window_size_minutes
            n_windows_total += 1
            
            window_ts = timestamps[start_idx:end_idx]
            window_oxygen = oxygen_values[start_idx:end_idx]
            
            # Skip only if still has NaN after imputation (safety check)
            if np.isnan(window_oxygen).any():
                n_windows_skipped += 1
                continue
            
            try:
                # Extract comprehensive anomaly-detection features
                features = self._extract_anomaly_features(window_oxygen)
                
                # Ensure all features are scalar (not arrays) - convert to Python native types
                features_clean = {}
                for k, v in features.items():
                    if isinstance(v, (np.ndarray, list)):
                        features_clean[k] = float(v.flat[0])  # Get first element if array
                    elif isinstance(v, (np.floating, np.integer)):
                        features_clean[k] = float(v)  # Convert numpy scalar to Python float
                    else:
                        features_clean[k] = v
                
                features_list.append(features_clean)
                
                # Store window metadata - convert timestamps to Python objects
                try:
                    ts_start = pd.Timestamp(window_ts[0]).to_pydatetime()
                    ts_end = pd.Timestamp(window_ts[-1]).to_pydatetime()
                    ts_center = pd.Timestamp(window_ts[len(window_ts)//2]).to_pydatetime()
                except:
                    ts_start = window_ts[0]
                    ts_end = window_ts[-1]
                    ts_center = window_ts[len(window_ts)//2]
                
                window_starts.append(ts_start)
                window_ends.append(ts_end)
                window_centers.append(ts_center)
                n_points_list.append(int(len(window_oxygen)))
                mean_oxygen_list.append(float(np.mean(window_oxygen)))
                std_oxygen_list.append(float(np.std(window_oxygen)))
                min_oxygen_list.append(float(np.min(window_oxygen)))
                max_oxygen_list.append(float(np.max(window_oxygen)))
                
                n_windows_processed += 1
                
            except Exception as e:
                # Log more detailed error info on first few errors only
                if n_windows_processed < 3:
                    print(f"    ⚠ Error at window {start_idx}: {str(e)[:80]}")
                    import traceback
                    traceback.print_exc()
                continue
        
        print(f"    Processed: {n_windows_total} windows total, {n_windows_processed} successful, {n_windows_skipped} NaN skipped")
        
        if len(features_list) == 0:
            print("⚠ No valid features extracted!")
            return None
        
        # Compile features DataFrame
        try:
            features_df = pd.DataFrame(features_list)
            
            # Add window metadata as separate columns
            features_df['window_start'] = window_starts
            features_df['window_end'] = window_ends
            features_df['window_center'] = window_centers
            features_df['n_points'] = n_points_list
            features_df['mean_oxygen'] = mean_oxygen_list
            features_df['std_oxygen'] = std_oxygen_list
            features_df['min_oxygen'] = min_oxygen_list
            features_df['max_oxygen'] = max_oxygen_list
            
            # Reorder columns: metadata first
            metadata_cols = ['window_start', 'window_end', 'window_center', 
                           'n_points', 'mean_oxygen', 'std_oxygen', 'min_oxygen', 'max_oxygen']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            features_df = features_df[metadata_cols + feature_cols]
            
            return features_df
        except Exception as e:
            print(f"⚠ Error creating features DataFrame: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_anomaly_features(self, window_oxygen):
        """
        Extract comprehensive features for 4 anomaly types:
        1. Point anomalies - outliers in single points
        2. Collective anomalies - unusual patterns/sequences
        3. Contextual anomalies - values that are outliers in specific context
        4. Sensor fault anomalies - stuck sensor, spikes, high noise
        """
        
        window_oxygen = np.array(window_oxygen)
        features = {}
        
        # ===== POINT ANOMALY FEATURES =====
        # Detect single outliers
        features.update(self._point_anomaly_features(window_oxygen))
        
        # ===== COLLECTIVE ANOMALY FEATURES =====
        # Detect pattern/sequence anomalies
        features.update(self._collective_anomaly_features(window_oxygen))
        
        # ===== CONTEXTUAL ANOMALY FEATURES =====
        # Detect values that are outliers in context
        features.update(self._contextual_anomaly_features(window_oxygen))
        
        # ===== SENSOR FAULT FEATURES =====
        # Detect sensor faults: stuck sensor, spikes, noise
        features.update(self._sensor_fault_features(window_oxygen))
        
        # ===== DISTRIBUTION FEATURES =====
        # General statistical features
        features.update(self._distribution_features(window_oxygen))
        
        return features
    
    def _point_anomaly_features(self, values):
        """Features for detecting point/outlier anomalies"""
        z_scores = np.abs(stats.zscore(values))
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        
        # Safe IQR outlier detection
        lower_fence = q25 - 1.5 * iqr if iqr > 0 else q25
        upper_fence = q75 + 1.5 * iqr if iqr > 0 else q75
        
        return {
            # Z-score based outliers
            'max_zscore': float(np.max(z_scores)),
            'mean_abs_zscore': float(np.mean(z_scores)),
            'n_zscore_outliers': float(np.sum(z_scores > 3)),
            
            # IQR-based outliers
            'iqr': float(iqr),
            'n_iqr_outliers': float(np.sum((values < lower_fence) | (values > upper_fence))),
            
            # Extreme values
            'max_deviation_from_median': float(np.max(np.abs(values - np.median(values)))),
            'n_values_above_upper_fence': float(np.sum(values > np.percentile(values, 95))),
            'n_values_below_lower_fence': float(np.sum(values < np.percentile(values, 5))),
        }
    
    def _collective_anomaly_features(self, values):
        """Features for detecting collective/pattern anomalies"""
        diffs = np.diff(values)
        roc = np.abs(diffs)
        
        # Safe autocorrelation computation
        def safe_autocorr(vals, lag):
            try:
                if len(vals) <= lag:
                    return 0.0
                corr_matrix = np.corrcoef(vals[:-lag], vals[lag:])
                return float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0
            except:
                return 0.0
        
        return {
            # Rate of change patterns
            'max_roc': float(np.max(roc)) if len(roc) > 0 else 0.0,
            'mean_roc': float(np.mean(roc)) if len(roc) > 0 else 0.0,
            'std_roc': float(np.std(roc)) if len(roc) > 0 else 0.0,
            'roc_volatility': float(np.std(np.diff(roc))) if len(roc) > 1 else 0.0,
            
            # Trend features
            'trend_strength': float(np.abs(np.corrcoef(range(len(values)), values)[0, 1])),
            'n_direction_changes': float(np.sum(np.diff(np.sign(diffs)) != 0)),
            
            # Monotonicity
            'is_monotonic_increasing': float(np.all(diffs >= 0)),
            'is_monotonic_decreasing': float(np.all(diffs <= 0)),
            'consecutive_increases': float(self._max_consecutive_direction(diffs > 0)),
            'consecutive_decreases': float(self._max_consecutive_direction(diffs < 0)),
            
            # Autocorrelation
            'autocorr_lag1': safe_autocorr(values, 1),
            'autocorr_lag5': safe_autocorr(values, 5),
        }
    
    def _contextual_anomaly_features(self, values):
        """Features for detecting contextual anomalies"""
        
        # Safe rolling mean computation - keep as Series to avoid pandas type issues
        values_series = pd.Series(values)
        rolling_5 = values_series.rolling(5, center=True).mean().fillna(values_series)
        rolling_10 = values_series.rolling(10, center=True).mean().fillna(values_series)
        
        return {
            # Deviation from recent history
            'deviation_from_rolling_mean_5': float(np.max(np.abs(values - rolling_5.values))),
            'deviation_from_rolling_mean_10': float(np.max(np.abs(values - rolling_10.values))),
            
            # Local anomalies
            'max_local_outlier_factor': self._compute_local_outlier_factor(values),
            
            # Conditional statistics
            'skewness': self._compute_skewness(values),
            'kurtosis': self._compute_kurtosis(values),
            
            # Relative changes
            'max_pct_change': float(np.max(np.abs(np.diff(values) / (np.abs(values[:-1]) + 1e-6)))),
        }
    
    def _sensor_fault_features(self, values):
        """Features for detecting sensor faults: stuck sensor, spikes, noise"""
        diffs = np.diff(values)
        
        return {
            # Stuck sensor detection
            'consecutive_constants': float(self._max_consecutive_constants(values, tolerance=0.01)),
            'n_constant_values': float(self._count_constants(values, tolerance=0.01)),
            'fraction_constant': float(self._count_constants(values, tolerance=0.01) / len(values)),
            
            # Spike detection
            'max_abs_change': float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0,
            'n_spikes': float(self._count_spikes(values, threshold_std=3)),
            'spike_ratio': float(self._count_spikes(values, threshold_std=3) / len(values)),
            
            # Noise detection
            'derivative_std': float(np.std(diffs)) if len(diffs) > 0 else 0.0,
            'high_frequency_noise': self._estimate_noise(values),
            
            # Zero variance regions
            'zero_variance_ratio': self._zero_variance_ratio(values),
        }
    
    def _distribution_features(self, values):
        """General statistical/distribution features"""
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'range': float(np.max(values) - np.min(values)),
            'coefficient_variation': float(np.std(values) / (np.mean(values) + 1e-6)),
            'percentile_25': float(np.percentile(values, 25)),
            'percentile_75': float(np.percentile(values, 75)),
            'entropy': self._compute_entropy(values),
        }
    
    # ===== HELPER METHODS =====
    
    def _max_consecutive_constants(self, values, tolerance=0.01):
        """Count longest sequence of constant values"""
        diffs = np.abs(np.diff(values))
        consecutive = 0
        max_consecutive = 0
        
        for d in diffs:
            if d <= tolerance:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return float(max_consecutive)
    
    def _count_constants(self, values, tolerance=0.01):
        """Count number of constant values"""
        diffs = np.abs(np.diff(values))
        return float(np.sum(diffs <= tolerance))
    
    def _max_consecutive_direction(self, direction_bool):
        """Max consecutive True values in boolean array"""
        consecutive = 0
        max_consecutive = 0
        for d in direction_bool:
            if d:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        return float(max_consecutive)
    
    def _count_spikes(self, values, threshold_std=3):
        """Count number of spike anomalies"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.sum(np.abs(values - mean) > threshold_std * std))
    
    def _estimate_noise(self, values):
        """Estimate high-frequency noise level"""
        diffs = np.diff(values)
        if len(diffs) < 2:
            return 0.0
        # RMS of second derivative as noise proxy
        second_diffs = np.diff(diffs)
        return float(np.sqrt(np.mean(second_diffs**2)))
    
    def _zero_variance_ratio(self, values, window_size=5):
        """Ratio of windows with zero variance"""
        if len(values) <= window_size:
            return 0.0
        
        var_count = 0
        for i in range(len(values) - window_size):
            if np.std(values[i:i+window_size]) < 1e-6:
                var_count += 1
        return float(var_count / (len(values) - window_size))
    
    def _compute_local_outlier_factor(self, values, k=5):
        """Simple LOF approximation - returns scalar"""
        if len(values) <= k:
            return 0.0
        
        distances = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
        k_distances = np.sort(distances, axis=1)[:, min(k, len(values)-1)]
        return float(np.max(k_distances))
    
    def _compute_skewness(self, values):
        """Compute skewness"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 3))
    
    def _compute_kurtosis(self, values):
        """Compute kurtosis (excess)"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 4) - 3)
    
    def _compute_entropy(self, values, n_bins=20):
        """Compute Shannon entropy of distribution"""
        try:
            hist, _ = np.histogram(values, bins=min(n_bins, len(values)))
            hist = hist[hist > 0] / len(values)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def extract_features_for_entity(self, entity_data):
        """Extract features for complete entity data"""
        
        features_result = {}
        
        # Extract features for training data
        if 'train' in entity_data and len(entity_data['train']) > 0:
            train_df = entity_data['train'].sort_values('time')
            train_features = self.extract_features_for_timeseries(
                train_df['time'].values,
                train_df['Oxygen[%sat]'].values
            )
            if train_features is not None:
                features_result['train_features'] = train_features
                print(f"  ✓ Train features: {len(train_features)} windows extracted")
            else:
                print(f"  ✗ Train features: Failed")
        
        # Extract features for test data
        if 'test' in entity_data and len(entity_data['test']) > 0:
            test_df = entity_data['test'].sort_values('time')
            test_features = self.extract_features_for_timeseries(
                test_df['time'].values,
                test_df['Oxygen[%sat]'].values
            )
            
            if test_features is not None:
                # Add anomaly labels if present
                if 'anomaly_label' in test_df.columns:
                    test_features['anomaly_label'] = 0
                    
                    for idx, row in test_features.iterrows():
                        window_center = row['window_center']
                        closest_idx = (test_df['time'] - window_center).abs().argmin()
                        label_value = int(test_df.iloc[closest_idx]['anomaly_label'])
                        test_features.at[idx, 'anomaly_label'] = label_value
                
                features_result['test_features'] = test_features
                print(f"  ✓ Test features: {len(test_features)} windows extracted")
            else:
                print(f"  ✗ Test features: Failed")
        
        return features_result
    
    def extract_and_save_features(self, preprocessor, output_dir='./'):
        """Extract features for all entities and save to disk"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n[Feature Extraction] Processing all entities...")
        
        all_features = {}
        successful_entities = 0
        
        for entity_id in preprocessor.get_all_entities():
            print(f"\n  {entity_id}:")
            
            entity_data = preprocessor.get_entity_data(entity_id)
            features = self.extract_features_for_entity(entity_data)
            
            all_features[entity_id] = features
            
            if 'train_features' in features or 'test_features' in features:
                successful_entities += 1
            
            # Save features
            if 'train_features' in features:
                features['train_features'].to_csv(
                    f"{output_dir}/{entity_id}_train_features.csv",
                    index=False
                )
            
            if 'test_features' in features:
                features['test_features'].to_csv(
                    f"{output_dir}/{entity_id}_test_features.csv",
                    index=False
                )
        
        print(f"\n{'='*70}")
        print(f"✓ Feature extraction complete!")
        print(f"✓ Successfully processed {successful_entities}/{len(preprocessor.get_all_entities())} entities")
        print(f"✓ Features saved to {output_dir}")
        print(f"{'='*70}")
        
        return all_features
    
    def get_feature_statistics(self, features_df):
        """Compute statistics on extracted features"""
        
        if features_df is None or len(features_df) == 0:
            return None
        
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_df = features_df[numeric_features]
        
        stats = {
            'n_features': len(numeric_features),
            'n_windows': len(features_df),
            'mean_values': numeric_df.mean(),
            'std_values': numeric_df.std(),
            'feature_correlation': numeric_df.corr()
        }
        
        return stats


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("="*70)
    print("OXYGEN SENSOR ANOMALY DETECTION - FEATURE EXTRACTION")
    print("="*70)
    
    preprocessor = DataPreprocessor("/Users/ferdousbinali/rag_pipeline/cefalo/data/oxygen_14_11_25.csv")
    preprocessor.load_data()
    preprocessor.create_entity_ids()
    unique_entities = preprocessor.raw_df['entity_id'].unique()
    preprocessor.segment_by_entity(unique_entities)
    preprocessor.handle_data_quality(max_gap_minutes=10)
    split_info = preprocessor.create_train_test_split(test_size_days=7)
    
    # Create synthetic anomalies
    if len(preprocessor.get_all_entities()) > 0:
        test_data, labels, details = preprocessor.create_synthetic_anomalies()
    
    # Step 2: Extract features with robust imputation
    print("\n" + "="*70)
    extractor = RobustFeatureExtractor(
        window_size_minutes=60,
        overlap_minutes=59,
        imputation_method='ensemble'
    )
    all_features = extractor.extract_and_save_features(
        preprocessor,
        output_dir='/Users/ferdousbinali/rag_pipeline/cefalo/data/features/'
    )
    
    # Display statistics
    first_entity = preprocessor.get_all_entities()[0]
    if first_entity in all_features and 'train_features' in all_features[first_entity]:
        train_features = all_features[first_entity]['train_features']
        stats = extractor.get_feature_statistics(train_features)
        
        if stats is not None:
            print(f"\n✓ Feature Statistics for {first_entity}:")
            print(f"  Total features: {stats['n_features']}")
            print(f"  Windows: {stats['n_windows']}")
            print(f"\n  Top features by variance:")
            top_features = stats['std_values'].nlargest(15)
            for feat_name, std_val in top_features.items():
                print(f"    {feat_name:40s}: {std_val:.4f}")