"""
Data Preprocessing Module
Handles loading, cleaning, and segmenting oxygen sensor data by equipment entity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Preprocesses raw oxygen sensor data for anomaly detection and forecasting
    """
    
    def __init__(self, csv_path):
        """
        Initialize preprocessor with raw data
        
        Args:
            csv_path: Path to CSV file with oxygen sensor data
        """
        self.csv_path = csv_path
        self.raw_df = None
        self.processed_df = None
        self.entities = {}  # Dict of entity_id -> entity data
        
    def load_data(self):
        """Load CSV and perform initial validation"""
        print("[1/6] Loading data...")
        
        self.raw_df = pd.read_csv(self.csv_path)
        
        # Parse timestamps with mixed format handling (nanosecond precision support)
        # Try ISO8601 format first, which handles nanoseconds
        try:
            self.raw_df['time'] = pd.to_datetime(self.raw_df['time'], format='ISO8601')
            print("  → Timestamps parsed with ISO8601 format (nanosecond precision supported)")
        except Exception as e:
            try:
                # Fallback: Use 'mixed' format which infers per-element
                self.raw_df['time'] = pd.to_datetime(self.raw_df['time'], format='mixed')
                print("  → Timestamps parsed with mixed format inference")
            except Exception as e2:
                # Final fallback: Let pandas infer automatically
                self.raw_df['time'] = pd.to_datetime(self.raw_df['time'])
                print("  → Timestamps parsed with automatic inference")
        
        # Initial stats
        print(f"✓ Loaded {len(self.raw_df)} records")
        print(f"✓ Date range: {self.raw_df['time'].min()} to {self.raw_df['time'].max()}")
        print(f"✓ Null values in Oxygen[%sat]: {self.raw_df['Oxygen[%sat]'].isna().sum()} ({self.raw_df['Oxygen[%sat]'].isna().sum()/len(self.raw_df)*100:.1f}%)")
        print(f"✓ Unique equipment: {self.raw_df['EquipmentUnit'].nunique()}")
        print(f"✓ Unique systems: {self.raw_df['System'].nunique()}")
        
        return self.raw_df
    
    def create_entity_ids(self):
        """Create unique entity identifiers for tag-agnostic routing"""
        print("\n[2/6] Creating entity IDs...")
        
        # Entity ID = combination of EquipmentUnit + System (tag-agnostic)
        self.raw_df['entity_id'] = (
            self.raw_df['EquipmentUnit'].astype(str) + "_" + 
            self.raw_df['System'].astype(str)
        )
        
        unique_entities = self.raw_df['entity_id'].unique()
        print(f"✓ Created {len(unique_entities)} unique entities")
        print(f"✓ Entities: {list(unique_entities)[:5]}...")
        
        return unique_entities
    
    def segment_by_entity(self, unique_entities):
        """Segment data by entity and identify continuous data windows"""
        print("\n[3/6] Segmenting data by entity...")
        
        self.entities = {}
        
        for entity_id in unique_entities:
            entity_data = self.raw_df[self.raw_df['entity_id'] == entity_id].copy()
            entity_data = entity_data.sort_values('time').reset_index(drop=True)
            
            # Remove rows with null oxygen values (sensor offline)
            entity_clean = entity_data[entity_data['Oxygen[%sat]'].notna()].copy()
            
            if len(entity_clean) > 0:
                self.entities[entity_id] = {
                    'raw': entity_data,
                    'clean': entity_clean,
                    'n_records': len(entity_clean),
                    'time_span_hours': (entity_clean['time'].max() - entity_clean['time'].min()).total_seconds() / 3600,
                    'mean_oxygen': entity_clean['Oxygen[%sat]'].mean(),
                    'std_oxygen': entity_clean['Oxygen[%sat]'].std(),
                    'null_count': entity_data['Oxygen[%sat]'].isna().sum()
                }
        
        print(f"✓ Segmented into {len(self.entities)} entities with data")
        
        for entity_id, info in self.entities.items():
            print(f"  {entity_id}: {info['n_records']} clean records, "
                  f"{info['time_span_hours']:.1f}h span, "
                  f"O2: {info['mean_oxygen']:.2f}% ± {info['std_oxygen']:.2f}%")
        
        return self.entities
    
    def handle_data_quality(self, max_gap_minutes=10):
        """
        Handle missing data and gaps
        Keep gaps as-is (don't interpolate) but flag them for feature extraction
        """
        print(f"\n[4/6] Handling data quality (max gap: {max_gap_minutes} min)...")
        
        for entity_id in self.entities.keys():
            entity_data = self.entities[entity_id]['clean'].copy()
            
            # Identify gaps
            time_diff = entity_data['time'].diff().dt.total_seconds() / 60  # Convert to minutes
            gaps = time_diff[time_diff > max_gap_minutes]
            
            if len(gaps) > 0:
                print(f"  {entity_id}: Found {len(gaps)} gaps > {max_gap_minutes} min")
                print(f"    Max gap: {gaps.max():.0f} minutes")
            
            # Mark continuous segments
            entity_data['segment_id'] = (time_diff > max_gap_minutes).cumsum()
            
            self.entities[entity_id]['clean'] = entity_data
            self.entities[entity_id]['n_segments'] = entity_data['segment_id'].max() + 1
        
        print("✓ Data quality assessment complete")
        return self.entities
    
    def create_train_test_split(self, test_size_days=14):
        """
        Create training and test splits per entity
        Test = most recent N days, Train = remainder
        """
        print(f"\n[5/6] Creating train/test splits (test: {test_size_days} days)...")
        
        split_info = {}
        
        for entity_id in self.entities.keys():
            entity_data = self.entities[entity_id]['clean'].copy()
            
            # Calculate split point (most recent N days)
            max_time = entity_data['time'].max()
            split_time = max_time - timedelta(days=test_size_days)
            
            train_data = entity_data[entity_data['time'] < split_time].copy()
            test_data = entity_data[entity_data['time'] >= split_time].copy()
            
            self.entities[entity_id]['train'] = train_data
            self.entities[entity_id]['test'] = test_data
            
            split_info[entity_id] = {
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_period': f"{train_data['time'].min()} to {train_data['time'].max()}",
                'test_period': f"{test_data['time'].min()} to {test_data['time'].max()}"
            }
        
        for entity_id, info in split_info.items():
            print(f"  {entity_id}:")
            print(f"    Train: {info['train_size']} records")
            print(f"    Test:  {info['test_size']} records")
        
        return split_info
    
    def create_synthetic_anomalies(self, test_entity_id=None, anomaly_fraction=0.1):
        """
        Create synthetic anomalies in test data for validation
        Types: point, collective, stuck sensor, spikes
        
        Args:
            test_entity_id: Which entity to inject anomalies (if None, use first)
            anomaly_fraction: Fraction of test data to inject anomalies
        """
        print(f"\n[6/6] Creating synthetic anomalies ({anomaly_fraction*100:.0f}% of test data)...")
        
        if test_entity_id is None:
            test_entity_id = list(self.entities.keys())[0]
        
        test_data = self.entities[test_entity_id]['test'].copy()
        synthetic_labels = np.zeros(len(test_data))  # 0=normal, 1=anomaly
        
        n_anomalies = max(1, int(len(test_data) * anomaly_fraction))
        anomaly_indices = np.random.choice(len(test_data), size=n_anomalies, replace=False)
        
        anomaly_details = []
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['point_spike', 'point_dip', 'collective_high', 
                                            'collective_low', 'stuck_sensor', 'high_noise'])
            
            normal_value = test_data.iloc[idx]['Oxygen[%sat]']
            
            if anomaly_type == 'point_spike':
                # Single spike: +5% oxygen
                test_data.iloc[idx, test_data.columns.get_loc('Oxygen[%sat]')] = normal_value + 5.0
                
            elif anomaly_type == 'point_dip':
                # Single dip: -5% oxygen
                test_data.iloc[idx, test_data.columns.get_loc('Oxygen[%sat]')] = normal_value - 5.0
                
            elif anomaly_type == 'collective_high':
                # 60-minute sustained high
                end_idx = min(idx + 60, len(test_data))
                for j in range(idx, end_idx):
                    test_data.iloc[j, test_data.columns.get_loc('Oxygen[%sat]')] = normal_value + 2.0
                    synthetic_labels[j] = 1
                anomaly_indices = np.setdiff1d(anomaly_indices, np.arange(idx+1, end_idx))
                
            elif anomaly_type == 'collective_low':
                # 60-minute sustained low
                end_idx = min(idx + 60, len(test_data))
                for j in range(idx, end_idx):
                    test_data.iloc[j, test_data.columns.get_loc('Oxygen[%sat]')] = normal_value - 2.0
                    synthetic_labels[j] = 1
                anomaly_indices = np.setdiff1d(anomaly_indices, np.arange(idx+1, end_idx))
                
            elif anomaly_type == 'stuck_sensor':
                # Constant value for 30 minutes
                constant_val = normal_value + np.random.uniform(-1, 1)
                end_idx = min(idx + 30, len(test_data))
                for j in range(idx, end_idx):
                    test_data.iloc[j, test_data.columns.get_loc('Oxygen[%sat]')] = constant_val
                    synthetic_labels[j] = 1
                anomaly_indices = np.setdiff1d(anomaly_indices, np.arange(idx+1, end_idx))
                
            elif anomaly_type == 'high_noise':
                # High noise: ±3% random for 30 minutes
                end_idx = min(idx + 30, len(test_data))
                for j in range(idx, end_idx):
                    test_data.iloc[j, test_data.columns.get_loc('Oxygen[%sat]')] = (
                        normal_value + np.random.uniform(-3, 3)
                    )
                    synthetic_labels[j] = 1
                anomaly_indices = np.setdiff1d(anomaly_indices, np.arange(idx+1, end_idx))
            
            synthetic_labels[idx] = 1
            
            anomaly_details.append({
                'index': idx,
                'type': anomaly_type,
                'timestamp': test_data.iloc[idx]['time'],
                'injected_value': test_data.iloc[idx]['Oxygen[%sat]'],
                'original_value': normal_value
            })
        
        test_data['anomaly_label'] = synthetic_labels
        self.entities[test_entity_id]['test'] = test_data
        
        print(f"✓ Injected {len(anomaly_details)} synthetic anomalies")
        print(f"✓ Anomaly types distribution:")
        type_counts = pd.Series([a['type'] for a in anomaly_details]).value_counts()
        for atype, count in type_counts.items():
            print(f"  - {atype}: {count}")
        
        return test_data, synthetic_labels, anomaly_details
    
    def get_entity_data(self, entity_id):
        """Get preprocessed data for specific entity"""
        return self.entities.get(entity_id)
    
    def get_all_entities(self):
        """Get all entity IDs"""
        return list(self.entities.keys())
    
    def save_preprocessed_data(self, output_dir='./'):
        """Save preprocessed data to CSV for inspection"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for entity_id, entity_data in self.entities.items():
            entity_data['clean'].to_csv(
                f"{output_dir}/entity_{entity_id}_clean.csv",
                index=False
            )
        
        print(f"✓ Saved preprocessed data to {output_dir}")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    # Load and preprocess data
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
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Ready for feature extraction and model training")
    
    preprocessor.save_preprocessed_data('/Users/ferdousbinali/rag_pipeline/cefalo/data/')