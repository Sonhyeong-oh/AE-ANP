import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler
import ast

import os

class DataGenerator:
    def __init__(self, config, excel_path, image_dirs, image_col='Images', scaler = None):
        """
        Initializes the data loader for variable-length sequences.
        Args:
            config: The main configuration dictionary.
            excel_path (str): Path to the input excel file.
            image_dirs (list): A list of directories where images are stored.
            image_col (str): The name of the column containing image paths.
        """
        self.config = config
        self.scaler = scaler
        self.image_col = image_col
        self.image_width = config['image_width']
        self.image_height = config['image_height']
        self.feature_cols = None
        self.image_dirs = image_dirs
        self.img_path_map = self._create_image_path_map()
        
        self.num_channels = config.get('num_img_channels', 3)
        
        self.sequences, self.image_paths, self.sequence_lengths, self.labels = self._load_data(excel_path)

    def _create_image_path_map(self):
        """Creates a mapping from image filenames to their full paths by recursively searching directories."""
        print("Creating image path map (recursive search)...")
        path_map = {}
        for directory in self.image_dirs:
            if not os.path.isdir(directory):
                print(f"Warning: Directory not found: {directory}")
                continue
            for root, _, files in os.walk(directory):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if filename not in path_map:
                            path_map[filename] = os.path.join(root, filename)
        print(f"Found {len(path_map)} unique images in the provided directories.")
        return path_map

    def _load_data(self, excel_path):
            """
            Loads and preprocesses all data from the excel and image directories.
            """
            print(f"Reading and preprocessing excel file: {excel_path}")
            df = pd.read_excel(excel_path)
    
            bool_features = df.select_dtypes(include=['bool']).columns.tolist()
    
            # Process 'time' and 'label' columns if they exist
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if 'label' in df.columns:
                df['label'] = pd.to_numeric(df['label'], errors='coerce')
                df.dropna(subset=['label'], inplace=True)
                df['label'] = df['label'].astype(int)
            else:
                df['label'] = 0
    
            # 1. Initial selection
            initial_features = [col for col in df.columns if col not in ['time', self.image_col, 'label']]
            
            # Force numeric
            for col in initial_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 2. Filter for numeric types
            numeric_features = df[initial_features].select_dtypes(include=np.number).columns.tolist()
            self.feature_cols = numeric_features
    
            # 스케일링 및 차분을 적용할 컬럼 선정 (Bool 제외)
            scaling_features = [col for col in self.feature_cols if col not in bool_features]
    
            # ==============================================================================
            # [수정됨] 1차 차분 (Delta) 적용 구간
            # 위치: 스케일링 대상 컬럼이 선정된 직후, 스케일러 적용 전
            # ==============================================================================
            if scaling_features:
                print("Applying 1st order difference (Delta) processing...")
                
                # (1) 차분 전 결측치 보간: 중간에 NaN이 있으면 diff 결과도 NaN이 되므로 방지
                # ffill로 직전 값을 채우고, 맨 앞쪽 NaN은 0으로 채움
                df[scaling_features] = df[scaling_features].ffill().fillna(0)
    
                # (2) 1차 차분 적용 (변화량 계산)
                df[scaling_features] = df[scaling_features].diff()
    
                # (3) 차분 후 첫 번째 행 처리
                # diff()를 하면 첫 번째 행은 무조건 NaN이 되므로 0으로 채움 (변화량 없음 간주)
                df[scaling_features] = df[scaling_features].fillna(0)
            # ==============================================================================
    
            # 3. Remove columns where all values are NaN
            # (차분 후 모든 값이 0이거나 NaN인 컬럼이 생길 수 있으므로 여기서 확인)
            all_nan_cols = df[numeric_features].isnull().all()
            cols_to_drop = all_nan_cols[all_nan_cols].index.tolist()
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                print(f"Dropped all-NaN columns: {cols_to_drop}")
            
            # Update feature columns list
            self.feature_cols = [key for key in numeric_features if key not in cols_to_drop]
            # 차분 후 컬럼이 삭제되었을 수 있으므로 scaling_features 리스트도 업데이트
            scaling_features = [col for col in self.feature_cols if col not in bool_features]
    
            # 4. Fit scaler on normal data & 5. Transform all data
            # (차분이 완료된 데이터를 기반으로 스케일링 수행)
            if scaling_features and 'label' in df.columns:
                if self.scaler is None:
                    print("Fitting a new StandardScaler on normal data (after diff).")
                    normal_df = df[df['label'] == 0]
                    if not normal_df.empty:
                        self.scaler = StandardScaler()
                        self.scaler.fit(normal_df[scaling_features])
                    else:
                        print("Warning: No normal data available. Using empty scaler.")
                        self.scaler = StandardScaler()
                
                print(f"Applying StandardScaler to columns: {len(scaling_features)}")
                df[scaling_features] = self.scaler.transform(df[scaling_features])
            
            # Fill any remaining NaNs with 0 after scaling
            if self.feature_cols:
                df[self.feature_cols] = df[self.feature_cols].fillna(0)
    
            # ... (이하 시퀀스 생성 코드는 동일) ...
            image_indices = df.index[df[self.image_col].notna()].tolist()
            
            all_sequences = []
            all_image_full_paths = []
            all_sequence_lengths = []
            all_labels = [] 
            skip_num = 0
            start_index = 0
            
            for end_index in image_indices:
                sequence_df = df.iloc[start_index:end_index]
                
                image_paths_str = df.at[end_index, self.image_col]
                try:
                    image_paths = ast.literal_eval(image_paths_str)
                    if not isinstance(image_paths, list):
                        image_paths = [image_paths_str]
                except (ValueError, SyntaxError):
                    image_paths = [image_paths_str]
    
                if not sequence_df.empty:
                    sequence_label = 1 if (sequence_df['label'] == 1).any() else 0
                    numerical_sequence = sequence_df[self.feature_cols].values
                    
                    for image_path in image_paths:
                        image_filename = os.path.basename(image_path)
                        full_image_path = self.img_path_map.get(image_filename)
    
                        if full_image_path:
                            all_sequences.append(numerical_sequence)
                            all_image_full_paths.append(full_image_path)
                            all_sequence_lengths.append(len(numerical_sequence))
                            all_labels.append(sequence_label)
    
                        if not full_image_path:
                            skip_num += 1
                            continue
                        
                start_index = end_index + 1
    
            print(f"Number of Skip Image: {skip_num}") 
            print(f"Successfully loaded {len(all_sequences)} sequences and images.\n")
    
            return all_sequences, all_image_full_paths, all_sequence_lengths, all_labels

    def _parse_image_function(self, sequence, image_path, seq_len, label):
            """텐서플로우 파이프라인에서 경로를 받아 이미지를 로드하는 함수"""
            # 경로를 통해 이미지 읽기
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image_string, channels=3)
            # 여기서 미리 224x224로 리사이즈하면 메모리와 속도면에서 매우 유리합니다.
            image = tf.image.resize(image, [224, 224]) 
            image = image / 255.0  # 정규화
            return sequence, image, seq_len, label
    
    def get_dataset(self):
            # 1. 경로와 숫자 데이터를 포함한 기초 데이터셋 생성
            dataset = tf.data.Dataset.from_generator(
                lambda: iter(zip(self.sequences, self.image_paths, self.sequence_lengths, self.labels)),
                output_types=(tf.float32, tf.string, tf.int32, tf.int32), # 이미지는 아직 string(경로)
            )

            # 2. Map을 사용하여 배치를 만들 때마다 이미지를 로드 (On-the-fly)
            # num_parallel_calls=tf.data.AUTOTUNE을 사용하여 CPU 병렬 처리를 활성화
            dataset = dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)

            # 3. 패딩 및 배치 설정
            dataset = dataset.padded_batch(
                self.config['batch_size'],
                padded_shapes=(
                    [None, len(self.feature_cols)],
                    [224, 224, 3], # 리사이즈된 크기
                    [],
                    []
                ),
                padding_values=(0.0, 0.0, 0, 0),
                drop_remainder=True
            )

            return dataset.prefetch(tf.data.AUTOTUNE) # 미리 준비하여 GPU 대기 시간 감소