import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn.functional as F
import copy

# VAE.vqvae는 외부 모듈이므로 그대로 import
# Colab 환경에 맞게 VAE 폴더를 업로드하고 경로를 맞춰주어야 합니다.
from VAE.vqvae import VQVAE

# --- [train_colab.py] Configuration ---
IMAGE_RESOLUTION = 256
ORIGINAL_IMAGE_DIMS = (1080, 720) # (width, height)
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 1

# --- 경로 설정 (Train/Validation/Test) ---
BASE_DATA_PATH = "D:/archive/Sensor_Data(Label)/EDT/sample"
BEST_MODEL_SAVE_PATH = "best_model_labeled.pth"

PATHS = {
    'train': {
        'excel_files': [os.path.join(BASE_DATA_PATH, 'train', 'FFCell_CycleManagement_train.xlsx'), 
                        os.path.join(BASE_DATA_PATH, 'train', 'FFCellSafetyManagement_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'Conveyor_Signals_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R01_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R02_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R03_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R04_Data_train.xlsx')
                       ],
        'normal_jsons': ["D:/archive/Batch 1/data_1000.json", "D:/archive/Batch 1/data_2000.json"],
        'normal_imgs': ["D:/archive/Batch 1/BATCH1000", "D:/archive/Batch 1/BATCH2000"],
        'error_jsons': ["D:/archive/Batch 1/data_24000.json", "D:/archive/Batch 1/data_25000.json"],
        'error_imgs': ["D:/archive/Batch 1/BATCH24000", "D:/archive/Batch 1/BATCH25000"],
    },
    'val': {
        'excel_files': [os.path.join(BASE_DATA_PATH, 'val', 'FFCell_CycleManagement_val.xlsx'), 
                        os.path.join(BASE_DATA_PATH, 'val', 'FFCellSafetyManagement_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'Conveyor_Signals_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R01_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R02_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R03_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R04_Data_val.xlsx')
                       ],
        'normal_jsons': ["D:/archive/Batch 1/data_3000.json"],
        'normal_imgs': ["D:/archive/Batch 1/BATCH3000"],
        'error_jsons': ["D:/archive/Batch 1/data_26000.json"],
        'error_imgs': ["D:/archive/Batch 1/BATCH26000"],
    },
    'test': {
        'excel_files': [os.path.join(BASE_DATA_PATH, 'test', 'FFCell_CycleManagement_test.xlsx'), 
                        os.path.join(BASE_DATA_PATH, 'test', 'FFCellSafetyManagement_test.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'test', 'Conveyor_Signals_test.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'test', 'R01_Data_test.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'test', 'R02_Data_test.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'test', 'R03_Data_test.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'test', 'R04_Data_test.xlsx')
                       ],
        'normal_jsons': ["D:/archive/Batch 1/data_4000.json"],
        'normal_imgs': ["D:/archive/Batch 1/BATCH4000"],
        'error_jsons': ["D:/archive/Batch 1/data_27000.json"],
        'error_imgs': ["D:/archive/Batch 1/BATCH27000"],
    }
}
# --- End of Configuration ---

# --- 모델 정의 (이전과 동일) ---
class VQVAE_ANP_Predictor_V3(nn.Module):
    # ... (No changes needed in model definition)
    def __init__(self, vqvae_model, excel_dim, coord_dim, original_image_dims,
                 component_coords_0, component_order_0,
                 component_coords_1, component_order_1,
                 num_components, num_continuous, num_boolean, 
                 hidden_dim=128, region_feature_dim=128):
        super(VQVAE_ANP_Predictor_V3, self).__init__()
        self.vqvae = vqvae_model
        self.original_image_dims = original_image_dims
        self.component_coords_0, self.component_order_0 = component_coords_0, component_order_0
        self.component_coords_1, self.component_order_1 = component_coords_1, component_order_1
        self.num_components, self.region_feature_dim = num_components, region_feature_dim
        self.num_continuous, self.num_boolean = num_continuous, num_boolean
        x_dim = (num_components * region_feature_dim) + 1 + coord_dim
        context_y_dim = excel_dim
        self.determ_encoder = nn.Sequential(nn.Linear(x_dim + context_y_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.continuous_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, num_continuous * 2))
        self.boolean_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_boolean))

    def _get_scaled_boxes(self, feature_map_size, device, component_order, component_coords):
        map_h, map_w = feature_map_size
        img_w, img_h = self.original_image_dims
        boxes = [[component_coords[name]['x1'] * (map_w / img_w), component_coords[name]['y1'] * (map_h / img_h), component_coords[name]['x2'] * (map_w / img_w), component_coords[name]['y2'] * (map_h / img_h)] for name in component_order]
        return torch.tensor(boxes, dtype=torch.float32, device=device)

    def forward(self, img, excel_seq, coords, component_type):
        batch_size, seq_len, _ = excel_seq.shape
        with torch.no_grad():
            qt, qb, _, _, _ = self.vqvae.encode(img)
        comp_order, comp_coords = (self.component_order_1, self.component_coords_1) if component_type.item() == 1 else (self.component_order_0, self.component_coords_0)
        scaled_boxes = self._get_scaled_boxes(qt.shape[2:], qt.device, comp_order, comp_coords)
        box_indices = torch.zeros(len(comp_order), dtype=torch.int, device=qt.device)
        boxes_for_align = torch.cat([box_indices.unsqueeze(1), scaled_boxes], dim=1)
        z_regions_t = roi_align(qt, boxes_for_align, output_size=(1, 1), aligned=True)
        z_regions_b = roi_align(qb, boxes_for_align, output_size=(1, 1), aligned=True)
        z_regions = torch.cat([z_regions_t, z_regions_b], dim=1).squeeze().view(1, -1)
        z_img_seq = z_regions.unsqueeze(1).expand(-1, seq_len, -1)
        time_idx = torch.linspace(0, 1, seq_len).view(1, seq_len, 1).expand(batch_size, -1, -1).to(img.device)
        coords_seq = coords.unsqueeze(1).expand(-1, seq_len, -1)
        context_x = torch.cat([z_img_seq, time_idx, coords_seq], dim=-1)
        context_y = excel_seq
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        r = self.determ_encoder(encoder_input)
        r_agg = r.mean(dim=1)
        continuous_params = self.continuous_decoder(r_agg)
        boolean_logits = self.boolean_decoder(r_agg)
        mu, log_sigma = torch.chunk(continuous_params, 2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        return mu, sigma, boolean_logits

# --- 상수 및 헬퍼 함수 정의 ---
COMPONENT_COORDS_1 = { 'MHS': {'h': 251, 'w': 397, 'x1': 363, 'x2': 760, 'y1': 278, 'y2': 529}, 'R01': {'h': 439, 'w': 227, 'x1': 199, 'x2': 426, 'y1': 124, 'y2': 563}, 'R02': {'h': 187, 'w': 128, 'x1': 379, 'x2': 507, 'y1': 108, 'y2': 295}, 'R03': {'h': 148, 'w': 120, 'x1': 287, 'x2': 407, 'y1': 106, 'y2': 254}, 'R04': {'h': 264, 'w': 268, 'x1': 589, 'x2': 857, 'y1': 164, 'y2': 428}, 'Conv1': {'h': 39, 'w': 183, 'x1': 179, 'x2': 362, 'y1': 178, 'y2': 217}, 'Conv2': {'h': 182, 'w': 301, 'x1': 168, 'x2': 469, 'y1': 208, 'y2': 390}, 'Conv3': {'h': 81, 'w': 215, 'x1': 457, 'x2': 672, 'y1': 264, 'y2': 345}, 'Conv4': {'h': 106, 'w': 338, 'x1': 354, 'x2': 692, 'y1': 173, 'y2': 279} }
COMPONENT_ORDER_1 = ['MHS', 'R01', 'R02', 'R03', 'R04', 'Conv1', 'Conv2', 'Conv3', 'Conv4']
COMPONENT_COORDS_0 = { 'MHS': {'h': 53, 'w': 183, 'x1': 232, 'x2': 415, 'y1': 182, 'y2': 235}, 'R01': {'h': 160, 'w': 99, 'x1': 385, 'x2': 484, 'y1': 102, 'y2': 262}, 'R02': {'h': 148, 'w': 140, 'x1': 273, 'x2': 413, 'y1': 143, 'y2': 291}, 'R03': {'h': 187, 'w': 173, 'x1': 372, 'x2': 545, 'y1': 161, 'y2': 348}, 'R04': {'h': 139, 'w': 131, 'x1': 173, 'x2': 304, 'y1': 145, 'y2': 284}, 'Conv1': {'h': 158, 'w': 334, 'x1': 414, 'x2': 748, 'y1': 362, 'y2': 520}, 'Conv2': {'h': 139, 'w': 346, 'x1': 380, 'x2': 726, 'y1': 210, 'y2': 349}, 'Conv3': {'h': 43, 'w': 225, 'x1': 131, 'x2': 356, 'y1': 228, 'y2': 271}, 'Conv4': {'h': 294, 'w': 380, 'x1': 91, 'x2': 471, 'y1': 268, 'y2': 562} }
COMPONENT_ORDER_0 = ['MHS', 'R01', 'R02', 'R03', 'R04', 'Conv1', 'Conv2', 'Conv3', 'Conv4']
NUM_COMPONENTS = len(COMPONENT_ORDER_0)
COORD_DIM = NUM_COMPONENTS * 4
BOOLEAN_SENSOR_KEYS = [ "I_SafetyDoor1_Status", "I_SafetyDoor2_Status", "I_HMI_EStop_Status", "I_MHS_GreenRocketTray", "I_Stopper1_Status", "I_Stopper2_Status", "I_Stopper3_Status", "I_Stopper4_Status", "I_Stopper5_Status" ]
_BASE_KEYS = [ "Q_VFD1_Temperature", "Q_VFD2_Temperature", "Q_VFD3_Temperature", "Q_VFD4_Temperature", "M_Conv1_Speed_mmps", "M_Conv2_Speed_mmps", "M_Conv3_Speed_mmps", "M_Conv4_Speed_mmps", "I_R01_Gripper_Pot", "I_R01_Gripper_Load", "I_R02_Gripper_Pot", "I_R02_Gripper_Load", "I_R03_Gripper_Pot", "I_R03_Gripper_Load", "I_R04_Gripper_Pot", "I_R04_Gripper_Load", "I_R01_GripperLoad_lbf", "I_R02_GripperLoad_lbf", "I_R03_GripperLoad_lbf", "I_R04_GripperLoad_lbf", "M_R01_SJointAngle_Degree", "M_R01_LJointAngle_Degree", "M_R01_UJointAngle_Degree", "M_R01_RJointAngle_Degree", "M_R01_BJointAngle_Degree", "M_R01_TJointAngle_Degree", "M_R02_SJointAngle_Degree", "M_R02_LJointAngle_Degree", "M_R02_UJointAngle_Degree", "M_R02_RJointAngle_Degree", "M_R02_BJointAngle_Degree", "M_R02_TJointAngle_Degree", "M_R03_SJointAngle_Degree", "M_R03_LJointAngle_Degree", "M_R03_UJointAngle_Degree", "M_R03_RJointAngle_Degree", "M_R03_BJointAngle_Degree", "M_R03_TJointAngle_Degree", "M_R04_SJointAngle_Degree", "M_R04_LJointAngle_Degree", "M_R04_UJointAngle_Degree", "M_R04_RJointAngle_Degree", "M_R04_BJointAngle_Degree", "M_R04_TJointAngle_Degree", "Q_Cell_CycleCount", "CycleState" ]
ALL_SENSOR_KEYS = sorted(list(set(_BASE_KEYS + BOOLEAN_SENSOR_KEYS)))
CONTINUOUS_SENSOR_KEYS = [key for key in ALL_SENSOR_KEYS if key not in BOOLEAN_SENSOR_KEYS]

def get_normalized_coords_tensor(coords_map, order, dims):
    w, h = dims
    all_coords = []
    for name in order:
        c = coords_map[name]
        all_coords.extend([(c['x1']+c['x2'])/2/w, (c['y1']+c['y2'])/2/h, c['w']/w, c['h']/h])
    return torch.tensor(all_coords, dtype=torch.float32)

def load_and_label_excel_data(excel_files):
    """
    FFCell_CycleManagement.xlsx'의 'Description' 열을 기준으로 시간대별 normal/error를 판별하고,
    모든 Excel 데이터를 병합한 후 normal과 error 데이터프레임으로 분리하여 반환합니다.
    (수정: 파일 검색 및 경로 처리 로직 강화)
    """
    master_filename_base = 'FFCell_CycleManagement'
    master_file = next((f for f in excel_files if os.path.basename(f).startswith(master_filename_base)), None)
    
    if not master_file:
        raise FileNotFoundError(f"Master file starting with '{master_filename_base}' not found in the list: {excel_files}")

    # 경로를 OS에 맞게 표준화하고 파일 존재 여부 확인
    normalized_master_path = os.path.normpath(master_file)
    if not os.path.exists(normalized_master_path):
        raise FileNotFoundError(f"Master file '{master_file}' was found in the path list but does not actually exist on disk at the normalized path: '{normalized_master_path}'")
    
    print(f"Using master file for labeling: {normalized_master_path}")
    master_df = pd.read_excel(normalized_master_path)
    master_df['time'] = pd.to_datetime(master_df['time'])
    
    # Description 열의 내용 유무로 label 생성 (내용 있으면 True=error, 없으면 False=normal)
    master_df['label'] = master_df['Description'].notna()
    
    # 연속된 레이블 블록의 시간 구간 찾기
    master_df['block'] = (master_df['label'] != master_df['label'].shift()).cumsum()
    labeled_periods = master_df.groupby('block').agg(
        start_time=('time', 'min'),
        end_time=('time', 'max'),
        is_error=('label', 'first')
    )
    
    # 모든 Excel 파일 병합
    dfs = []
    for f in excel_files:
        normalized_f = os.path.normpath(f)
        if not os.path.exists(normalized_f): 
            print(f"Warning: Excel file not found at {normalized_f}. Skipping.")
            continue
        df = pd.read_excel(normalized_f)
        df['time'] = pd.to_datetime(df['time'])
        # Description 열은 여기서 필요 없으므로, 숫자/시간/불리언 타입만 선택
        df = df.select_dtypes(include=[np.number, 'datetime64', 'bool'])
        dfs.append(df.set_index('time'))
        
    if not dfs: raise FileNotFoundError("No valid Excel files found to merge.")
    merged_df = pd.concat(dfs, axis=1).sort_index().dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill').dropna().reset_index()

    # 시간 구간에 따라 normal/error 데이터프레임으로 분리
    normal_df_list, error_df_list = [], []
    for _, period in labeled_periods.iterrows():
        mask = (merged_df['time'] >= period['start_time']) & (merged_df['time'] <= period['end_time'])
        data_chunk = merged_df[mask]
        if period['is_error']:
            error_df_list.append(data_chunk)
        else:
            normal_df_list.append(data_chunk)
            
    normal_df = pd.concat(normal_df_list, ignore_index=True) if normal_df_list else pd.DataFrame(columns=merged_df.columns)
    error_df = pd.concat(error_df_list, ignore_index=True) if error_df_list else pd.DataFrame(columns=merged_df.columns)
    
    print(f"Labeled Excel data loaded. Normal rows: {len(normal_df)}, Error rows: {len(error_df)}")
    return normal_df, error_df

# --- Dataset Class (Zero-Masking Applied) ---
class MultiModalManufacturingDataset(Dataset):
    def __init__(self, json_files, img_dirs, excel_df, transform, continuous_scaler, excel_scaler):
        self.img_dirs, self.transform = img_dirs, transform
        self.excel_df, self.continuous_scaler, self.excel_scaler = excel_df, continuous_scaler, excel_scaler
        self.normalized_coords_1 = get_normalized_coords_tensor(COMPONENT_COORDS_1, COMPONENT_ORDER_1, ORIGINAL_IMAGE_DIMS)
        self.normalized_coords_0 = get_normalized_coords_tensor(COMPONENT_COORDS_0, COMPONENT_ORDER_0, ORIGINAL_IMAGE_DIMS)
        
        self.samples = []
        for i, json_file in enumerate(json_files):
            img_dir = img_dirs[i]
            if not (os.path.exists(json_file) and os.path.exists(img_dir)):
                print(f"Warning: Data path not found. Skipping {json_file} or {img_dir}")
                continue
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            for time_str, record in json_data.items():
                if not record.get('Images'): continue
                for img_filename in record['Images']:
                    self.samples.append({'time_str': time_str, 'record': record, 'img_filename': img_filename, 'img_dir': img_dir})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        record, time_str = sample['record'], sample['time_str']
        current_time = pd.to_datetime(time_str)
        
        # excel_df가 비어있는 경우 처리
        if self.excel_df.empty: return None

        mask = (self.excel_df['time'] >= current_time) & (self.excel_df['time'] < (current_time + pd.Timedelta(milliseconds=500)))
        excel_window = self.excel_df.loc[mask].drop(columns=['time']).values
        if len(excel_window) == 0:
            # 가장 가까운 시간대의 데이터로 대체
            time_diff = (self.excel_df['time'] - current_time).abs()
            if time_diff.empty: return None
            closest_idx = time_diff.idxmin()
            excel_window = self.excel_df.iloc[[closest_idx]].drop(columns=['time']).values

        json_vals = record.get("Sensor_values", {})
        continuous_vals = np.array([float(json_vals.get(k, 0)) for k in CONTINUOUS_SENSOR_KEYS])
        boolean_vals = np.array([float(json_vals.get(k, 0)) for k in BOOLEAN_SENSOR_KEYS])
        
        img_path = os.path.join(sample['img_dir'], os.path.basename(sample['img_filename']))
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            return None

        coords, c_type = (self.normalized_coords_1, 1) if os.path.splitext(sample['img_filename'])[0].endswith('_1') else (self.normalized_coords_0, 0)
        
        # --- Start of Zero-Masking Logic ---
        # 1. Resize and convert to tensor
        resizer = transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION))
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(resizer(img))

        # 2. Create and apply mask
        map_h, map_w = IMAGE_RESOLUTION, IMAGE_RESOLUTION
        img_w, img_h = ORIGINAL_IMAGE_DIMS
        coords_map, order = (COMPONENT_COORDS_1, COMPONENT_ORDER_1) if c_type == 1 else (COMPONENT_COORDS_0, COMPONENT_ORDER_0)
        
        mask = torch.zeros(1, map_h, map_w)
        for name in order:
            c = coords_map[name]
            x1 = int(c['x1'] * map_w / img_w)
            y1 = int(c['y1'] * map_h / img_h)
            x2 = int(c['x2'] * map_w / img_w)
            y2 = int(c['y2'] * map_h / img_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(map_w, x2), min(map_h, y2)
            mask[:, y1:y2, x1:x2] = 1
        
        masked_img_tensor = img_tensor * mask

        # 3. Normalize
        normalizer = transforms.Normalize([0.5]*3, [0.5]*3)
        final_img = normalizer(masked_img_tensor)
        # --- End of Zero-Masking Logic ---
        
        return {"img": final_img, 
                "excel_context": torch.tensor(self.excel_scaler.transform(excel_window), dtype=torch.float32), 
                "cont_target": torch.tensor(self.continuous_scaler.transform(continuous_vals.reshape(1, -1))[0], dtype=torch.float32), 
                "bool_target": torch.tensor(boolean_vals, dtype=torch.float32), 
                "coords": coords, 
                "c_type": torch.tensor(c_type, dtype=torch.long)}

# --- 학습/평가 함수 (이전과 동일) ---
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def gaussian_nll(mu, sigma, target):
    return (0.5 * torch.log(2 * np.pi * sigma**2) + (target - mu)**2 / (2 * sigma**2)).sum()

def train_one_epoch(model, dataloader, optimizer, bce_loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if batch is None: continue
        img, excel_seq, cont_target, bool_target, coords, c_type = [batch[k].to(device) for k in batch]
        if torch.isnan(excel_seq).any(): continue
        optimizer.zero_grad()
        mu, sigma, bool_logits = model(img, excel_seq, coords, c_type)
        loss_cont = gaussian_nll(mu, sigma, cont_target)
        loss_bool = bce_loss_fn(bool_logits, bool_target)
        loss = loss_cont + loss_bool
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(pbar) if len(pbar) > 0 else 0

def evaluate(model, dataloader, bce_loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            if batch is None: continue
            img, excel_seq, cont_target, bool_target, coords, c_type = [batch[k].to(device) for k in batch]
            if torch.isnan(excel_seq).any(): continue
            mu, sigma, bool_logits = model(img, excel_seq, coords, c_type)
            loss_cont = gaussian_nll(mu, sigma, cont_target)
            loss_bool = bce_loss_fn(bool_logits, bool_target)
            loss = loss_cont + loss_bool
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(pbar) if len(pbar) > 0 else 0


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 1. Excel 데이터 로드 및 레이블링 ---
    print("--- Loading and Labeling Excel Data ---")
    normal_excel_train, error_excel_train = load_and_label_excel_data(PATHS['train']['excel_files'])
    normal_excel_val,   error_excel_val   = load_and_label_excel_data(PATHS['val']['excel_files'])
    normal_excel_test,  error_excel_test  = load_and_label_excel_data(PATHS['test']['excel_files'])

    # --- 2. 스케일러 피팅 (오직 Train 데이터 기준) ---
    print("\n--- Fitting Scalers on Training Data ---")
    # Excel 스케일러: normal/error 학습용 Excel 데이터를 모두 합쳐서 피팅
    combined_excel_train = pd.concat([normal_excel_train, error_excel_train], ignore_index=True)
    excel_scaler = StandardScaler().fit(combined_excel_train.drop(columns=['time']).values)

    # JSON(Continuous Sensor) 스케일러: 학습용 JSON 데이터를 모두 합쳐서 피팅
    def get_continuous_sensors_from_files(json_files):
        all_sensors = []
        for file in json_files:
            if os.path.exists(file):
                with open(file, 'r') as f: data = json.load(f)
                all_sensors.extend([[float(v["Sensor_values"].get(k, 0)) for k in CONTINUOUS_SENSOR_KEYS] for v in data.values()])
        return all_sensors
    train_sensor_data = get_continuous_sensors_from_files(PATHS['train']['normal_jsons'] + PATHS['train']['error_jsons'])
    continuous_scaler = StandardScaler().fit(train_sensor_data)
    print("Scalers fitted.")

    # --- 3. 데이터셋 및 데이터로더 생성 ---
    transform = transforms.Compose([transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    print("\n--- Creating Datasets and Dataloaders ---")
    
    # Train
    train_ds_normal = MultiModalManufacturingDataset(PATHS['train']['normal_jsons'], PATHS['train']['normal_imgs'], normal_excel_train, transform, continuous_scaler, excel_scaler)
    train_ds_error = MultiModalManufacturingDataset(PATHS['train']['error_jsons'], PATHS['train']['error_imgs'], error_excel_train, transform, continuous_scaler, excel_scaler)
    train_loader = DataLoader(ConcatDataset([train_ds_normal, train_ds_error]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    # Validation
    val_ds_normal = MultiModalManufacturingDataset(PATHS['val']['normal_jsons'], PATHS['val']['normal_imgs'], normal_excel_val, transform, continuous_scaler, excel_scaler)
    val_ds_error = MultiModalManufacturingDataset(PATHS['val']['error_jsons'], PATHS['val']['error_imgs'], error_excel_val, transform, continuous_scaler, excel_scaler)
    val_loader = DataLoader(ConcatDataset([val_ds_normal, val_ds_error]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # Test
    test_ds_normal = MultiModalManufacturingDataset(PATHS['test']['normal_jsons'], PATHS['test']['normal_imgs'], normal_excel_test, transform, continuous_scaler, excel_scaler)
    test_ds_error = MultiModalManufacturingDataset(PATHS['test']['error_jsons'], PATHS['test']['error_imgs'], error_excel_test, transform, continuous_scaler, excel_scaler)
    test_loader = DataLoader(ConcatDataset([test_ds_normal, test_ds_error]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    print("Dataloaders created.")

    # --- 4. 모델 초기화 및 학습 ---
    vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512).to(DEVICE)
    model = VQVAE_ANP_Predictor_V3(
        vqvae_model=vqvae, excel_dim=excel_scaler.n_features_in_, coord_dim=COORD_DIM, original_image_dims=ORIGINAL_IMAGE_DIMS,
        component_coords_0=COMPONENT_COORDS_0, component_order_0=COMPONENT_ORDER_0,
        component_coords_1=COMPONENT_COORDS_1, component_order_1=COMPONENT_ORDER_1,
        num_components=NUM_COMPONENTS, num_continuous=len(CONTINUOUS_SENSOR_KEYS), num_boolean=len(BOOLEAN_SENSOR_KEYS)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model_wts = None
    print("\n--- Starting Model Training ---")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss_fn, DEVICE)
        val_loss = evaluate(model, val_loader, bce_loss_fn, DEVICE)
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Validation loss improved to {val_loss:.4f}. Saving best model weights.")

    print("\n--- Training Finished ---")
    
    if best_model_wts is not None:
        print("\n--- Final Testing ---")
        print("Loading best model for final testing...")
        model.load_state_dict(best_model_wts)
        test_loss = evaluate(model, test_loader, bce_loss_fn, DEVICE)
        print(f"Final Test Loss: {test_loss:.4f}")
        torch.save(best_model_wts, BEST_MODEL_SAVE_PATH)
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
    else:
        print("No model was saved as validation loss did not improve.")
