import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import roi_align
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 사용자 참고: train_colab_v2.py와 호환되도록 일부 정의를 아래에 직접 포함합니다.
from train_colab_v2 import (
    VQVAE_ANP_Predictor_V3,
    COMPONENT_COORDS_0, COMPONENT_ORDER_0,
    COMPONENT_COORDS_1, COMPONENT_ORDER_1,
    NUM_COMPONENTS, COORD_DIM,
    BOOLEAN_SENSOR_KEYS, CONTINUOUS_SENSOR_KEYS, ALL_SENSOR_KEYS,
    load_and_label_excel_data as load_and_merge_excel, # 이름 일치
    gaussian_nll,
    get_normalized_coords_tensor
)
from VAE.vqvae import VQVAE

# --- train_colab_v2.py와 설정을 동기화하기 위한 상수 정의 ---
IMAGE_RESOLUTION = 256
IMAGE_DIMS = (1080, 720) # ORIGINAL_IMAGE_DIMS 대신 IMAGE_DIMS 사용

# --- XAI 시각화 함수 정의 ---

def map_sensor_to_component(sensor_name, component_order):
    """센서 이름을 부품 이름으로 매핑합니다."""
    for component in sorted(component_order, key=len, reverse=True):
        if component in sensor_name:
            return component
    return None

def overlay_cam_on_image(original_img, mask, colormap='jet', alpha=0.6):
    """(공통 함수) 마스크를 원본 이미지 위에 오버레이합니다."""
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(mask)[:, :, :3]
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = heatmap.resize(original_img.size, Image.Resampling.LANCZOS)
    overlay = Image.blend(original_img.convert("RGB"), heatmap, alpha)
    return overlay

def visualize_bounding_box(save_path, img_path, component_coords, culprit_name, analysis_text):
    """(수치 분석용) 결함 부품에 바운딩 박스를 표시하고 이미지를 저장합니다."""
    original_img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    for name, coords_vals in component_coords.items():
        draw.rectangle([coords_vals['x1'], coords_vals['y1'], coords_vals['x2'], coords_vals['y2']], outline="grey", width=2)
    
    if culprit_name and culprit_name in component_coords:
        coords = component_coords[culprit_name]
        box = [coords['x1'], coords['y1'], coords['x2'], coords['y2']]
        draw.rectangle(box, outline="red", width=6)
        draw.text((box[0], box[1] - 35), f"Culprit: {culprit_name}", fill="red", font=font)

    plt.figure(figsize=(12, 12))
    plt.imshow(original_img)
    plt.title("[XAI] Numerical Anomaly Analysis", fontsize=16, weight='bold')
    plt.axis('off')
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, bbox={"facecolor":"#FFDDC1", "alpha":0.7, "pad":10}, wrap=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def visualize_difference_map(save_path, img_path, diff_map, analysis_text):
    """(시각 분석용) Difference Map(Residual)을 원본 이미지 위에 오버레이하여 저장합니다."""
    original_img = Image.open(img_path).convert("RGB")
    
    # diff_map 정규화 (0~1 범위)
    # diff_map에 0으로만 구성된 경우 max-min이 0이 되어 division by zero 발생 가능
    min_val = diff_map.min()
    max_val = diff_map.max()
    if max_val > min_val:
        diff_map_norm = (diff_map - min_val) / (max_val - min_val)
    else:
        diff_map_norm = torch.zeros_like(diff_map)

    overlay_img = overlay_cam_on_image(original_img, diff_map_norm.cpu().detach().numpy(), colormap='inferno', alpha=0.6)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(overlay_img)
    plt.title("[XAI] Visual Anomaly Analysis (Difference Map)", fontsize=16, weight='bold')
    plt.axis('off')
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, bbox={"facecolor":"#D8BFD8", "alpha":0.7, "pad":10}, wrap=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best_model_labeled.pth"
    VIZ_DIR = "anomaly_visualizations_XAI_v10" # 결과 저장 폴더 변경
    if os.path.isfile(VIZ_DIR):
        os.remove(VIZ_DIR)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # --- 독립적 분석을 위한 임계값 설정 ---
    ANOMALY_THRESHOLD = 50.0
    PER_SENSOR_THRESHOLD = 5.0
    # 배경을 제외하고 부품 영역만 오차를 계산하므로 이전보다 임계값을 낮추거나 재조정해야 할 수 있습니다.
    RECON_ERROR_THRESHOLD = 1.0

    EXCEL_FILES = ["D:/archive/Sensor_Data(Label)/EDT/sample/test/Conveyor_Signals_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/FFCell_CycleManagement_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/FFCellSafetyManagement_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/R01_Data_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/R02_Data_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/R03_Data_test.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/test/R04_Data_test.xlsx"]
    JSON_FILE_NORMAL, IMG_DIR_NORMAL = r"D:\archive\Batch 1\data_4000.json", r"D:\archive\Batch 1\BATCH4000"
    JSON_FILE_ERROR, IMG_DIR_ERROR = r"D:\archive\Batch 1\data_27000.json", r"D:\archive\Batch 1\BATCH27000"

    print("Re-fitting scalers and loading model...")
    _, merged_excel_df = load_and_merge_excel(EXCEL_FILES)
    excel_scaler = StandardScaler().fit(merged_excel_df.drop(columns=['time']).values)

    def get_sensors_from_json(path, keys):
        with open(path, 'r') as f:
            data = json.load(f)
        return [[float(v["Sensor_values"].get(k, 0)) for k in keys] for v in data.values()]

    all_continuous_sensors = get_sensors_from_json(JSON_FILE_NORMAL, CONTINUOUS_SENSOR_KEYS) + get_sensors_from_json(JSON_FILE_ERROR, CONTINUOUS_SENSOR_KEYS)
    continuous_scaler = StandardScaler().fit(all_continuous_sensors)
    
    normalized_coords_1 = get_normalized_coords_tensor(COMPONENT_COORDS_1, COMPONENT_ORDER_1, IMAGE_DIMS)
    normalized_coords_0 = get_normalized_coords_tensor(COMPONENT_COORDS_0, COMPONENT_ORDER_0, IMAGE_DIMS)
    
    vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512).to(DEVICE)
    model = VQVAE_ANP_Predictor_V3(
        vqvae, excel_dim=merged_excel_df.shape[1] - 1, coord_dim=COORD_DIM, original_image_dims=IMAGE_DIMS,
        component_coords_0=COMPONENT_COORDS_0, component_order_0=COMPONENT_ORDER_0,
        component_coords_1=COMPONENT_COORDS_1, component_order_1=COMPONENT_ORDER_1,
        num_components=NUM_COMPONENTS, num_continuous=len(CONTINUOUS_SENSOR_KEYS), num_boolean=len(BOOLEAN_SENSOR_KEYS)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model and scalers ready.")

    all_results, y_true, y_pred = [], [], []
    test_targets = [{"json_path": JSON_FILE_NORMAL, "img_dir": IMG_DIR_NORMAL, "actual_label": "Normal"},
                    {"json_path": JSON_FILE_ERROR, "img_dir": IMG_DIR_ERROR, "actual_label": "Anomaly"}]
    
    ORDERED_KEYS = CONTINUOUS_SENSOR_KEYS + BOOLEAN_SENSOR_KEYS

    for target in test_targets:
        with open(target["json_path"], 'r') as f:
            test_data = json.load(f)
        time_keys = sorted(list(test_data.keys()))
        print(f"\nTesting {len(time_keys)} timestamps from {os.path.basename(target['json_path'])} (Label: {target['actual_label']})...")
        for t_key in tqdm(time_keys):
            record = test_data[t_key]
            current_time = pd.to_datetime(t_key)
            
            mask_excel = (merged_excel_df['time'] >= current_time) & (merged_excel_df['time'] < (current_time + pd.Timedelta(milliseconds=500)))
            excel_window = merged_excel_df.loc[mask_excel].drop(columns=['time']).values
            if len(excel_window) == 0:
                excel_window = merged_excel_df.iloc[[(merged_excel_df['time'] - current_time).abs().idxmin()]].drop(columns=['time']).values
            excel_seq_scaled = torch.tensor(excel_scaler.transform(excel_window), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            json_vals = record.get("Sensor_values", {})
            continuous_vals = np.array([float(json_vals.get(k, 0)) for k in CONTINUOUS_SENSOR_KEYS])
            boolean_vals = np.array([float(json_vals.get(k, 0)) for k in BOOLEAN_SENSOR_KEYS])
            
            cont_target = torch.tensor(continuous_scaler.transform(continuous_vals.reshape(1, -1))[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bool_target = torch.tensor(boolean_vals, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            for img_filename in record.get('Images', []):
                img_path = os.path.join(target["img_dir"], os.path.basename(img_filename))
                try:
                    img = Image.open(img_path).convert("RGB")
                except FileNotFoundError:
                    continue

                img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                if img_name_no_ext.endswith('_1'):
                    c_type_val, coords_norm, comp_order, comp_coords = 1, normalized_coords_1, COMPONENT_ORDER_1, COMPONENT_COORDS_1
                elif img_name_no_ext.endswith('_0'):
                    c_type_val, coords_norm, comp_order, comp_coords = 0, normalized_coords_0, COMPONENT_ORDER_0, COMPONENT_COORDS_0
                else:
                    continue
                
                resizer = transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION))
                to_tensor = transforms.ToTensor()
                normalizer = transforms.Normalize([0.5]*3, [0.5]*3)
                img_tensor_unmasked = to_tensor(resizer(img)) # .to(DEVICE)
                map_h, map_w = IMAGE_RESOLUTION, IMAGE_RESOLUTION
                img_w, img_h = IMAGE_DIMS
                
                # Zero-masking을 위한 마스크 생성 (CPU에서)
                bbox_mask = torch.zeros(1, map_h, map_w)
                for name in comp_order:
                    c = comp_coords[name]
                    x1, y1 = int(c['x1'] * map_w / img_w), int(c['y1'] * map_h / img_h)
                    x2, y2 = int(c['x2'] * map_w / img_w), int(c['y2'] * map_h / img_h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(map_w, x2), min(map_h, y2)
                    bbox_mask[:, y1:y2, x1:x2] = 1
                
                masked_img_tensor = img_tensor_unmasked * bbox_mask
                img_tensor = normalizer(masked_img_tensor).unsqueeze(0).to(DEVICE)

                c_type = torch.tensor([c_type_val], dtype=torch.long, device=DEVICE)
                coords = coords_norm.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    mu, sigma, bool_logits = model(img_tensor, excel_seq_scaled, coords, c_type)
                    reconstructed_img, _ = model.vqvae(img_tensor)

                loss_cont = gaussian_nll(mu, sigma, cont_target)
                loss_bool = F.binary_cross_entropy_with_logits(bool_logits, bool_target)
                nll_score = (loss_cont + loss_bool).item()
                
                # --- 수정된 복원 오차 계산 ---
                # 픽셀별 제곱 오차 계산
                pixel_squared_error = (img_tensor - reconstructed_img).pow(2)
                # 바운딩 박스 마스크를 이용해 배경 오차는 0으로 만듦
                masked_squared_error = pixel_squared_error * bbox_mask.to(DEVICE)
                # 마스크가 적용된 영역의 픽셀 수
                num_masked_pixels = bbox_mask.sum()
                if num_masked_pixels > 0:
                    # 오직 부품 영역에 대해서만 평균 오차를 계산
                    recon_error = masked_squared_error.sum() / (num_masked_pixels * img_tensor.shape[1])
                    recon_error = recon_error.item()
                else:
                    recon_error = 0.0

                predicted_label_str = "Anomaly" if nll_score > ANOMALY_THRESHOLD else "Normal"
                description = "Normal"

                if predicted_label_str == "Anomaly" and target['actual_label'] == "Anomaly":
                    has_numerical_anomaly, has_visual_anomaly = False, False
                    numerical_reason, visual_reason = "", ""
                    numerical_culprit = None

                    # 1. 시각적 이상 분석 (수정된 복원 오차 기준)
                    if recon_error > RECON_ERROR_THRESHOLD:
                        has_visual_anomaly = True
                        # 시각화를 위한 diff_map은 마스크 영역만 계산
                        diff_map = (img_tensor.squeeze(0) - reconstructed_img.squeeze(0)).abs().mean(dim=0) * bbox_mask.squeeze(0).to(DEVICE)
                        visual_reason = f"ROOT CAUSE (VISUAL)\n- High reconstruction error in component regions ({recon_error:.4f}).\n- VAE failed to reconstruct highlighted parts."

                    # 2. 수치적 이상 분석 (기존과 동일)
                    mu_orig = continuous_scaler.inverse_transform(mu.cpu().detach().numpy())
                    probs_orig = torch.sigmoid(bool_logits).cpu().detach().numpy()
                    cont_actual_orig = continuous_scaler.inverse_transform(cont_target.cpu().numpy())
                    all_pred = np.concatenate([mu_orig.flatten(), probs_orig.flatten()])
                    all_actual = np.concatenate([cont_actual_orig.flatten(), bool_target.cpu().numpy().flatten()])
                    diffs = np.abs(all_pred - all_actual)
                    max_diff_idx, max_diff_value = np.argmax(diffs), np.max(diffs)
                    
                    if max_diff_value > PER_SENSOR_THRESHOLD:
                        has_numerical_anomaly = True
                        max_diff_sensor_name = ORDERED_KEYS[max_diff_idx]
                        numerical_culprit = map_sensor_to_component(max_diff_sensor_name, comp_order)
                        reason_text = f"ROOT CAUSE (NUMERICAL)\n- Culprit Sensor: '{max_diff_sensor_name}' (Deviation: {max_diff_value:.4f})"
                        if numerical_culprit:
                            numerical_reason = f"{reason_text}\n- Associated Component: '{numerical_culprit}'"
                        else:
                            numerical_reason = f"{reason_text}\n- Component not visible in this view."

                    # 3. 분석 결과 시각화 및 Description 생성
                    sanitized_t_key = t_key.replace(":", "-").replace(" ", "_").replace(".", "_")
                    base_save_name = f"xai_anomaly_{sanitized_t_key}_{img_name_no_ext}"
                    header = f"Time: {t_key} | Image: {os.path.basename(img_path)} | Score: {nll_score:.2f}"
                    
                    if has_visual_anomaly:
                        save_path_vis = os.path.join(VIZ_DIR, f"{base_save_name}_visual_diff.png")
                        # visualize_difference_map(save_path_vis, img_path, diff_map, f"{header}\n\n{visual_reason}")
                        # print(f"  -> Visual anomaly detected (Recon Error: {recon_error:.4f}). Diff map saved to {save_path_vis}")

                    if has_numerical_anomaly:
                        save_path_num = os.path.join(VIZ_DIR, f"{base_save_name}_numerical.png")
                        # visualize_bounding_box(save_path_num, img_path, comp_coords, numerical_culprit, f"{header}\n\n{numerical_reason}")
                        # print(f"  -> Numerical anomaly detected (Sensor Dev: {max_diff_value:.4f}). BBox saved to {save_path_num}")
                    
                    desc_parts = []
                    if has_numerical_anomaly:
                        culprit_str = f"({numerical_culprit})" if numerical_culprit else "(Not in view)"
                        desc_parts.append(f"Numerical{culprit_str}")
                    if has_visual_anomaly:
                        desc_parts.append("Visual")

                    if desc_parts:
                        description = " & ".join(desc_parts)
                    elif predicted_label_str == "Anomaly":
                        description = "Complex Anomaly (High NLL Score)"

                y_true.append(1 if target['actual_label'] == "Anomaly" else 0)
                y_pred.append(1 if predicted_label_str == "Anomaly" else 0)
                all_results.append({"time": t_key, "image_file": os.path.basename(img_path), "score": nll_score, "recon_error": recon_error, "Predicted_Label": predicted_label_str, "Actual_Label": target['actual_label'], "description": description})

    print("\n--- Model Performance Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    df_results = pd.DataFrame(all_results)
    output_filename = "detection_results.xlsx"
    df_results.to_excel(output_filename, index=False)
    print(f"\n--- Detection Summary ---\n{df_results['Predicted_Label'].value_counts()}\n\nResults saved to {output_filename}")
