# AE-ANP
주제 : 오토인코더-ANP 결합 모델을 통한 공정 이상 예측

저자 : 손형오, 최우성, 이영재, 김영균

# Dataset
<img width="840" height="560" alt="image" src="https://github.com/user-attachments/assets/e814ac95-1f3d-4e4e-ab59-6fb2df2f3382" />

This dataset is one of two industry-grade datasets captured during an 8-hour continuous operation of the manufacturing assembly line at the Future Factories Lab, University of South Carolina, on 08/13/2024.
* Link : https://www.kaggle.com/datasets/ramyharik/ff-2024-08-13-multi-modal-dataset-13

## Dataset in Google Drive
* Multimodal Data Link : https://drive.google.com/drive/folders/1qkE7suLerbn4aBD4ZFyMlKmlNV9bkUIQ
* Sensor Data Link (with Label) : https://drive.google.com/drive/folders/1R71k4fW60_n5c6xpqV3_a8dTtf7nWYnK

# Code Description

1. ANP: Probabilistic time-series modeling for high-frequency sensor data prediction.

2. VAE: High-level visual feature extraction and latent representation of process images.

3. grad-CAM: Visual explanation and evidence localization for anomaly detection via Grad-CAM & Eigen-CAM.

4. train_v2: Multimodal fusion and joint training using visual and sensor information for VAE-ANP combined model.

5. XAI_test10: Anomaly detection evaluation and automated root cause analysis.

# AE(AutoEncoder)
* Reference Code Link 1 : https://github.com/AntixK/PyTorch-VAE (VAE, CVAE)
* Reference Code Link 2 : https://github.com/rosinality/vq-vae-2-pytorch (VQ-VAE)
* Reference Paper (VAE) : https://arxiv.org/abs/1312.6114
* Reference Paper (CVAE) : https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf

# ANP(Attentive Neural Process)
* Reference Code Link : https://github.com/wassname/attentive-neural-processes
* Reference Paper : https://arxiv.org/abs/1901.05761 (Hyunjik Kim, et al., "Attentive Neural Processes", 2019)

# Grad-CAM & Eigen-CAM
* Reference Code Link : https://github.com/jacobgil/pytorch-grad-cam
