import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
from tqdm import tqdm
import os 

class ModelTrainer:
    def __init__(self, model, train_data_loader, val_data_loader, config):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.train_loss_tracker = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        
        # 최적의 Validation Loss를 추적하기 위한 변수 초기화
        self.best_val_loss = float('inf')
        
        # [추가] Early Stopping을 위한 변수
        self.patience = 5  # 개선이 없어도 기다려줄 횟수
        self.wait = 0      # 현재 개선되지 않은 연속 횟수 카운터

        # 그래프 작성을 위해 에포크별 Loss를 저장할 리스트
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self):
        for epoch in range(self.config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            self.train_epoch()
            self.validate_epoch()
            
            # 현재 에포크의 Loss 값 가져오기
            current_train_loss = self.train_loss_tracker.result().numpy()
            current_val_loss = self.val_loss_tracker.result().numpy()

            print(f"Train Loss: {current_train_loss:.4f}, "
                  f"Validation Loss: {current_val_loss:.4f}")

            # [수정] Best Model 저장 및 Early Stopping 로직 적용
            if current_val_loss < self.best_val_loss:
                print(f"✅ Validation Loss improved from {self.best_val_loss:.4f} to {current_val_loss:.4f}. Saving model...")
                self.best_val_loss = current_val_loss
                
                # Validation Loss가 개선되었으므로 기다림 카운터 초기화
                self.wait = 0 
                
                # 저장 경로 설정
                save_path = os.path.join(self.config['result_dir'], 'best_model_vl.weights.h5')
                self.model.save_weights(save_path)
            else:
                # Validation Loss가 개선되지 않음
                self.wait += 1
                print(f"⚠️ Validation Loss did not improve. (Wait: {self.wait}/{self.patience})")

            # 그래프를 위해 기록 저장
            self.train_loss_history.append(current_train_loss)
            self.val_loss_history.append(current_val_loss)

            # self.plot_reconstructed_signal(epoch)
            # self.plot_train_and_val_loss(epoch)
            
            self.train_loss_tracker.reset_state()
            self.val_loss_tracker.reset_state()

            # [추가] Early Stopping 체크
            if self.wait >= self.patience:
                print(f"\n🛑 Early Stopping Triggered! Validation loss did not improve for {self.patience} consecutive epochs.")
                print(f"Training stopped at Epoch {epoch + 1}.")
                break

    def train_epoch(self):
        total_batches = (len(self.train_data_loader.sequences) + self.config['batch_size'] - 1) // self.config['batch_size']
        
        with tqdm(self.train_data_loader.get_dataset(), desc="Training", total=total_batches) as pbar:
            for batch in pbar:
                self.train_step(batch)
                current_loss = self.train_loss_tracker.result().numpy()
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

    def train_step(self, batch):
        sensor_signal, image_signal, _, _ = batch = batch
        with tf.GradientTape() as tape:
            decoded_signal = self.model([sensor_signal, image_signal])
            loss, _, _ = self.model.define_loss(sensor_signal, decoded_signal)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_tracker.update_state(loss)

    def validate_epoch(self):
        total_batches = (len(self.val_data_loader.sequences) + self.config['batch_size'] - 1) // self.config['batch_size']
        
        with tqdm(self.val_data_loader.get_dataset(), desc="Validating", total=total_batches) as pbar:
            for batch in pbar:
                self.val_step(batch)
                current_val_loss = self.val_loss_tracker.result().numpy()
                pbar.set_postfix({'val_loss': f'{current_val_loss:.4f}'})

    def val_step(self, batch):
        sensor_signal, image_signal, _, _ = batch = batch
        decoded_signal = self.model([sensor_signal, image_signal])
        loss, _, _ = self.model.define_loss(sensor_signal, decoded_signal)
        self.val_loss_tracker.update_state(loss)
    
    # def plot_reconstructed_signal(self, epoch):
    #     # 데이터셋에서 배치 하나 가져오기 (iter 재성성 방지를 위해 주의 필요하지만, 여기선 단순화)
    #     # tf.data.Dataset은 반복 가능하므로 매번 새로 호출됨
    #     try:
    #         batch = next(iter(self.val_data_loader.get_dataset()))
    #     except StopIteration:
    #         return 

    #     sensor_signal, image_signal, _, _ = batch

    #     decoded_signal = self.model([sensor_signal, image_signal])
    #     n_signals = min(10, self.config['batch_size'])
        
    #     for j in range(sensor_signal.shape[-1]):
    #         fig, axs = plt.subplots(2, 5, figsize=(15, 6), edgecolor='k')
    #         fig.subplots_adjust(hspace=.4, wspace=.4)
    #         axs = axs.ravel()
    #         for i in range(n_signals):
    #             # zero-padding 제거 후 plotting (원본 코드 로직 유지)
    #             input_len = np.trim_zeros(sensor_signal[i, :, j], 'b').shape[0]
    #             # 데이터가 전부 0인 경우 shape이 0일 수 있으므로 예외 처리 혹은 원본 길이 사용 권장
    #             if input_len == 0: input_len = sensor_signal.shape[1]

    #             axs[i].plot(sensor_signal[i, :input_len, j])
    #             axs[i].plot(decoded_signal[i, :input_len, j])
    #             axs[i].grid(True)
    #             axs[i].set_title(f'Sample {i}')
    #             if i == 0:
    #                 axs[i].legend(('Original', 'Reconstructed'))
            
    #         plt.suptitle(f'Epoch {epoch + 1} - Channel {j} Reconstruction')
    #         save_path = os.path.join(self.config['result_dir'], f'reconstruction_epoch_{epoch + 1}_ch{j}.pdf')
    #         # savefig(save_path)
    #         fig.clf()
    #         plt.close()

    # def plot_train_and_val_loss(self, epoch):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.train_loss_history, label='Train Loss')
    #     plt.plot(self.val_loss_history, label='Validation Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
        
    #     save_path = os.path.join(self.config['result_dir'], f'loss_graph_epoch_{epoch + 1}.pdf')
    #     # savefig(save_path)
    #     plt.close()