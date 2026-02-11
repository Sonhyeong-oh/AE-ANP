import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tfd = tfp.distributions

class VAELSTMModel(tf.keras.Model):
    def __init__(self, config, input_dims):
        super(VAELSTMModel, self).__init__()
        self.config = config
        self.input_dims = input_dims
        self.encoder = self._build_encoder()

        # Define decoder layers directly to handle variable length sequences
        self.decoder_lstm_1 = tf.keras.layers.LSTM(units=self.config['num_hidden_units'], return_sequences=True)
        self.decoder_lstm_2 = tf.keras.layers.LSTM(units=self.config['num_hidden_units'], return_sequences=True)
        self.decoder_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=self.input_dims, activation=None)
        )

        if self.config['TRAIN_sigma'] == 1:
            self.sigma = tf.Variable(tf.cast(self.config['sigma'], tf.float32),
                                     dtype=tf.float32, trainable=True)
        else:
            self.sigma = tf.cast(self.config['sigma'], tf.float32)
        self.sigma2_offset = tf.constant(self.config['sigma2_offset'])

    # models.py 수정 예시 (개념적 코드)
    def _build_encoder(self):
        # 1. 센서 입력 (기존)
        sensor_input = tf.keras.layers.Input(shape=(None, self.input_dims), name='sensor_input')
        x_sensor = tf.keras.layers.Masking(mask_value=0.0)(sensor_input)
        x_sensor = tf.keras.layers.LSTM(units=self.config['num_hidden_units'])(x_sensor)

        # 2. 이미지 입력
        # MobileNetV2의 기본 권장 크기인 (256, 256) 근처로 줄이면 속도가 비약적으로 상승합니다.
        image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image_input') 
        
        # [중요] Preprocessing (MobileNetV2 전용 전처리: 픽셀 값을 -1 ~ 1로 맞춤)
        x_img_preprocessed = preprocess_input(image_input)
        
        # 3. MobileNetV2로 특징 추출 (가벼운 모델)
        # weights='imagenet': ImageNet 데이터로 미리 학습된 가중치를 가져옵니다.
        base_model = MobileNetV2(input_shape=(224, 224, 3),
                                 include_top=False, 
                                 weights='imagenet')
        
        # 가중치 고정 (Feature Extractor로만 사용)
        base_model.trainable = False
        
        x_img = base_model(x_img_preprocessed)
        x_img = GlobalAveragePooling2D()(x_img) # (Batch, 1280) 벡터로 변환
        x_img = tf.keras.layers.Dense(units=128, activation='relu')(x_img) # 차원 축소

        # 3. 멀티모달 융합 (Concatenate)
        # 센서 특징 벡터와 이미지 특징 벡터를 합칩니다.
        combined = tf.keras.layers.Concatenate()([x_sensor, x_img])

        # 4. Latent Space 생성 (기존 로직 연결)
        encoded_signal = tf.keras.layers.Dense(units=self.config['num_hidden_units'], activation='tanh')(combined)

        code_mean = tf.keras.layers.Dense(self.config['code_size'])(encoded_signal)
        code_std_dev = tf.keras.layers.Dense(self.config['code_size'], activation=tf.nn.softplus)(encoded_signal)

        # 입력이 2개인 모델로 리턴
        return tf.keras.Model([sensor_input, image_input], [code_mean, code_std_dev], name='encoder')

    def call(self, inputs, is_code_input=False, code_input=None):
        if is_code_input:
            encoded = code_input
        else:
            sensor_data, image_data = inputs
            self.code_mean, self.code_std_dev = self.encoder([sensor_data, image_data])
            mvn = tfp.distributions.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev)
            self.code_sample = mvn.sample()
            encoded = self.code_sample

        # Dynamic decoding using tf.tile to match input sequence length
        input_seq_len = tf.shape(sensor_data)[1]
        encoded_expanded = tf.expand_dims(encoded, 1)
        decoded_repeated = tf.tile(encoded_expanded, [1, input_seq_len, 1])

        lstm_dec_1_out = self.decoder_lstm_1(decoded_repeated)
        lstm_dec_2_out = self.decoder_lstm_2(lstm_dec_1_out)
        decoded = self.decoder_dense(lstm_dec_2_out)
        
        if self.config['TRAIN_sigma'] == 1:
            self.sigma2 = tf.square(self.sigma) + self.sigma2_offset
        else:
            self.sigma2 = tf.square(self.sigma)
            
        return decoded

    def define_loss(self, original_signal, decoded_signal):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(original_signal - decoded_signal),
                axis=[1, 2]
            )
        )

        kl_div_loss = tfd.kl_divergence(
            tfd.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev),
            tfd.MultivariateNormalDiag(loc=tf.zeros_like(self.code_mean), scale_diag=tf.ones_like(self.code_std_dev))
        )
        kl_div_loss = tf.reduce_mean(kl_div_loss)
        
        cost = 0.5 * reconstruction_loss + 0.5 * kl_div_loss
        return cost, reconstruction_loss, kl_div_loss