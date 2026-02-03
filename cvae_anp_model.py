# ANP와 CVAE의 수정된 모듈을 가져옵니다.
from ANP.anp_attn_log import NeuralProcess
from VAE.cvae_v2_fix import ConditionalVAE_v2
import torch.nn as nn
import torch

class AttentiveCVAEANP(nn.Module):
    def __init__(self, cvae_latent_dim, condition_dim, x_dim, hidden_dim, anp_latent_dim, img_size=256, beta=1.0):
        super().__init__()
        self.beta = beta  # beta 값을 인스턴스 변수로 저장

        # 1. 수정된 CVAE 모듈 인스턴스화
        # cvae_v2_fix.py의 ConditionalVAE_v2는 hidden_dims를 멤버 변수로 가집니다.
        self.cvae = ConditionalVAE_v2(in_channels=3, 
                                      condition_dim=condition_dim, 
                                      latent_dim=cvae_latent_dim, 
                                      img_size=img_size)
        
        # CVAE 인코더의 마지막 채널 수를 이미지 피쳐 차원으로 사용
        image_feature_dim = self.cvae.hidden_dims[-1]

        # 2. 수정된 ANP 모듈 인스턴스화
        # anp_attn_log.py의 NeuralProcess는 image_feature_dim 인자를 받습니다.
        self.anp = NeuralProcess(x_dim=x_dim, 
                                 y_dim=1, 
                                 hidden_dim=hidden_dim, 
                                 latent_dim=anp_latent_dim, 
                                 latent_enc_self_attn_type="ptmultihead", 
                                 det_enc_self_attn_type="ptmultihead", 
                                 det_enc_cross_attn_type="ptmultihead",
                                 image_attention_type="ptmultihead", # 이미지 크로스 어텐션 타입
                                 use_self_attn=True, 
                                 global_context_dim=cvae_latent_dim,
                                 image_feature_dim=image_feature_dim) # 이미지 피쳐 차원 전달

    def forward(self, image, condition, context_x, context_y, target_x, target_y=None):
        # 1. CVAE로 이미지에서 글로벌 컨텍스트(img_mu)와 피쳐 맵(features) 추출
        # cvae_v2_fix.py의 forward는 5개의 값을 반환합니다.
        cvae_output = self.cvae(image, condition=condition)
        img_mu = cvae_output[2]
        features = cvae_output[4] # [B, C, H, W] 형태의 피쳐 맵
        
        # CVAE 손실 계산
        cvae_loss_dict = self.cvae.loss_function(*cvae_output[:4])

        # beta 가중치를 적용하여 CVAE 손실 재계산
        # self.cvae.loss_function이 'Reconstruction_Loss'와 'KLD'를 포함한 딕셔너리를 반환한다고 가정합니다.
        if 'Reconstruction_Loss' in cvae_loss_dict and 'KLD' in cvae_loss_dict:
            cvae_loss_dict['loss'] = cvae_loss_dict['Reconstruction_Loss'] + self.beta * cvae_loss_dict['KLD']

        # 2. ANP로 예측 수행
        # anp_attn_log.py의 forward는 image_features 인자를 추가로 받습니다.
        # 또한 이제 6개의 값을 반환합니다.
        _, anp_loss_dict, anp_misc_outputs, latent_self_attn, image_cross_attn, det_attn_weights = self.anp(
            context_x, 
            context_y, 
            target_x, 
            target_y, 
            global_context=img_mu,
            image_features=features # 추출한 피쳐 맵을 전달
        )
        
        # 3. 최종 결과 반환
        # y_dist: 예측 분포
        # anp_loss_dict, cvae_loss_dict: 각 모델의 손실
        # latent_self_attn: ANP의 LatentEncoder에서 나온 센서 간의 self-attention 가중치
        # image_cross_attn: Decoder에서 나온 이미지 피쳐에 대한 cross-attention 가중치
        # det_attn_weights: DeterministicEncoder에서 나온 self-attention 및 cross-attention 가중치
        return anp_misc_outputs['y_dist'], anp_loss_dict, cvae_loss_dict, latent_self_attn, image_cross_attn, det_attn_weights
