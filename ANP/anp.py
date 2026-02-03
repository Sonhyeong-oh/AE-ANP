import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math
import numpy as np
from ANP.atten import Attention
from ANP.modules import BatchNormSequence, BatchMLP, LSTMBlock

from ANP.utils import kl_loss_var, log_prob_sigma
from ANP.utils import hparams_power

class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=3,
        min_std=0.01,
        batchnorm=False,
        dropout=0.2,
        attention_dropout=0,
        use_lvar=False,
        use_self_attn=False,
        attention_layers=2,
        use_lstm=False
    ):
        super().__init__()
        if use_lstm:
            self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_encoder_layers)
        else:
            self._encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self._min_std = min_std
        self._use_lvar = use_lvar
        self._use_lstm = use_lstm
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)
        encoded = self._encoder(encoder_input)

        attention_weights = None
        if self._use_self_attn:
            attention_output, attention_weights = self._self_attention(encoded, encoded, encoded)
            mean_repr = attention_output.mean(dim=1)
        else:
            mean_repr = encoded.mean(dim=1)

        mean_repr = torch.relu(self._penultimate_layer(mean_repr))
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        if self._use_lvar:
            log_var = F.logsigmoid(log_var)
            log_var = torch.clamp(log_var, np.log(self._min_std), -np.log(self._min_std))
            sigma = torch.exp(0.5 * log_var)
        else:
            sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_var * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var, attention_weights


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=3,
        self_attention_type="dot",
        cross_attention_type="dot",
        use_self_attn=False,
        attention_layers=2,
        batchnorm=False,
        dropout=0.2,
        attention_dropout=0,
        use_lstm=False,
    ):
        super().__init__()
        self._use_self_attn = use_self_attn
        if use_lstm:
            self._d_encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        else:
            self._d_encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            attention_layers=attention_layers,
        )

    def forward(self, context_x, context_y, target_x):
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)
        d_encoded = self._d_encoder(d_encoder_input)

        self_attn_weights = None
        if self._use_self_attn:
            d_encoded, self_attn_weights = self._self_attention(d_encoded, d_encoded, d_encoded)

        h, cross_attn_weights = self._cross_attention(context_x, d_encoded, target_x)
        
        det_attn_weights = {"self": self_attn_weights, "cross": cross_attn_weights}
        
        return h, det_attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        use_deterministic_path=True,
        use_latent_path=True,
        min_std=0.01,
        use_lvar=False,
        batchnorm=False,
        dropout=0.2,
        use_lstm=False,
        global_context_dim=0,
        image_feature_dim=0,
        image_attention_type="ptmultihead",
        attention_layers=2,
        attention_dropout=0
    ):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)

        self._use_deterministic_path = use_deterministic_path
        self._use_latent_path = use_latent_path
        self._use_image_attention = image_feature_dim > 0
        self._min_std = min_std
        self._use_lvar = use_lvar
        
        decoder_input_dim = hidden_dim  # For target_x
        if use_deterministic_path:
            decoder_input_dim += hidden_dim
        if use_latent_path:
            decoder_input_dim += latent_dim
        if global_context_dim > 0:
            decoder_input_dim += global_context_dim
        if self._use_image_attention:
            self._image_feature_projection = nn.Linear(image_feature_dim, hidden_dim)
            self._image_cross_attention = Attention(
                hidden_dim,
                image_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout
            )
            decoder_input_dim += hidden_dim # For image context

        if use_lstm:
            self._decoder = LSTMBlock(decoder_input_dim, decoder_input_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        else:
            self._decoder = BatchMLP(decoder_input_dim, decoder_input_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        
        self._mean = nn.Linear(decoder_input_dim, y_dim)
        self._std = nn.Linear(decoder_input_dim, y_dim)

    def forward(self, r, z, target_x, global_context=None, image_features=None):
        reps = []
        image_attention_weights = None

        if self._use_deterministic_path and r is not None:
            reps.append(r)
        if self._use_latent_path and z is not None:
            reps.append(z)
        
        x_transformed = self._target_transform(target_x) # This is the Query for image cross-attention
        
        if self._use_image_attention and image_features is not None:
            # Reshape image features from [B, C, H, W] to [B, S, C] where S=H*W
            batch_size, num_channels, height, width = image_features.shape
            reshaped_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            
            # Project features to hidden_dim to be used as Key and Value
            projected_features = self._image_feature_projection(reshaped_features)

            # Perform cross-attention
            image_context, image_attention_weights = self._image_cross_attention(
                k=projected_features, v=projected_features, q=x_transformed
            )
            reps.append(image_context)
            
        reps.append(x_transformed) # Append query after it has been used
        
        if global_context is not None:
            reps.append(global_context)

        decoder_input = torch.cat(reps, dim=-1)
        hidden = self._decoder(decoder_input)

        mean = self._mean(hidden)
        log_sigma = self._std(hidden)

        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_sigma, image_attention_weights


class NeuralProcess(nn.Module):

    @staticmethod
    def FROM_HPARAMS(hparams):
        hparams = hparams_power(hparams)
        return NeuralProcess(**hparams)

    def __init__(self,
                 x_dim, 
                 y_dim, 
                 hidden_dim=32,
                 latent_dim=32,
                 use_latent_path=True,
                 latent_enc_self_attn_type="ptmultihead",
                 det_enc_self_attn_type="ptmultihead",
                 det_enc_cross_attn_type="ptmultihead",
                 image_attention_type="ptmultihead", # New
                 n_latent_encoder_layers=2,
                 n_det_encoder_layers=2,
                 n_decoder_layers=2,
                 use_deterministic_path=True,
                 min_std=0.01,
                 dropout=0.2,
                 use_self_attn=False,
                 attention_dropout=0,
                 batchnorm=False,
                 use_lvar=False,
                 attention_layers=2,
                 use_rnn=True,
                 use_lstm_le=False,
                 use_lstm_de=False,
                 use_lstm_d=False,
                 context_in_target=False,
                 global_context_dim=0,
                 image_feature_dim=0, # New
                 **kwargs,
                ):

        super(NeuralProcess, self).__init__()

        self._use_rnn = use_rnn
        self.context_in_target = context_in_target
        self._use_latent_path = use_latent_path
        self._use_deterministic_path = use_deterministic_path
        self._latent_dim = latent_dim

        self.norm_x = BatchNormSequence(x_dim, affine=False)
        self.norm_y = BatchNormSequence(y_dim, affine=False)

        if self._use_rnn:
            self._lstm_x = nn.LSTM(input_size=x_dim, hidden_size=hidden_dim, num_layers=attention_layers, dropout=dropout, batch_first=True)
            self._lstm_y = nn.LSTM(input_size=y_dim, hidden_size=hidden_dim, num_layers=attention_layers, dropout=dropout, batch_first=True)
            x_dim_encoded = hidden_dim
            y_dim_encoded = hidden_dim
        else:
            x_dim_encoded = x_dim
            y_dim_encoded = y_dim

        if self._use_latent_path:
            self._latent_encoder = LatentEncoder(
                x_dim_encoded + y_dim_encoded, hidden_dim, latent_dim, latent_enc_self_attn_type, 
                n_latent_encoder_layers, min_std, batchnorm, dropout, attention_dropout, 
                use_lvar, use_self_attn, attention_layers, use_lstm_le)

        if self._use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
                input_dim=x_dim_encoded + y_dim_encoded, x_dim=x_dim_encoded, hidden_dim=hidden_dim,
                n_d_encoder_layers=n_det_encoder_layers, self_attention_type=det_enc_self_attn_type,
                cross_attention_type=det_enc_cross_attn_type, use_self_attn=use_self_attn,
                attention_layers=attention_layers, batchnorm=batchnorm, dropout=dropout,
                attention_dropout=attention_dropout, use_lstm=use_lstm_de)

        self._decoder = Decoder(
            x_dim_encoded, y_dim, hidden_dim, latent_dim, n_decoder_layers, use_deterministic_path,
            use_latent_path, min_std, use_lvar, batchnorm, dropout, use_lstm_d,
            global_context_dim, image_feature_dim, image_attention_type, attention_layers, attention_dropout)
            
        self._use_lvar = use_lvar

    def forward(self, context_x, context_y, target_x, target_y=None, sample_latent=None, global_context=None, image_features=None):
        if sample_latent is None: sample_latent = self.training
        device = next(self.parameters()).device

        target_x = self.norm_x(target_x)
        context_x = self.norm_x(context_x)
        context_y = self.norm_y(context_y)

        if self._use_rnn:
            target_x, _ = self._lstm_x(target_x)
            context_x_encoded, _ = self._lstm_x(context_x)
            context_y_encoded, _ = self._lstm_y(context_y)
        else:
            context_x_encoded, context_y_encoded = context_x, context_y

        latent_self_attention_weights = None
        if self._use_latent_path:
            dist_post, log_var_post, latent_self_attention_weights = self._latent_encoder(context_x_encoded, context_y_encoded)
            prior_mu = torch.zeros_like(dist_post.loc)
            prior_sigma = torch.ones_like(dist_post.scale)
            dist_prior = torch.distributions.Normal(prior_mu, prior_sigma)
            z = dist_post.rsample() if sample_latent else dist_post.loc
            z = z.unsqueeze(1).repeat(1, target_x.size(1), 1)
        else:
            z, dist_post, dist_prior = None, None, None

        det_attn_weights = None
        if self._use_deterministic_path:
            r, det_attn_weights = self._deterministic_encoder(context_x_encoded, context_y_encoded, target_x)
        else:
            r = None

        if global_context is not None:
            global_context = global_context.unsqueeze(1).repeat(1, target_x.size(1), 1)

        dist, log_sigma, image_cross_attention_weights = self._decoder(r, z, target_x, global_context=global_context, image_features=image_features)
        
        if target_y is not None:
            if self._use_latent_path and dist_post is not None:
                if self._use_lvar:
                    loss_kl = kl_loss_var(dist_post.loc, log_var_post, torch.zeros_like(dist_post.loc), torch.zeros_like(log_var_post)).sum(-1)
                else:
                    loss_kl = torch.distributions.kl_divergence(dist_post, dist_prior).sum(-1)
            else:
                loss_kl = torch.tensor(0.0, device=device)

            log_p = dist.log_prob(target_y).mean(-1)
            if self.context_in_target:
                log_p[:, :context_x.size(1)] /= 100
            
            loss_p = -log_p.mean()
            loss = loss_p + loss_kl.mean()
            mse_loss = F.mse_loss(dist.loc, target_y, reduction='none')[:,:context_x.size(1)].mean()
            loss_kl_mean, log_p_mean = loss_kl.mean(), log_p.mean()
        else:
            loss, loss_p, loss_kl_mean, log_p_mean, mse_loss = None, None, None, None, None

        y_pred = dist.rsample() if self.training else dist.loc
        return y_pred, dict(loss=loss, loss_p=loss_p, loss_kl=loss_kl_mean, loss_mse=mse_loss), dict(log_sigma=log_sigma, y_dist=dist), latent_self_attention_weights, image_cross_attention_weights, det_attn_weights
