import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from VAE.types import *


class ConditionalVAE_v2(BaseVAE):
    """
    This CVAE is redesigned to be conditioned on a multi-dimensional vector
    (derived from time-series data) instead of a simple class label.
    """
    def __init__(self,
                 in_channels: int,
                 condition_dim: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE_v2, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.condition_dim = condition_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Build Encoder
        encoder_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            encoder_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(hidden_dims[-1] * (img_size // 2**len(hidden_dims))**2 + condition_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (img_size // 2**len(hidden_dims))**2 + condition_dim, latent_dim)


        # Build Decoder
        modules = []
        
        final_encoder_size = img_size // (2**len(hidden_dims))
        self.decoder_input = nn.Linear(latent_dim + condition_dim, hidden_dims[-1] * (final_encoder_size ** 2))

        reversed_hidden_dims = hidden_dims.copy()
        reversed_hidden_dims.reverse()

        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i],
                                       reversed_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(reversed_hidden_dims[-1],
                                               reversed_hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(reversed_hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(reversed_hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor, condition_vec: Tensor) -> List[Tensor]:
        """
        Encodes the input, returns latent codes and the feature map.
        """
        features = self.encoder(input)
        result = torch.flatten(features, start_dim=1)

        combined = torch.cat([result, condition_vec], dim=1)

        mu = self.fc_mu(combined)
        log_var = self.fc_var(combined)

        return [mu, log_var, features]

    def decode(self, z: Tensor, condition_vec: Tensor) -> Tensor:
        """
        Decodes the latent vector z, conditioned on condition_vec.
        """
        z_cond = torch.cat([z, condition_vec], dim=1)
        
        result = self.decoder_input(z_cond)

        final_encoder_size = self.img_size // (2**len(self.encoder))
        num_channels = self.decoder_input.out_features // (final_encoder_size ** 2)

        result = result.view(-1, num_channels, final_encoder_size, final_encoder_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        condition_vec = kwargs.get('condition', None)

        if condition_vec is None:
            batch_size = input.shape[0]
            condition_vec = torch.zeros(batch_size, self.condition_dim, device=input.device)

        mu, log_var, features = self.encode(input, condition_vec)
        z = self.reparameterize(mu, log_var)

        reconstruction = self.decode(z, condition_vec)
        return [reconstruction, input, mu, log_var, features]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        # 'features' is args[4] but not used in loss

        kld_weight = kwargs.get('M_N', 1.0)
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        condition_seq = kwargs['condition']
        condition_vec = torch.mean(condition_seq, dim=1)

        if num_samples > condition_vec.shape[0]:
            repeats = num_samples // condition_vec.shape[0] + 1
            condition_vec = condition_vec.repeat(repeats, 1)[:num_samples]

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z, condition_vec)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]
