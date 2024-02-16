import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from clustering import compute_jaccard_similarity
from utils import load_blendshape

# Utility: one-hot encoding function
def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Flatten to 1D
    if indices.dim() == 2 and indices.size(1) == 1:
        indices = indices.squeeze(1)
    return F.one_hot(indices, num_classes)

# Utility: KL divergence between q(z|x)=N(mu,var) and p(z)=N(0,1)
def KL_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute batch-wise KL divergence D_KL(q(z|x) || N(0,1)).
    Args:
        mu:    [batch, latent_dim] mean of approximate posterior
        logvar: [batch, latent_dim] log-variance of approximate posterior
    Returns:
        KL divergence per sample [batch]
    """
    # KL = -0.5 * sum(1 + logvar - mu^2 - var)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

class NeuralFaceControllerCVAE(nn.Module):
    def __init__(self,
                 n_features: int,
                 affinity_matrix: torch.Tensor,
                 hidden_features: int = 64,
                 num_encoder_layers: int = 4,
                 latent_dim: int = 5,
                 num_decoder_layers: int = 4,
                 nonlinearity: str = 'ReLU'):
        super().__init__()
        # Validate affinity
        assert affinity_matrix.shape == (n_features, n_features), \
            f"Affinity must be [{n_features},{n_features}]"
        self.n_features = n_features
        self.affinity = affinity_matrix

        # Nonlinearity
        nls = {'ReLU': nn.ReLU(), 'ELU': nn.ELU()}
        nl = nls[nonlinearity]

        # Encoder: stacks of Linear+NL -> two heads (mu, logvar)
        enc_layers = []
        enc_layers.append(nn.Linear(n_features, hidden_features))
        enc_layers.append(nl)
        for _ in range(num_encoder_layers - 1):
            enc_layers.append(nn.Linear(hidden_features, hidden_features))
            enc_layers.append(nl)
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu  = nn.Linear(hidden_features, latent_dim)
        self.fc_logvar = nn.Linear(hidden_features, latent_dim)

        # Decoder: input is [z, alpha_scalar, id_onehot]
        dec_input = latent_dim + 1 + n_features
        dec_layers = []
        dec_layers.append(nn.Linear(dec_input, hidden_features))
        dec_layers.append(nl)
        for _ in range(num_decoder_layers - 1):
            dec_layers.append(nn.Linear(hidden_features, hidden_features))
            dec_layers.append(nl)
        dec_layers.append(nn.Linear(hidden_features, n_features))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, w_in: torch.Tensor):
        h = self.encoder(w_in)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,
               z: torch.Tensor,
               alpha: torch.Tensor,
               selected_id: torch.Tensor):
        # one-hot encode selected blendshape id
        id_hot = one_hot(selected_id, num_classes=self.n_features).float()
        inp = torch.cat([z, alpha, id_hot], dim=1)
        delta = self.decoder(inp)
        return delta

    def valid_mask(self, alpha: torch.Tensor, selected_id: torch.Tensor):
        # mask = S[selected_id] >= alpha (broadcast)
        mask = (self.affinity[selected_id].squeeze(1) >= alpha).float().to(alpha.device)
        return mask

    def forward(self,
                w_in: torch.Tensor,
                selected_id: torch.Tensor,
                alpha: torch.Tensor):
        # Encode
        mu, logvar = self.encode(w_in)
        # Sample latent
        z = self.reparameterize(mu, logvar)
        # Decode delta
        delta = self.decode(z, alpha, selected_id)
        # Compute mask
        mask = self.valid_mask(alpha, selected_id)
        # Final prediction
        w_pred = torch.clamp(w_in + mask * delta, 0.0, 1.0)
        return w_pred, mu, logvar, mask
    
    # overload to, so we can move affinity matrix also to GPU
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.affinity = self.affinity.to(*args, **kwargs)
        return self


def build_model(config: dict):
    net_cfg = config['network']
    if net_cfg['type'] == 'controller':
        bs = load_blendshape(model='SP')
        S = compute_jaccard_similarity(bs)
        affinity = torch.tensor(S, dtype=torch.float32)
        model = NeuralFaceControllerCVAE(
            n_features=net_cfg['n_features'],
            affinity_matrix=affinity,
            hidden_features=net_cfg['hidden_features'],
            num_encoder_layers=net_cfg['num_encoder_layers'],
            latent_dim=net_cfg.get('latent_dimension', 5),
            num_decoder_layers=net_cfg['num_decoder_layers'],
            nonlinearity=net_cfg['nonlinearity']
        )
    else:
        raise ValueError(f"Unknown network type {net_cfg['type']}")
    return model

def load_model(config: dict):
    model = build_model(config)
    model_path = config["path"]
    state_dict = torch.load(os.path.join(model_path, "model.pt"))
    model.load_state_dict(state_dict)
    return model

def save_model(model, config:dict):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
