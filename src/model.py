import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from clustering import compute_jaccard_similarity
from data.utils import load_blendshape


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
    def __init__(
        self,
        n_features: int,
        affinity_matrix: torch.Tensor,
        hidden_features: int = 64,
        num_encoder_layers: int = 4,
        latent_dim: int = 5,
        num_decoder_layers: int = 4,
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        # Validate affinity
        assert affinity_matrix.shape == (
            n_features,
            n_features,
        ), f"Affinity must be [{n_features},{n_features}]"
        self.n_features = n_features
        self.affinity = affinity_matrix

        # Nonlinearity
        nls = {"ReLU": nn.ReLU(), "ELU": nn.ELU()}
        nl = nls[nonlinearity]

        # Encoder: stacks of Linear+NL -> two heads (mu, logvar)
        enc_layers = []
        enc_layers.append(nn.Linear(n_features, hidden_features))
        enc_layers.append(nl)
        for _ in range(num_encoder_layers - 1):
            enc_layers.append(nn.Linear(hidden_features, hidden_features))
            enc_layers.append(nl)
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_features, latent_dim)
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

    def decode(self, z: torch.Tensor, alpha: torch.Tensor, selected_id: torch.Tensor):
        # one-hot encode selected blendshape id
        id_hot = one_hot(selected_id, num_classes=self.n_features).float()
        inp = torch.cat([z, alpha, id_hot], dim=1)
        delta = self.decoder(inp)
        return delta

    def valid_mask(self, alpha: torch.Tensor, selected_id: torch.Tensor):
        # mask = S[selected_id] >= alpha (broadcast)
        mask = (self.affinity[selected_id].squeeze(1) >= alpha).float().to(alpha.device)
        return mask

    def forward(
        self, w_in: torch.Tensor, selected_id: torch.Tensor, alpha: torch.Tensor
    ):
        # Encode
        mu, logvar = self.encode(w_in)
        # Sample latent
        z = self.reparameterize(mu, logvar)
        # Decode delta
        delta = self.decode(z, alpha, selected_id)  # these are three inputs?
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


class RinatNeuralFaceController(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_features: int = 64,
        num_encoder_layers: int = 4,
        latent_dim: int = 5,
        num_decoder_layers: int = 4,
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        # Validate affinity

        self.n_features = n_features
        if n_features == 48:
            self.lower_face_indices = [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                26,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                46,
                47,
            ]
            self.upper_face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 25, 27, 28, 44, 45]
        else:
            raise ValueError(f"Unsupported number of features: {n_features}")
        # Nonlinearity
        nls = {"ReLU": nn.ReLU(), "ELU": nn.ELU()}
        nl = nls[nonlinearity]

        # Encoder for upper face
        enc_layers_upper = []
        enc_layers_upper.append(
            nn.Linear(len(self.upper_face_indices), hidden_features)
        )
        enc_layers_upper.append(nl)
        for _ in range(num_encoder_layers - 1):
            enc_layers_upper.append(nn.Linear(hidden_features, hidden_features))
            enc_layers_upper.append(nl)
        fc_mu_upper = nn.Linear(hidden_features, latent_dim)
        enc_layers_upper.append(fc_mu_upper)
        self.encoder_upper = nn.Sequential(*enc_layers_upper)

        # Decoder: input is z
        dec_input = latent_dim
        dec_layers_upper = []
        dec_layers_upper.append(nn.Linear(dec_input, hidden_features))
        dec_layers_upper.append(nl)
        for _ in range(num_decoder_layers - 1):
            dec_layers_upper.append(nn.Linear(hidden_features, hidden_features))
            dec_layers_upper.append(nl)
        dec_layers_upper.append(
            nn.Linear(hidden_features, len(self.upper_face_indices))
        )
        dec_layers_upper.append(nn.Sigmoid())
        self.dec_layers_upper = nn.Sequential(*dec_layers_upper)

        # Encoder for lower face
        enc_layers_lower = []
        enc_layers_lower.append(
            nn.Linear(len(self.lower_face_indices), hidden_features)
        )
        enc_layers_lower.append(nl)
        for _ in range(num_encoder_layers - 1):
            enc_layers_lower.append(nn.Linear(hidden_features, hidden_features))
            enc_layers_lower.append(nl)
            # dec_layers_lower.append(nn.Dropout(0.2))
        fc_mu_lower = nn.Linear(hidden_features, latent_dim)
        enc_layers_lower.append(fc_mu_lower)
        self.encoder_lower = nn.Sequential(*enc_layers_lower)
        # Decoder: input is z
        dec_input = latent_dim
        dec_layers_lower = []
        dec_layers_lower.append(nn.Linear(dec_input, hidden_features))
        dec_layers_lower.append(nl)
        for _ in range(num_decoder_layers - 1):
            dec_layers_lower.append(nn.Linear(hidden_features, hidden_features))
            dec_layers_lower.append(nl)
            # add dropout
            # dec_layers_lower.append(nn.Dropout(0.2))
        dec_layers_lower.append(
            nn.Linear(hidden_features, len(self.lower_face_indices))
        )
        dec_layers_lower.append(nn.Sigmoid())
        self.dec_layers_lower = nn.Sequential(*dec_layers_lower)

    def encode(self, w_in: torch.Tensor):
        w_in_upper = w_in[:, self.upper_face_indices]
        w_in_lower = w_in[:, self.lower_face_indices]
        z_upper = self.encoder_upper(w_in_upper)
        z_lower = self.encoder_lower(w_in_lower)
        return z_upper, z_lower

    def decode(self, z_upper, z_lower, w_in):
        delta_upper = self.dec_layers_upper(z_upper)
        delta_lower = self.dec_layers_lower(z_lower)
        # Combine the deltas
        delta = torch.zeros_like(w_in)
        delta[:, self.upper_face_indices] = delta_upper
        delta[:, self.lower_face_indices] = delta_lower
        return delta

    def forward(
        self, w_in: torch.Tensor, selected_id: torch.Tensor, alpha: torch.Tensor
    ):
        # Encode
        z_upper, z_lower = self.encode(w_in)
        # Sample latent
        pred = self.decode(z_upper, z_lower, w_in)  # these are three inputs?
        # Compute mask
        return pred


class MultiClusterNeuralFaceController(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_features: int = 64,
        num_encoder_layers: int = 4,
        latent_dim: int = 5,
        num_decoder_layers: int = 4,
        nonlinearity: str = "ReLU",
        clustering: dict = None,
    ):
        super().__init__()
        # Validate affinity

        if clustering is None:
            lower_face_indices = [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                26,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
            ]
            upper_face_indices = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                25,
                27,
                28,
                44,
                45,
                46,
                47,
            ]
            self.clustering = {"upper": upper_face_indices, "lower": lower_face_indices}
        elif isinstance(clustering, list):
            self.clustering = {}
            for i, indices in enumerate(clustering):
                self.clustering[f"cluster_{i}"] = indices
        else:
            self.clustering = clustering

        self.cluster_names = list(self.clustering.keys())
        self.cluster_names.sort()

        # Nonlinearity
        nls = {"ReLU": nn.ReLU(), "ELU": nn.ELU()}
        nl = nls[nonlinearity]

        # Encoder for upper face

        self.encoder_lists = nn.ModuleDict({})
        self.decoder_lists = nn.ModuleDict({})

        for name in self.cluster_names:
            # Encoder: outputs z
            enc_layers = []
            enc_layers.append(nn.Linear(len(self.clustering[name]), hidden_features))
            enc_layers.append(nl)
            for _ in range(num_encoder_layers - 1):
                enc_layers.append(nn.Linear(hidden_features, hidden_features))
                enc_layers.append(nl)
            fc_mu = nn.Linear(hidden_features, latent_dim)
            enc_layers.append(fc_mu)
            self.encoder_lists[name] = nn.Sequential(*enc_layers)

            # Decoder: input is z
            dec_input = latent_dim
            dec_layers = []
            dec_layers.append(nn.Linear(dec_input, hidden_features))
            dec_layers.append(nl)
            for _ in range(num_decoder_layers - 1):
                dec_layers.append(nn.Linear(hidden_features, hidden_features))
                dec_layers.append(nl)
            dec_layers.append(nn.Linear(hidden_features, len(self.clustering[name])))
            dec_layers.append(nn.Sigmoid())
            self.decoder_lists[name] = nn.Sequential(*dec_layers)

    def encode(self, w_in: torch.Tensor):
        # Encode each cluster
        z_list = []
        for name in self.cluster_names:
            indices = self.clustering[name]
            w_in_cluster = w_in[:, indices]
            z = self.encoder_lists[name](w_in_cluster)
            z_list.append(z)

        return z_list

    def decode(self, z_list, w_in):
        # Decode each cluster
        delta = torch.zeros_like(w_in)
        for i, name in enumerate(self.cluster_names):
            indices = self.clustering[name]
            z = z_list[i]
            delta_cluster = self.decoder_lists[name](z)
            delta[:, indices] = delta_cluster

        return delta

    def forward(
        self, w_in: torch.Tensor, selected_id: torch.Tensor, alpha: torch.Tensor
    ):
        # Encode
        z_list = self.encode(w_in)
        # Sample latent
        pred = self.decode(z_list, w_in)  # these are three inputs?
        # Compute mask
        return pred


def build_model(config: dict):
    net_cfg = config["network"]
    if net_cfg["type"] == "controller":
        bs = load_blendshape(model="SP")
        S = compute_jaccard_similarity(bs)
        affinity = torch.tensor(S, dtype=torch.float32)
        model = NeuralFaceControllerCVAE(
            n_features=net_cfg["n_features"],
            affinity_matrix=affinity,
            hidden_features=net_cfg["hidden_features"],
            num_encoder_layers=net_cfg["num_encoder_layers"],
            latent_dim=net_cfg.get("latent_dimension", 5),
            num_decoder_layers=net_cfg["num_decoder_layers"],
            nonlinearity=net_cfg["nonlinearity"],
        )
    elif net_cfg["type"] == "rinat_controller":
        clustering = net_cfg.get("clustering", None)
        model = MultiClusterNeuralFaceController(
            n_features=net_cfg["n_features"],
            hidden_features=net_cfg["hidden_features"],
            num_encoder_layers=net_cfg["num_encoder_layers"],
            latent_dim=net_cfg.get("latent_dimension", 5),
            num_decoder_layers=net_cfg["num_decoder_layers"],
            nonlinearity=net_cfg["nonlinearity"],
            clustering=clustering,
        )
    else:
        raise ValueError(f"Unknown network type {net_cfg['type']}")
    return model


def load_model(config: dict):
    model = build_model(config)
    model_dir_path = config["path"]
    model_path = os.path.join(model_dir_path, "model.pt")
    if model_dir_path[0] == ".":
        proj_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        )
        model_path = os.path.join(proj_root, model_dir_path[2:], "model.pt")

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model


def save_model(model, config: dict):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
