import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reg_mse_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

def beta_elbo_loss(args, x, beta=1.0):
    recon_x, mu, logvar = args
    mse = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta*KLD

class AutoEncoder(nn.Module):
    def __init__(self, n_features,\
                 encoder_hidden_features=[10],\
                 latent_dimention=5,\
                 decoder_hidden_features=[10],\
                 nonlinearity='ReLU'):
        super().__init__()
        
        nls = {'ReLU':nn.ReLU(inplace=True), 'ELU':nn.ELU(inplace=True)}
        nl = nls[nonlinearity]
        
        n_encoder_layers = len(encoder_hidden_features)
        self.encoder = [nn.Linear(n_features, latent_dimention), nl]
        if n_encoder_layers >= 1:
            self.encoder = [nn.Linear(n_features, encoder_hidden_features[0]), nl]
            for i in range(n_encoder_layers-1):
                self.encoder.extend([nn.Linear(encoder_hidden_features[i], encoder_hidden_features[i+1]), nl])
        
        if n_encoder_layers >= 1:
            self.latent = [nn.Linear(encoder_hidden_features[-1], latent_dimention), nl]
        else:
            self.latent = [nn.Linear(latent_dimention, latent_dimention), nl]
        
        n_decoder_layers = len(decoder_hidden_features)
        self.decoder = []
        if n_decoder_layers >= 1:
            self.decoder = [nn.Linear(latent_dimention, decoder_hidden_features[0]), nl]
            for i in range(n_decoder_layers-1):
                self.decoder.extend([nn.Linear(decoder_hidden_features[i], decoder_hidden_features[i+1]), nl])
        
        if n_decoder_layers >= 1:
            self.decoder.append(nn.Linear(decoder_hidden_features[-1], n_features))
        else:
            self.decoder.append(nn.Linear(latent_dimention, n_features))
        # output layer with sigmoid to clamp values between 0 and 1
        self.decoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*self.encoder)
        self.latent = nn.Sequential(*self.latent)
        self.decoder = nn.Sequential(*self.decoder)
        
    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.latent(encoded)
        return latent
    
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
        
    def forward(self, x):
        return self.decode(self.encode(x))
    
    
class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, n_features,\
                 encoder_hidden_features=[10],\
                 latent_dimention=5,\
                 decoder_hidden_features=[10],\
                 nonlinearity='ReLU',\
                 noise_std=0.5):
        super().__init__(n_features,\
                         encoder_hidden_features,\
                         latent_dimention,\
                         decoder_hidden_features,\
                         nonlinearity)
        self.noise_std = noise_std
        
    def add_noise(self, x):
        x += torch.normal(mean=0.0, std=self.noise_std, size=x.shape).to(x.device)
        x = torch.clip(x, 0, 1)
        return x
    
    def encode(self, x):
        x = self.add_noise(x)
        return super().encode(x)


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, n_features,\
                 encoder_hidden_features=[10],\
                 latent_dimention=5,\
                 decoder_hidden_features=[10],\
                 nonlinearity='ReLU'):
        super().__init__(n_features,\
                         encoder_hidden_features,\
                         latent_dimention,\
                         decoder_hidden_features,\
                         nonlinearity)
        
        n_encoder_layers = len(encoder_hidden_features)
        if n_encoder_layers >= 1:
            self.mu = nn.Linear(encoder_hidden_features[-1], latent_dimention)
            self.logvar = nn.Linear(encoder_hidden_features[-1], latent_dimention)
        else:
            self.mu = nn.Linear(latent_dimention, latent_dimention)
            self.logvar = nn.Linear(latent_dimention, latent_dimention)
        
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def build_model(config:json):
    network_config = config["network"]
    if config["type"] == "ae":
        return AutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"]), reg_mse_loss
    elif config["type"] == "dae":
        return DenoisingAutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"],
                            noise_std=network_config["noise_std"]), reg_mse_loss
    elif config["type"] == "vae":
        return VariationalAutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"]), beta_elbo_loss
    else:
        raise Exception("Invalid network type")

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 nonlinearity='ReLU', output_nonlinearity=None, weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'Sine':(Sine(), sine_init, first_layer_sine_init),
                         'ReLU':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'ELU':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        self.net.append(nn.Linear(hidden_features, out_features))
        if output_nonlinearity:
            self.net.append(nls_and_inits[output_nonlinearity][0])

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        return self.net(coords)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))
            
def load_model(config:json):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = os.path.join(config["path"], "model.pt")
    network_type = config["type"]
    network_config = config["network"]
    if network_type == "ae":
        model = AutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"]).to(device)
    elif network_type == "dae":
        model = DenoisingAutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"],
                            noise_std=network_config["noise_std"]).to(device)
    elif network_type == "vae":
        model = VariationalAutoEncoder(n_features=network_config["n_features"],\
                            encoder_hidden_features=network_config["encoder_hidden_features"],\
                            latent_dimention=network_config["latent_dim"],\
                            decoder_hidden_features=network_config["decoder_hidden_features"],\
                            nonlinearity=network_config["nonlinearity"]).to(device)
    else:
        raise Exception("Invalid network type")
    
    model.load_state_dict(torch.load(model_path))
    
    return model

def save_model(model, config:json):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
