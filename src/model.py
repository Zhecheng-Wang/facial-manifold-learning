import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

def reg_mse_loss(recon_x, x):
    # regularize sparsity in output
    return F.mse_loss(recon_x, x, reduction='mean') + 0.1 * torch.norm(recon_x, p=1)

class AutoEncoder(nn.Module):
    def __init__(self, n_features,\
                 hidden_features=64,\
                 num_encoder_layers=4,\
                 latent_dimension=5,\
                 num_decoder_layers=4,\
                 nonlinearity='ReLU'):
        super().__init__()
        
        nls = {'ReLU':nn.ReLU(), 'ELU':nn.ELU()}
        nl = nls[nonlinearity]
        
        if num_encoder_layers < 1:
            raise Exception("Invalid number of encoder layers")
        
        self.encoder = [nn.Linear(n_features, hidden_features), nl]
        for i in range(num_encoder_layers):
            self.encoder.extend([nn.Linear(hidden_features, hidden_features), nl])
        self.encoder.append(nn.Linear(hidden_features, latent_dimension))
                
        if num_decoder_layers < 1:
            raise Exception("Invalid number of decoder layers")
        
        self.decoder = [nn.Linear(latent_dimension, hidden_features), nl]
        for i in range(num_decoder_layers):
            self.decoder.extend([nn.Linear(hidden_features, hidden_features), nl])
        self.decoder.append(nn.Linear(hidden_features, n_features))
        # output layer clamp values between 0 and 1
        self.decoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        return self.decode(self.encode(x))

def build_model(config:json):
    network_config = config["network"]
    return AutoEncoder(n_features=network_config["n_features"],\
                           hidden_features=network_config["hidden_features"],\
                           num_encoder_layers=network_config["num_encoder_layers"],\
                           latent_dimension=network_config["latent_dimension"],\
                           num_decoder_layers=network_config["num_decoder_layers"],\
                           nonlinearity=network_config["nonlinearity"]), mse_loss
            
def load_model(config:json):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = os.path.join(config["path"], "model.pt")
    network_config = config["network"]
    model = AutoEncoder(n_features=network_config["n_features"],\
                        hidden_features=network_config["hidden_features"],\
                        num_encoder_layers=network_config["num_encoder_layers"],\
                        latent_dimension=network_config["latent_dimension"],\
                        num_decoder_layers=network_config["num_decoder_layers"],\
                        nonlinearity=network_config["nonlinearity"]).to(device)
    
    model.load_state_dict(torch.load(model_path))
    
    return model

def save_model(model, config:json):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
