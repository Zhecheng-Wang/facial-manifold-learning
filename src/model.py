import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hierarchical_loss(recon_x, x):
    return F.mse_loss(recon_x[0], x, reduction='mean') + F.mse_loss(recon_x[1], x, reduction='mean')

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
    
class HierarchicalAutoEncoder(nn.Module):
    def __init__(self, n_features,\
                 clusters,\
                 hidden_features=64,\
                 num_encoder_layers=4,\
                 num_decoder_layers=4,\
                 nonlinearity='ReLU'):
        super().__init__()
        
        nls = {'ReLU':nn.ReLU(), 'ELU':nn.ELU()}
        nl = nls[nonlinearity]
        
        self.clusters = clusters
        
        if num_encoder_layers < 1:
            raise Exception("Invalid number of encoder layers")
        
        latent_dimension = 0
        self.ensemble_encoder = []
        for cluster in clusters:
            n_cluster_features = len(cluster)
            if n_cluster_features < 0 or n_cluster_features >= n_features:
                raise Exception("Invalid cluster")
            cluster_latent_dimension = math.ceil(n_cluster_features / 2)
            latent_dimension += cluster_latent_dimension
            encoder = [nn.Linear(n_cluster_features, hidden_features), nl]
            for i in range(num_encoder_layers):
               encoder.extend([nn.Linear(hidden_features, hidden_features), nl])
            encoder.append(nn.Linear(hidden_features, cluster_latent_dimension))
            self.ensemble_encoder.append(encoder)
        
        self.ensemble_encoder = nn.ModuleList([nn.Sequential(*encoder) for encoder in self.ensemble_encoder])
        
        if num_decoder_layers < 1:
            raise Exception("Invalid number of decoder layers")
        
        self.ensemble_decoder = []
        for cluster in clusters:
            n_cluster_features = len(cluster)
            if n_cluster_features < 0 or n_cluster_features >= n_features:
                raise Exception("Invalid cluster")
            cluster_latent_dimension = math.ceil(n_cluster_features / 2)
            decoder = [nn.Linear(cluster_latent_dimension, hidden_features), nl]
            for i in range(num_decoder_layers):
                decoder.extend([nn.Linear(hidden_features, hidden_features), nl])
            decoder.append(nn.Linear(hidden_features, n_cluster_features))
            # output layer clamp values between 0 and 1
            decoder.append(nn.Sigmoid())
            self.ensemble_decoder.append(decoder)
            
        self.ensemble_decoder = nn.ModuleList([nn.Sequential(*decoder) for decoder in self.ensemble_decoder])
        
        self.decoder = [nn.Linear(latent_dimension, hidden_features), nl]
        for i in range(num_decoder_layers):
            self.decoder.extend([nn.Linear(hidden_features, hidden_features), nl])
        self.decoder.append(nn.Linear(hidden_features, n_features))
        # output layer clamp values between 0 and 1
        self.decoder.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*self.decoder)
        
    def gather(self, x):
        return torch.cat(x, dim=-1)
    
    def scatter(self, x):
        return [x[...,cluster] for cluster in self.clusters]
        
    def ensemble_encode(self, x):
        return [encoder(x[...,cluster]) for encoder, cluster in zip(self.ensemble_encoder, self.clusters)]
    
    def ensemble_decode(self, x):
        return [decoder(code) for decoder, code in zip(self.ensemble_decoder, x)]
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        ensemble_pred = self.ensemble_decode(self.ensemble_encode(x))
        pred = self.decode(self.gather(self.ensemble_encode(x)))
        return self.gather(ensemble_pred), pred

def build_model(config:json):
    network_type = config["type"]
    network_config = config["network"]
    if network_type == "ae":
        model, loss =  AutoEncoder(n_features=network_config["n_features"],\
                                    hidden_features=network_config["hidden_features"],\
                                    num_encoder_layers=network_config["num_encoder_layers"],\
                                    latent_dimension=network_config["latent_dimension"],\
                                    num_decoder_layers=network_config["num_decoder_layers"],\
                                    nonlinearity=network_config["nonlinearity"]), mse_loss
    elif network_type == "hae":
        model, loss = HierarchicalAutoEncoder(n_features=network_config["n_features"],\
                                                clusters=config["clusters"],\
                                                hidden_features=network_config["hidden_features"],\
                                                num_encoder_layers=network_config["num_encoder_layers"],\
                                                num_decoder_layers=network_config["num_decoder_layers"],\
                                                nonlinearity=network_config["nonlinearity"]), hierarchical_loss
    else:
        raise Exception("Invalid network type")
    return model, loss
            
def load_model(config:json):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = os.path.join(config["path"], "model.pt")
    network_type = config["type"]
    network_config = config["network"]
    if network_type == "ae":
        model = AutoEncoder(n_features=network_config["n_features"],\
                            hidden_features=network_config["hidden_features"],\
                            num_encoder_layers=network_config["num_encoder_layers"],\
                            latent_dimension=network_config["latent_dimension"],\
                            num_decoder_layers=network_config["num_decoder_layers"],\
                            nonlinearity=network_config["nonlinearity"])
    elif network_type == "hae":
        model = HierarchicalAutoEncoder(n_features=network_config["n_features"],\
                                        clusters=config["clusters"],\
                                        hidden_features=network_config["hidden_features"],\
                                        num_encoder_layers=network_config["num_encoder_layers"],\
                                        num_decoder_layers=network_config["num_decoder_layers"],\
                                        nonlinearity=network_config["nonlinearity"])
    else:
        raise Exception("Invalid network type")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict) 
    
    return model

def save_model(model, config:json):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
