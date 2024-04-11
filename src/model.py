import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from clustering import compute_jaccard_similarity
from utils import load_blendshape
import math
class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)   

class LipschitzMLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, nonlinearity='ReLU'):
        super().__init__()

        nls = {'ReLU':nn.ReLU(), 'ELU':nn.ELU()}
        nl = nls[nonlinearity]

        self.net = []
        self.net.extend([LipschitzLinear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([LipschitzLinear(hidden_features, hidden_features), nl])

        self.net.append(LipschitzLinear(hidden_features, out_features))
        self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def infer(self, x):
        return self.net(x)

    def forward(self, x):
        lipschitz_term = 1.0
        for layer in self.net:
            if isinstance(layer, LipschitzLinear):
                lipschitz_term *= layer.get_lipschitz_constant()
        return self.net(x), lipschitz_term
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, nonlinearity='ReLU'):
        super().__init__()

        nls = {'ReLU':nn.ReLU(), 'ELU':nn.ELU()}
        nl = nls[nonlinearity]

        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)
        
    def infer(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.net(x)  

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
    
    def infer(self, x):
        return self.forward(x)
        
    def forward(self, x):
        return self.decode(self.encode(x))
    
class NeuralFaceController(nn.Module):
    def __init__(self, n_features,\
                 affinity_matrix,\
                 hidden_features=64,\
                 num_encoder_layers=4,\
                 latent_dimension=5,\
                 num_decoder_layers=4,\
                 nonlinearity='ReLU'):
        super().__init__()
        
        self.n_features = n_features
        
        if affinity_matrix.shape[0] != n_features or affinity_matrix.shape[1] != n_features:
            raise Exception(f"Invalid affinity matrix, shape should be ({n_features}, {n_features})")
        
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
        
        self.decoder = [nn.Linear(latent_dimension+1+n_features, hidden_features), nl]
        for i in range(num_decoder_layers):
            self.decoder.extend([nn.Linear(hidden_features, hidden_features), nl])
        self.decoder.append(nn.Linear(hidden_features, n_features))
        # output layer clamp values between 0 and 1
        self.decoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
        self.affinity_matrix = nn.Parameter(torch.tensor(affinity_matrix, dtype=torch.float32), requires_grad=False)
        
    def valid_mask(self, alpha, id):
        return self.affinity_matrix[id].squeeze(1) >= alpha
    
    def one_hot_encode_id(self, id):
        return F.one_hot(id, num_classes=self.n_features).float()
        
    def encode(self, w):
        return self.encoder(w)
    
    def decode(self, z, alpha, id):
        id = self.one_hot_encode_id(id).squeeze(1)
        z = torch.cat([z, alpha, id], dim=1)
        return self.decoder(z)
    
    def infer(self, w, alpha, id):
        return self.forward(w, alpha, id)
        
    def forward(self, w, alpha, id):
        z = self.encode(w)
        w_pred = self.decode(z, alpha, id)
        mask = self.valid_mask(alpha, id)
        # for valid index i in mask, w_output[i] = w_pred[i] * mask[i] + w[i] * (1 - mask[i])
        # for invalid index i in mask, w_output[i] = w_pred[i] + w[i]
        # therefore enforce only highly related blendshapes are in output (controlled by alpha/id and affinity matrix)
        return (w_pred * mask) * alpha + ((w * mask) * (1 - alpha)) + (w * ~mask) + (w_pred * ~mask)

def build_model(config:dict):
    network_config = config["network"]
    network_type = network_config["type"]
    if network_type == "ae":
        model =  AutoEncoder(n_features=network_config["n_features"],\
                            hidden_features=network_config["hidden_features"],\
                            num_encoder_layers=network_config["num_encoder_layers"],\
                            latent_dimension=network_config["latent_dimension"],\
                            num_decoder_layers=network_config["num_decoder_layers"],\
                            nonlinearity=network_config["nonlinearity"])
    elif network_type == "mlp":
        model = MLP(in_features=network_config["n_features"],\
                    out_features=network_config["n_features"],\
                    num_hidden_layers=network_config["num_hidden_layers"],\
                    hidden_features=network_config["hidden_features"],\
                    nonlinearity=network_config["nonlinearity"])
    elif network_type == "controller":
        affinity_matrix = compute_jaccard_similarity(load_blendshape(model="SP"))
        model = NeuralFaceController(n_features=network_config["n_features"],\
                                affinity_matrix=affinity_matrix,\
                                hidden_features=network_config["hidden_features"],\
                                num_encoder_layers=network_config["num_hidden_layers"],\
                                num_decoder_layers=network_config["num_hidden_layers"],\
                                nonlinearity=network_config["nonlinearity"])
    elif network_type == "lipmlp":
        model = LipschitzMLP(in_features=network_config["n_features"],\
                            out_features=network_config["n_features"],\
                            num_hidden_layers=network_config["num_hidden_layers"],\
                            hidden_features=network_config["hidden_features"],\
                            nonlinearity=network_config["nonlinearity"])
    else:
        raise Exception("Invalid network type")
    return model

def load_model(config:dict):
    model = build_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = os.path.join(config["path"], "model.pt")
    print(model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict) 
    return model

def save_model(model, config:dict):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
