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

class NeuralFaceController_learned_masking(nn.Module):
    def __init__(self, n_features,\
                 affinity_matrix,\
                 hidden_features=64,\
                 num_encoder_layers=4,\
                 latent_dimension=20,\
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
        
        self.encoder = [nn.Linear(n_features * 2 + 1, hidden_features), nl]
        # self.encoder = [nn.Linear(n_features *, hidden_features), nl]
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
        
        self.affinity_matrix = nn.Parameter(torch.tensor(affinity_matrix, dtype=torch.float32), requires_grad=False)
        
    def valid_mask(self, alpha, id):
        return self.affinity_matrix[id].squeeze(1) >= alpha
    
    def one_hot_encode_id(self, id):
        return F.one_hot(id, num_classes=self.n_features).float()
        
    def encode(self, w, alpha, id):
        
        id_onehot = self.one_hot_encode_id(id).squeeze(1)
        for i in range(0, id.shape[0]):
            if alpha[i] == 0:
                id_onehot[i] = torch.ones(id_onehot[i].shape).to(id_onehot[i].device)
            elif alpha[i] == 1:
                pass
            else:
                id_onehot[i] = self.valid_mask(alpha[i], id[i])
        return self.encoder(torch.cat([w, alpha, id_onehot], dim=1))


        # return self.encoder(w)
    
    def decode(self, z):
        # id = self.one_hot_encode_id(id).squeeze(1)
        # z = torch.cat([z, alpha, id], dim=1)
        return self.decoder(z)
    
    def infer(self, w, alpha, id):
        return self.forward(w, alpha, id)
        
    def forward(self, w, alpha, id):
        z = self.encode(w, alpha, id)
        w_pred = self.decode(z)
        # mask = self.valid_mask(alpha, id)
        # for valid index i in mask, w_output[i] = w_pred[i] * mask[i] + w[i] * (1 - mask[i])
        # for invalid index i in mask, w_output[i] = w_pred[i] + w[i]
        # therefore enforce only highly related blendshapes are in output (controlled by alpha/id and affinity matrix)
        # return (w_pred * mask) * alpha + ((w * mask) * (1 - alpha)) + (w * ~mask) + (w_pred * ~mask)
        return w_pred

class DiffusionController(nn.Module):
    def __init__(self, config, affinity_matrix):
        super().__init__()
        self.affinity_matrix = nn.Parameter(torch.tensor(affinity_matrix, dtype=torch.float32), requires_grad=False)
        self.config = config
        self.n_features = config["n_features"]
        # the encoder will be a 3 layered transformer encoder
        self.encoder = []
        self.decoder = []
        # encoder takes in the selection mask and the alpha
        self.encoder.append(nn.Linear(self.n_features * 2, config["hidden_features"]))
        self.encoder.append(nn.TransformerEncoderLayer(d_model=config["hidden_features"], nhead=8))
        for i in range(1, config["encoder_layers"]):
            self.encoder.append(nn.TransformerEncoderLayer(d_model=config["hidden_features"], nhead=8))
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder_input_layer = nn.Linear(self.n_features + 1, config["hidden_features"])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config["hidden_features"], nhead=8),
            num_layers=config["decoder_layers"]
        )
        self.output_layer = nn.Linear(config["hidden_features"], self.n_features)
    def valid_mask(self, alpha, id):
        return self.affinity_matrix[id].squeeze(1) >= alpha
    def infer(self, x, alpha, id, times=torch.tensor([1])):
        if alpha == 0:
            selection_mask = torch.ones_like(x)
        elif alpha == 1:
            selection_mask = torch.zeros_like(x)
            selection_mask[:, id[:, 0]] = 1
        else:
            selection_mask = self.valid_mask(alpha, id)
        times.to(x.device)
        return self.forward(x, times, alpha, selection_mask)
    def forward(self, x: torch.Tensor, times: torch.Tensor, alpha, selection_mask):
        # x: [batch_size, n_features]
        # times: [batch_size, 1]
        # cond_embed: [batch_size, n_features]
        times = times.unsqueeze(1)
        encoder_input = torch.cat([selection_mask, x], dim=1)
        encoder_output = self.encoder(encoder_input)
        decoder_input = torch.cat([x, times], dim=1)
        decoder_input = self.decoder_input_layer(decoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        return self.output_layer(decoder_output)

        

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
    elif network_type == "controller_learned_masking":
        affinity_matrix = compute_jaccard_similarity(load_blendshape(model="SP"))
        model = NeuralFaceController_learned_masking(n_features=network_config["n_features"],\
                                affinity_matrix=affinity_matrix,\
                                hidden_features=network_config["hidden_features"],\
                                num_encoder_layers=network_config["num_hidden_layers"],\
                                num_decoder_layers=network_config["num_hidden_layers"],\
                                nonlinearity=network_config["nonlinearity"])        
    elif network_type == "controller":
        affinity_matrix = compute_jaccard_similarity(load_blendshape(model="SP"))
        model = NeuralFaceController(n_features=network_config["n_features"],\
                                affinity_matrix=affinity_matrix,\
                                hidden_features=network_config["hidden_features"],\
                                num_encoder_layers=network_config["num_hidden_layers"],\
                                num_decoder_layers=network_config["num_hidden_layers"],\
                                nonlinearity=network_config["nonlinearity"])
    elif network_type == "diffusion":
        affinity_matrix = compute_jaccard_similarity(load_blendshape(model="SP"))
        model = DiffusionController(network_config, affinity_matrix)
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
