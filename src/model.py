import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def hierarchical_loss(recon_x, x):
    return F.mse_loss(recon_x[0], x, reduction='mean') + F.mse_loss(recon_x[1], x, reduction='mean')

def cluster_local_loss(recon_x, x):
    # vertorized the following operateion:
    xx = x.unsqueeze(1)
    xx = xx.repeat(1, recon_x[0].shape[1], 1)
    return F.mse_loss(recon_x[0], xx, reduction='mean')

def lipschitz_loss(recon_x, x, weight=1e-6):
    lipschitz_term = recon_x[1]
    recon_x = recon_x[0]
    return F.mse_loss(recon_x, x, reduction='mean') + weight * lipschitz_term

def l1_grad_reg_mse_loss(recon_x, x, weight=0.1):
    # regularize the gradient of doutput/dinput
    jacobian = torch.autograd.grad(recon_x, x, torch.ones_like(recon_x), create_graph=True)[0]
    print(jacobian.shape)
    # regularize non-diagonal elements
    jacobian = jacobian - torch.diag(torch.diag(jacobian))
    return F.mse_loss(recon_x, x, reduction='mean') + weight * torch.norm(jacobian, p=1)

def mse_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

def l1_reg_mse_loss(recon_x, x, weight=0.1):
    # regularize sparsity in output
    return F.mse_loss(recon_x, x, reduction='mean') + weight * torch.norm(recon_x, p=1)

# credit: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
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
    def infer(self, x):
        return self.forward(x)[1]

class Reshaper(nn.Module):
    def __init__(self, *args):
        super(Reshaper, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)
class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_value, num_heads, batch_first=True, average_attn_weights=False):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.dim_key = dim_value
        self.dim_query = dim_value
        self.dim_value = dim_value
        self.average_attn_weights = average_attn_weights
        self.query = nn.Linear(input_dim, dim_value*num_heads)
        self.key = nn.Linear(input_dim, dim_value*num_heads)
        self.value = nn.Linear(input_dim, dim_value*num_heads)
        self.multihead_attention = nn.MultiheadAttention(dim_value*num_heads, num_heads, batch_first=True)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.multihead_attention(q, k, v, average_attn_weights=self.average_attn_weights)
class AttentiveAutoEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_features=64,\
                 attention_query_features = 8,\
                 input_embedding_features = 8,\
                 attention_cluster_count = 5,\
                 num_encoder_layers=4,\
                 latent_dimension=5,\
                 num_decoder_layers=4,\
                 nonlinearity='ReLU'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_dimension = latent_dimension
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        # reshape the 2D input weights (batch, weight_dim) into a (2+1)D tensor (batch, weight_dim, 1)
        self.input_reshaper = Reshaper(-1, in_features, 1)
        # input layer (self attention layer)
        self.input_layer_embedding = nn.Linear(1 , input_embedding_features, bias=False)
        self.input_layer_attention = SelfAttention(input_embedding_features, hidden_features//attention_cluster_count, attention_cluster_count, batch_first=True)
        self.input_layer = nn.Sequential(self.input_reshaper, self.input_layer_embedding, self.input_layer_attention, self.nonlinearity)
        
        # encoders
        encoder_layer_contents = []
        for i in range(0, self.in_features):
            layer_i = []
            encoder_layer_contents.append(layer_i)
        for i in range(0, self.in_features):
            layer_i = encoder_layer_contents[i]
            for j in range(0, self.num_encoder_layers):
                layer_i.append(nn.Linear(hidden_features, hidden_features))
                layer_i.append(self.nonlinearity)
        # make sequentials out of them
        self.encoder_layer = []
        for i in range(0, self.in_features):
            self.encoder_layer.append(nn.Sequential(*encoder_layer_contents[i]))
        self.encoder_layer = nn.ModuleList(self.encoder_layer)
        # decoder layers
        decoder_layer_contents = []
        for i in range(0, self.in_features):
            layer_i = []
            decoder_layer_contents.append(layer_i)
        for i in range(0, self.in_features):
            layer_i = decoder_layer_contents[i]
            for j in range(0, self.num_decoder_layers):
                layer_i.append(nn.Linear(hidden_features, hidden_features))
                layer_i.append(self.nonlinearity)
        # make sequentials out of them
        self.decoding_layer = []
        for i in range(0, self.in_features):
            self.decoding_layer.append(nn.Sequential(*decoder_layer_contents[i]))
        self.decoding_layer = nn.ModuleList(self.decoding_layer)
        self.output_linear_layer = nn.Linear(hidden_features, self.out_features)
        self.output_activation = nn.Sigmoid()
        self.output_layer = nn.Sequential(self.output_linear_layer, self.output_activation)
        
    def forward(self, x):
        # input encoding
        x = self.input_reshaper(x)
        x = self.input_layer_embedding(x)
        x, attention_mask = self.input_layer_attention(x)
        
        # encoder
        z = []
        for i in range(0, self.in_features):
            z.append(self.encoder_layer[i](x[:, i]).unsqueeze(1))
        z = torch.cat(z, dim=1)
        # decoder
        out = []
        for i in range(0, self.in_features):
            out.append(self.decoding_layer[i](z[:, i]).unsqueeze(1))
        # output (per cluster output)
        out = torch.cat(out, dim=1)
        per_cluster_output = self.output_layer(out)
        
        return per_cluster_output, attention_mask

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
    elif network_type == "lipmlp":
        model = LipschitzMLP(in_features=network_config["n_features"],\
                            out_features=network_config["n_features"],\
                            num_hidden_layers=network_config["num_hidden_layers"],\
                            hidden_features=network_config["hidden_features"],\
                            nonlinearity=network_config["nonlinearity"])
    elif network_type == "hae":
        model, loss = HierarchicalAutoEncoder(n_features=network_config["n_features"],\
                                                clusters=config["clusters"],\
                                                hidden_features=network_config["hidden_features"],\
                                                num_encoder_layers=network_config["num_hidden_layers"],\
                                                num_decoder_layers=network_config["num_hidden_layers"],\
                                                nonlinearity=network_config["nonlinearity"]), hierarchical_loss
    elif network_type == "attention":
        model = AttentiveAutoEncoder(in_features=network_config["n_features"],\
                                        out_features=network_config["n_features"],\
                                        hidden_features=network_config["hidden_features"],\
                                        attention_query_features=network_config["attention_query_features"],\
                                        input_embedding_features = network_config["input_embedding_features"],\
                                        attention_cluster_count=network_config["attention_cluster_count"],\
                                        num_encoder_layers=network_config["num_hidden_layers"],\
                                        num_decoder_layers=network_config["num_hidden_layers"],\
                                        nonlinearity=network_config["nonlinearity"])
    else:
        raise Exception("Invalid network type")
    
    training_config = config["training"]
    loss_config = training_config["loss"]
    loss_type = loss_config["type"]
    if loss_type == "mse":
        loss = mse_loss
    elif loss_type == "l1_reg_mse":
        if "weight" not in loss_config:
            loss = l1_reg_mse_loss
        else:
            weight = loss_config["weight"]
            loss = lambda pred, x: l1_reg_mse_loss(pred, x, weight=weight)
    elif loss_type == "l1_grad_reg_mse":
        if "weight" not in loss_config:
            loss = l1_grad_reg_mse_loss
        else:
            weight = loss_config["weight"]
            loss = lambda pred, x: l1_grad_reg_mse_loss(pred, x, weight=weight)
    elif loss_type == "lipschitz":
        if "weight" not in loss_config:
            loss = lipschitz_loss
        else:
            weight = loss_config["weight"]
            loss = lambda pred, x: lipschitz_loss(pred, x, weight=weight)
    elif loss_type == "hierarchical":
        loss = hierarchical_loss
    elif loss_type == "cluster_local":
        loss = cluster_local_loss
    return model, loss

def load_model(config:dict):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = os.path.join(config["path"], "model.pt")
    network_config = config["network"]
    network_type = network_config["type"]
    if network_type == "ae":
        model = AutoEncoder(n_features=network_config["n_features"],\
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
    elif network_type == "lipmlp":
        model = LipschitzMLP(in_features=network_config["n_features"],\
                            out_features=network_config["n_features"],\
                            num_hidden_layers=network_config["num_hidden_layers"],\
                            hidden_features=network_config["hidden_features"],\
                            nonlinearity=network_config["nonlinearity"])
    elif network_type == "hae":
        model = HierarchicalAutoEncoder(n_features=network_config["n_features"],\
                                        clusters=config["clusters"],\
                                        hidden_features=network_config["hidden_features"],\
                                        num_encoder_layers=network_config["num_hidden_layers"],\
                                        num_decoder_layers=network_config["num_hidden_layers"],\
                                        nonlinearity=network_config["nonlinearity"])
    elif network_type == "attention":
        model = AttentiveAutoEncoder(in_features=network_config["n_features"],\
                                        out_features=network_config["n_features"],\
                                        hidden_features=network_config["hidden_features"],\
                                        input_embedding_features = network_config["input_embedding_features"],\
                                        attention_query_features=network_config["attention_query_features"],\
                                        attention_cluster_count=network_config["attention_head_count"],\
                                        num_encoder_layers=network_config["num_hidden_layers"],\
                                        latent_dimension=network_config["latent_dimension"],\
                                        num_decoder_layers=network_config["num_hidden_layers"],\
                                        nonlinearity=network_config["nonlinearity"])
    else:
        raise Exception("Invalid network type")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict) 
    
    return model

def save_model(model, config:dict):
    model_path = config["path"]
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
