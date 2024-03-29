import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def scale_during_training(x, scale_factor, is_training):
    if is_training:
        return x * (1 / (1 - scale_factor))
    else:
        return x

class Network(nn.Module):
    def __init__(self, input_size, params, class_num):
        super(Network, self).__init__()
        self.cluster_num = class_num
        backbone = []
        
        for i in range(params['n_layers']):
            out_features = params["{}_layer_size".format(i)]
            backbone.append(nn.Linear(input_size, out_features))
            
            if i == 0 and params['noise'] == 'zero':  # Check if it's the first layer
                backbone.append(nn.LeakyReLU())
                backbone.append(Lambda(lambda x: scale_during_training(x, params['masking_ratio'], self.training)))
            else:
                backbone.append(nn.LeakyReLU())
            
            input_size = out_features

        self.backbone = nn.Sequential(*backbone)
        self.latent_dim = input_size
        self.instance_projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim,),
            nn.ReLU(),
            nn.Linear(self.latent_dim, params['projection_size'])
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim,),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    def forward_backbone(self, x):
        h = self.backbone(x)
        return h