import torch.nn as nn
import torch
from torch.nn.functional import normalize
from hypertab import Hypernetwork
import numpy as np


class Network(nn.Module):
    def __init__(self, input_size, params, class_num, fraction=0.8, test_nodes=50):
        super(Network, self).__init__()
        self.cluster_num = class_num
        self.hypernet = Hypernetwork(
            input_size,
            target_architecture=[(int(input_size*fraction), input_size), (input_size, class_num)],
            device="cuda:2",
            test_nodes=test_nodes).train()
        
        backbone = []

        for i in range(params['n_layers']):
            out_features = params["{}_layer_size".format(i)]
            backbone.append(nn.Linear(input_size, out_features))
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

    def add_noise(self, x):
        device = x.device
        nets = self.hypernet._Hypernetwork__craft_nets(self.hypernet.test_mask)
        res = []
        net_idx = np.random.randint(0, len(nets), size=2)
        for idx in net_idx:
            network = nets[idx]
            cast = x[:, self.hypernet.test_mask[idx].to(device).bool()] @ network.layers[0][0].T
            res.append(cast)
            
        return res[0], res[1]

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        device = x.device
        nets = self.hypernet._Hypernetwork__craft_nets(self.hypernet.test_mask)
        res = []

        for idx, network in enumerate(nets):
            cast = x[:, self.hypernet.test_mask[idx].to(device).bool()] @ network.layers[0][0].T
            h = self.backbone(cast)
            c = self.cluster_projector(h)
            res.append(c)

        c = torch.stack(res)
        c1 = torch.mean(c, dim=0)
        assert c1.shape == (x.shape[0], self.cluster_num)
        # c1 = self.cluster_projector(self.backbone(x))

        c_max = torch.argmax(c1, dim=1)
        return c_max

    def forward_backbone(self, x):
        x, _ = self.add_noise(x)
        h = self.backbone(x)
        return h