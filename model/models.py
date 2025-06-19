import torch
from torch import nn
import torch.optim.adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class simple_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11520, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class from_evo(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class small_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class mini_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3840, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.3),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        # self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class modular_network(nn.Module):
    def __init__(self, n_layers, drop_out, input_size=3840, start_hidden_size=1024):
        super().__init__()
        layers = []
        # start_hidden_size = 1024
        hidden_size = [input_size]
        for i in range(n_layers):
            hidden_size.append(start_hidden_size)
            start_hidden_size = int(start_hidden_size/2)
        # print(hidden_size)
        
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            print(f"{hidden_size[i]} -> {hidden_size[i+1]}")
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size[i+1]))
            layers.append(nn.Dropout(drop_out))
        
        # print(f"Last: {hidden_size[-1]} -> 1")
        layers.append(nn.Linear(hidden_size[-1], 1))
        layers.append(nn.Sigmoid())

        self.linear_relu_stack = nn.Sequential(*layers).apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        net = self.linear_relu_stack(x)
        return net
