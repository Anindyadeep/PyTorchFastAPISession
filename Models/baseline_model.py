import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Model(nn.Module):
    def __init__(self, in_features : int) -> None:
        super(Model, self).__init__() 
        self.linear_layer1 = nn.Linear(in_features, 16)
        self.linear_layer2 = nn.Linear(16, 16)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier_layer = nn.Linear(16, 2)
    
    def forward(self, x : torch.Tensor):
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        return F.softmax(self.classifier_layer(x), dim=1)