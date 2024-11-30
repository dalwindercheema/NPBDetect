import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, feats, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(feats, 100)
        self.gelu1 = nn.GELU()
        self.layer1 = nn.LayerNorm(100)
        self.fc2 = nn.Linear(feats, 50)
        self.gelu2 = nn.GELU()
        self.layer2 = nn.LayerNorm(50)
        self.fc3 = nn.Linear(50, num_classes)
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu1(out)
        out = self.layer1(out)
        out = self.fc2(x)
        out = self.gelu2(out)
        out = self.layer2(out)
        out = self.fc3(out)
        return out

def get_model(base_dir):
    num_classes = 8
    class_wts = torch.tensor([ 0.9676,  2.4682,  1.5786,  6.1968, 13.2386, 15.7432, 15.3289,  9.5492])
    feats = 1131
    print('Loading model parameters')
    model = NeuralNet(feats, num_classes)
    state = torch.load(base_dir + 'model/npb_model', weights_only=True)
    model.load_state_dict(state)
    return model