import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(28 * 28, 512),
      nn.RELU(),
      nn.Linear(512, 512),
      nn.RELU(),
      nn.Linear(512, 10)
    )
    
  def forward(self, x):
    logits = self.mlp(x)
    return logits
    