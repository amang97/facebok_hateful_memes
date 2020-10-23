from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

class FbhmHAM():
    def __init__(self):
        super().__init__()
    
    def ham(self, ve, te):
        """
        trains and learns the classification weights using te and ve 
        """
        prediction = None
        return prediction