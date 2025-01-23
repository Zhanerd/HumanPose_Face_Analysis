import argparse
import time
import warnings
import numpy as np
import torch
import math
import torchvision
from torchvision import transforms
import cv2

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model_path = 'models/plfd_cosface_vgg.pth'
    checkpoint = torch.load(model_path, map_location=device)
    plfd_backbone = Model().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)