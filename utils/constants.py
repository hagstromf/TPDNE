from pathlib import Path
import os
import torch

ROOT_DIR = Path(__file__).parent.parent

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

if __name__ == '__main__':
        pass