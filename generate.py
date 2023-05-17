import numpy as np
import torch
import torch.nn as nn
from model import Transformer
from config import get_cfg_defaults


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = Transformer().to(device)
model.load_state_dict(torch.load("model.pth"))

