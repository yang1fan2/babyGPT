import numpy as np
import torch
import torch.nn as nn
from model import Transformer
from config import get_cfg_defaults
import tiktoken
enc = tiktoken.get_encoding("gpt2")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file("tiny.yaml")
    cfg.freeze()
    print(cfg)
    model =  Transformer(
            n_layers=cfg.TRAIN.N_LAYERS,
            n_head=cfg.TRAIN.N_HEAD,
            d_model=cfg.TRAIN.D_MODEL,
            n_vocab=enc.n_vocab,
            context_size=cfg.TRAIN.CONTEXT_SIZE,
            device=device,
            eot_token=enc.eot_token,
        ).to(device)
    model.load_state_dict(torch.load("model.pth"))

    while True:
        text = input("Enter prompt: ")
        if text == "exit":
            break
        X = enc.encode(text)
        y = model.generate(X)
        enc.decode(y)
        print(y)
