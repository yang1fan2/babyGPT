import numpy as np
import torch
import torch.nn as nn
from model import Transformer
from config import get_cfg_defaults
import tiktoken
from torchinfo import summary
enc = tiktoken.get_encoding("gpt2")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class DataLoader:
    def __init__(self, filename, batch_size, context_length, is_train):
        self.is_train = is_train
        self.batch_size = batch_size
        self.context_length = context_length
        self.fp = np.memmap(filename, dtype='uint16', mode='r').astype(np.int32)
        self.length = np.shape(self.fp)[0]

    def __getitem__(self, idx):
        rand_idx = np.random.randint(self.length - self.context_length - 1, size = self.batch_size)
        Xs, ys = [], []
        for i in range(self.batch_size):
            Xs.append(self.fp[rand_idx[i]:rand_idx[i] + self.context_length])
            ys.append(self.fp[rand_idx[i]+1:rand_idx[i] + self.context_length+1])
        X = np.stack(Xs)
        y = np.stack(ys)
        return torch.from_numpy(X), torch.from_numpy(y).to(torch.int64)

    def __len__(self):
        return int(self.length / self.batch_size)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.view(-1, enc.n_vocab), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 


def validate(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.view(-1, enc.n_vocab), y.view(-1)).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file("tiny.yaml")
    cfg.freeze()
    print(cfg)

    training_data = DataLoader("data/wikitext/train.bn", cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.CONTEXT_SIZE, True)
    validation_data = DataLoader("data/wikitext/validation.bn", cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.CONTEXT_SIZE, False)
    for X,y in training_data:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        break

    model = Transformer(
        n_layers=cfg.TRAIN.N_LAYERS,
        n_head=cfg.TRAIN.N_HEAD,
        d_model=cfg.TRAIN.D_MODEL,
        n_vocab=enc.n_vocab,
        context_size=cfg.TRAIN.CONTEXT_SIZE,
        device=device
    ).to(device)

    summary(model, input_data=torch.ones((cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.CONTEXT_SIZE)).int().to(device))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = model.get_optimizer()

    for t in range(cfg.TRAIN.N_EPOCH):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, loss_fn, optimizer)
        validate(validation_data, model, loss_fn)
    
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
