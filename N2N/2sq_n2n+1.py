#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : n2n+1.py
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.data.append(list(map(int, row)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(7, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 7)

    def forward(self, src):
        src = self.embedding(src)  # Re-add the embedding layer
        src = src.unsqueeze(1)  # Add an extra dimension
        output = self.transformer(src, src)  # Use src as both input and target
        output = output.squeeze(1)  # Remove the extra dimension
        return self.fc(output)


def train_model(rank, world_size, dataset_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    dataset = CustomDataset(dataset_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(dataset, batch_size=640, sampler=train_sampler)

    device = torch.device(f"cuda:{rank}")
    d_model = 1024
    model = TransformerModel(d_model, 128, 32).to(device)
    # 计算并打印模型参数总量
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    num_epochs = 5000
    losses = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch[:-1], batch[1:]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    if rank == 0:
        torch.save(model.state_dict(),
                   "/media/luolu/3cc826d6-a968-4df2-9ae7-ee2c324675bd/home/xkjs/Downloads/model_dir/ssq_best_model.pth")
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

        # Plot trends
        model.eval()
        X = np.arange(1, 36)
        fig, ax = plt.subplots(7, 1, figsize=(12, 30))
        for i in range(7):
            Y = [model(torch.tensor([[x if idx == i else 0 for idx in range(7)]], dtype=torch.float)).cpu().detach().numpy()[0] for x in X]
            ax[i].plot(X, Y)
            ax[i].set_title(f"X{i + 1}")
            ax[i].set_xlabel("Value")
            ax[i].set_ylabel("Prediction")

        plt.tight_layout()
        plt.show()
        # Generate the N+1 prediction
        last_input = torch.tensor(dataset[-1], dtype=torch.float).unsqueeze(0).to(device)
        next_prediction = model(last_input).cpu().detach().numpy()[0]

        print("N+1 Prediction:", next_prediction)


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dataset_path = "/home/luolu/PycharmProjects/lottery/Data/ssq_history.csv"
    world_size = torch.cuda.device_count()
    mp.spawn(train_model, args=(world_size, dataset_path), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
