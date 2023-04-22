#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/23
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : automl_forecasting.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


# 1. Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.astype(np.float32)
    return data.values


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + 1:idx + self.seq_length + 1, :]
        return x, y


# 2. Define the Transformer model architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add extra dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Fix the issue here
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, nlayers, output_size):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            nlayers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.fc(src)
        return src


# 3. Implement the training loop with parallel training and dynamic learning rate optimization
def train(model, dataloader, device, loss_fn, optimizer, scheduler, epochs):
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}')

    return losses


# 4. Evaluate and save the best model
def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# 5. Visualize the loss curve
def plot_loss_curve(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


# Main function
def main():
    # Load and preprocess data
    file_path = '/home/luolu/PycharmProjects/lottery/Data/dlt2_asc20230418.csv'
    data = load_data(file_path)
    seq_length = 10
    batch_size = 128

    dataset = TimeSeriesDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Define the model
    input_size = 7
    d_model = 140
    nhead = 7
    nlayers = 50
    output_size = 7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_size, d_model, nhead, nlayers, output_size).to(device)

    # Enable parallel training if multiple CUDA devices are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Define the loss function, optimizer, and learning rate scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Train the model
    epochs = 100
    losses = train(model, dataloader, device, loss_fn, optimizer, scheduler, epochs)

    # Save the best model
    save_checkpoint(model, optimizer, '/media/luolu/3cc826d6-a968-4df2-9ae7-ee2c324675bd/home/xkjs/Downloads/model_dir/automl_dlt_best_model.pth')
    # Load the best model
    load_checkpoint(model, optimizer,
                    '/media/luolu/3cc826d6-a968-4df2-9ae7-ee2c324675bd/home/xkjs/Downloads/model_dir/automl_dlt_best_model.pth')

    # Print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Predict the N+1 group
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(last_sequence)
        print(f"Predicted N+1 group X1, X2, X3, X4, X5, X6, X7: {prediction[-1].cpu().numpy()}")

    # Plot the loss curve
    plot_loss_curve(losses)


if __name__ == '__main__':
    main()
