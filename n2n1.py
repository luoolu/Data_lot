import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 1. Prepare the dataset
class IntegersDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return x, y

def data_preparation(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.apply(np.sort, axis=1)
    data = pd.get_dummies(data.stack()).sum(level=0)
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = IntegersDataset(train_data)
    valid_dataset = IntegersDataset(valid_data)
    return train_dataset, valid_dataset

# 2. Define the neural network architecture
class IntegerPredictor(nn.Module):
    def __init__(self):
        super(IntegerPredictor, self).__init__()
        self.fc1 = nn.Linear(35, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 35)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Train the model
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# 4. Evaluate the model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# 5. Generate predictions
def generate_predictions(model, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.eye(35).to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        return top5_probs.cpu().numpy(), top5_indices.cpu().numpy()

def main():
    train_dataset, valid_dataset = data_preparation("data.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IntegerPredictor().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_dataloader, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Plot the loss curve
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Generate predictions
    top5_probs, top5_indices = generate_predictions(model, device)
    print("Top 5 probabilities:\n", top5_probs)
    print("Top 5 indices:\n", top5_indices)

if __name__ == "__main__":
    main()

