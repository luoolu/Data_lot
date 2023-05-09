## heavey lstm 

class LotteryPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 2048, num_layers=50, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(2048, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
        
        





















