## deep modoel
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Add PositionalEncoding in the TransformerModel
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(7, d_model)
        self.pos_encoder = PositionalEncoding(d_model)  # Add positional encoding
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 7)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)  # Apply positional encoding
        src = src.transpose(0, 1)  # Swap batch size and sequence length dimensions
        output = self.transformer(src, src)
        output = output.transpose(0, 1)  # Swap back the dimensions
        return self.fc(output)

'''
