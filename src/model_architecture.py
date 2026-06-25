import torch
import torch.nn as nn

class ClimateCNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(128*4*4, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        cnn_in = x.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn(cnn_in)
        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        return self.fc(lstm_out[:, -1, :])