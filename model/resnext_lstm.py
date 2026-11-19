import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

class ResNeXtLSTM(nn.Module):
    def __init__(self, cnn_dropout=0.3, lstm_dropout=0.5, lstm_hidden_size=256, lstm_num_layers=1):
        super(ResNeXtLSTM, self).__init__()

        # Load pretrained ResNeXt and remove the final fully connected layer
        resnext = resnext50_32x4d(pretrained=True)
        modules = list(resnext.children())[:-2]  # remove avgpool & fc
        self.resnext_features = nn.Sequential(*modules)

        # Additional layers to compress feature maps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # shape → [batch, 2048, 1, 1]
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # LSTM to process temporal features
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=(lstm_dropout if lstm_num_layers > 1 else 0)
        )

        # Final classification layer
        self.fc = nn.Linear(lstm_hidden_size, 2)

    def forward(self, x):
        # x: [batch_size, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []

        for t in range(seq_len):
            out = self.resnext_features(x[:, t])            # [B, 2048, H', W']
            out = self.avgpool(out)                         # [B, 2048, 1, 1]
            out = out.view(batch_size, -1)                  # [B, 2048]
            out = self.cnn_dropout(out)                     # Apply dropout
            cnn_features.append(out)

        features_seq = torch.stack(cnn_features, dim=1)     # [B, T, 2048]
        lstm_out, _ = self.lstm(features_seq)               # [B, T, H]
        last_out = lstm_out[:, -1, :]                       # take last timestep

        logits = self.fc(last_out)                          # [B, 2]
        return logits

# Example usage
if __name__ == "__main__":
    model = ResNeXtLSTM()
    print(model)
    # Dummy input
    x = torch.randn(4, 10, 3, 224, 224)  # [batch_size, seq_len, C, H, W]
    output = model(x)
    print(output.shape)  # Should be [4, 2]