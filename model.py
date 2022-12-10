import torch
import torch.nn as nn
from torch.nn import functional as F

class DropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(DropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        if self.keep_shape:
            return x.transpose(1,2)
        else:
            return x

class AcousticModel(nn.Module):
    def __init__(self, hidden_size, num_classes, n_feats, dense_size, num_layers, dropout):
        super(AcousticModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            DropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, dense_size),
            nn.LayerNorm(dense_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, dense_size),
            nn.LayerNorm(dense_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=dense_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout = 0.0,
                            bidirectional = False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)
        
    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))
    
    def forward(self, x, hidden):
        x = x.squeeze(1) # batch, feature, time
        x = self.cnn(x) # batch, time, feature
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden) # note hidden = (h0, c0)
        x = self.layer_norm2(out)
        x = F.gelu(x)
        x = self.dropout2(x) # time, batch, n_class
        return self.final_fc(x), (hn, cn)
