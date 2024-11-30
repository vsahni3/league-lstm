import math
import torch
import torch.nn as nn
from data import LeagueDataset
import lightning as L
from collections import defaultdict
import matplotlib.pyplot as plt
import config 
config = config.get_league_config()


class LSTM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.W = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size * 4))
        self.sigmoid = torch.sigmoid 
        self.tanh = torch.tanh
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def single_pass(self, x, cur_states=None):
        # (b, input_size)
        h_t, c_t = cur_states
        # vectorize by stacking different gates instead of 4 different computations
        gates = x @ self.W + h_t @ self.U + self.bias
        hs = self.hidden_size
        input_gate, forget_gate, candidate_gate, output_gate = (
                self.sigmoid(gates[:, :hs]), 
                self.sigmoid(gates[:, hs:hs*2]),
                self.tanh(gates[:, hs*2:hs*3]),
                self.sigmoid(gates[:, hs*3:]), 
            )
        c_t = c_t * forget_gate + input_gate * candidate_gate
        h_t = self.tanh(c_t) * output_gate
        return h_t, c_t 
    
    def forward(self, x):
        # (b, seq_size, input_size)
        batch_size, seq_size, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_size):
            cur_x = x[:, t, :]
            h_t, c_t = self.single_pass(cur_x, (h_t, c_t))
        return h_t 

class LeagueModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lstm = LSTM(config)
        self.final = nn.Linear(config.hidden_size, 1)
        self.sigmoid = torch.sigmoid 
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.length_stats = defaultdict(lambda: {"correct": 0, "count": 0})
        self.prop_stats = defaultdict(lambda: {"correct": 0, "count": 0})
    def forward(self, x, labels=None):
        x = self.lstm(x)
        x = self.final(x).squeeze(-1)

        prob = self.sigmoid(x)
        if labels is None:
            return prob
        return prob, self.loss_fn(x, labels)

    def training_step(self, batch, batch_idx):
        x, _, _, labels = batch
        prob, loss = self(x, labels)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, total_lengths, labels = batch
        prob, loss = self(x, labels)
        acc = ((prob >= 0.5).float() == labels).float().mean()
        for i, length in enumerate(lengths):
            proportion = round((length / total_lengths[i]) * 20) / 20
            res = ((prob[i] >= 0.5) == labels[i]).item()
            self.length_stats[length]['correct'] += res
            self.prop_stats[proportion]['correct'] += res
            self.length_stats[length]['count'] += 1
            self.prop_stats[proportion]['count'] += 1
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    def on_validation_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.plot_length_accuracies(True)
            self.plot_length_accuracies(False)

    def plot_length_accuracies(self, is_abs):
        stats_dict, key = (self.length_stats, 'length') if is_abs else (self.prop_stats, 'prop')
        lengths = sorted(stats_dict.keys())
        accuracies = [
            stats_dict[length][f'correct'] / stats_dict[length][f"count"] 
            if stats_dict[length][f"count"] > 0 else 0
            for length in lengths
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(lengths, accuracies, marker="o", label="Accuracy by Length")
        plt.xlabel(f"Sequence {key}")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy by Sequence Length (Epoch {self.current_epoch})")
        plt.legend()
        plt.savefig(f'{key}.png')
        plt.close()



