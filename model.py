import math
import torch
import torch.nn as nn
from data import LeagueDataset
import lightning as L
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
        batch_size = x.size()[0]
        seq_size = x.size()[1]
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
    def forward(self, x, labels=None):
        x = self.lstm(x)
        x = self.final(x).squeeze(-1)

        prob = self.sigmoid(x)
        if labels is None:
            return prob
        return prob, self.loss_fn(x, labels)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        prob, loss = self(x, labels)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        prob, loss = self(x, labels)
        acc = ((prob >= 0.5).float() == labels).float().mean()
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




