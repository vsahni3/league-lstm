from data import LeagueDataset
from model import LeagueModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from data import collate_fn
import config 
config = config.get_league_config()




checkpoint_callback = ModelCheckpoint(
   dirpath="checkpoints",
   monitor="val_loss",
   filename="lstm-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
   save_top_k=3,
   mode="min",
)

logger = TensorBoardLogger(save_dir="lightning_logs", name="league_model")


model = LeagueModel(config)

dataset = LeagueDataset('combined_data.jsonl', 'offsets.pkl', 'indices.pkl')
# for i in range(30):
#     print(dataset[i][0].shape, dataset[i][2], dataset[i][-1].item())


train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=16, shuffle=False)

trainer = L.Trainer(
   max_epochs=100,
   callbacks=[checkpoint_callback],
   logger=logger,
   accelerator="gpu" if torch.cuda.is_available() else "cpu",
   devices="auto",
)
trainer.fit(model, train_loader, val_loader)

