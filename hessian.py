"""
```bash
python hessian.py
```
"""

import re
import os 
import shutil
import torch

from loss_landscape.utils import sorted_nicely
from loss_landscape.cosine_sim import w_to_vec
from src.modeling import TrainableTransformer

logdir = "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+"
pretrained_folder = logdir + "/checkpoints"

#pattern = '^epoch_[0-9]+.ckpt$'
pattern = '^epoch=[0-9]+-val_accuracy=[0-9]+\.[0-9]+.ckpt$'

model_files = os.listdir(pretrained_folder)
model_files = [f for f in model_files if re.match(pattern, f)]
model_files = sorted_nicely(model_files)
model_files = ["init.ckpt"] + model_files
model_files = [pretrained_folder + "/" + f for f in model_files]

hparams = torch.load(logdir + "/hparams.pt")
data_module = torch.load(logdir+"/data.pt")
states = torch.load(logdir+"/states.pt")

# model = TrainableTransformer(hparams).float()
model = TrainableTransformer.load_from_checkpoint(hparams = hparams, checkpoint_path = model_files[0]).float()

#model

model.set_states(states)
print(model.states)

print(model.is_grok(delay=10), model.is_grok(delay=100))

diff_epoch = model.comp_epoch - model.memo_epoch
print(diff_epoch)

data_size = hparams.data_module_params.train_data_size

# train_dataloader = data_module.train_dataloader()
# batch = next(iter(train_dataloader))

x = data_module.train_dataset.text[:2]
y = data_module.train_dataset.target[:2]
batch = {"text" : x, "target" : y}

loss, grad_vec = model._step(
        batch = batch,
        batch_idx = 0,
        data_size = data_size,
        reduction = "mean",
        grads = True)

print(grad_vec.shape)

lightning_module_class = TrainableTransformer
v = w_to_vec(model_files[0], lightning_module_class).detach()
print(v.shape)