"""
This file illustrate how to plot the condition number of the Hessian as a function of training epoch as in the paper. 
It corresponds to `train.sh 90 +`, for 800 epochs.

You need to do this modification in `loss_landscape/model_loader.py` :
```python
logdir = "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+"
```
"""

import torch
import re
import os 
import matplotlib.pyplot as plt

from loss_landscape.plot_hessian_eigen import plot_hessian_eigen_models, get_loss
from loss_landscape.utils import AttrDict
from loss_landscape.utils import sorted_nicely

from src.modeling import TrainableTransformer
lightning_module_class = TrainableTransformer

# step 1 : train the model

"""
max_epochs=800
use_wandb=False

```bash
train.sh 90 +
```
"""

# step 2 : load the checkpoints, the data and the params

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

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = [states[k] for k in phases_k]
print(phases)


## step 3 : select the epochs to save time

selected_epochs = list(range(0, 150+1, 5)) + list(range(150, 750+1, 2))
selected_epochs = sorted(list(dict.fromkeys(selected_epochs)))
print(len(selected_epochs))

tmp_model_files = [model_files[e] for e in selected_epochs]


## step 4 : params

args = AttrDict({ 
    'cuda' : True, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # direction parameters
    'dir_type' : 'weights', #'direction type: weights | states (including BN\'s running_mean/var)'
})

## step 5 : data

if True :
    dataloader = data_module.train_dataloader()
    data_size = len(data_module.train_dataset)
else :
    dataloader = data_module.val_dataloader()
    data_size = len(data_module.val_dataset)

## step 6 :

min_eig, max_eig = plot_hessian_eigen_models(
    args, 
    tmp_model_files,
    lightning_module_class, dataloader, data_size, get_loss) 

## step 7 :

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = { k : states[k] for k in phases_k}

## step 8 :

# plotting the given graph
L, C = 1, 3
figsize=(5*C, 4*L)
fig, (ax1, ax2, ax3) = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)

ax1.plot(selected_epochs, min_eig, 
#        marker = "+", markersize = 15, color = "red",
         label = "λ_min"
)
ax2.plot(selected_epochs, max_eig, 
#        marker = ".", markersize = 15, color = "green",
        label = "λ_max"
)
ax3.plot(
#ax3.semilogy(    
    selected_epochs, 
    [a/b for a, b in zip(max_eig, min_eig)],
    #[abs(a/b) for a, b in zip(max_eig, min_eig)], 
#    marker = "o", markersize = 15, color = "black",
    label = "λ_max/λ_min"
)

# plot with grid
#plt.grid(True)
for ax in [ax1, ax2, ax3] : ax.grid(True)

if phases is not None :
    colors = ["b", 'r', 'g', 'y']
    labels = {
        'pre_memo_epoch' : 'train_acc~5%', 
        'pre_comp_epoch' : 'val_acc~5%', 
        'memo_epoch' : 'train_acc~99%', 
        'comp_epoch' : 'val_acc~99%'
    }
    assert set(phases.keys()).issubset(set(labels.keys()))
    for i, k in enumerate(phases.keys()) :
        for ax in [ax1, ax2, ax3] :
            ax.axvline(x = phases[k], color = colors[i], label = labels[k])

for ax, ylabel in zip([ax1, ax2, ax3], ["λ_min", "λ_max", "λ_min/λ_max"]) :
    ax.set(xlabel='epochs', 
           #ylabel=ylabel
    )
    ax.legend()

# show the plot
plt.show()

## step 9 :
torch.save([min_eig, max_eig], "/content/eigens.pt")