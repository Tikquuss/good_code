"""
This file illustrate how to plot the condition number of the Hessian for a given landscape plot. 
It corresponds to `train.sh 90 +`, for 800 epochs.

You need to do this modification in `loss_landscape/model_loader.py` :
```python
logdir = "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+"
```

You can directly proceed like this :
```bash
model_file=/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/checkpoints/epoch=699-val_accuracy=99.7875.ckpt
python plot_hessian_eigen.py --model_file $model_file
```

Then :
```python
for f in os.listdir(pretrained_folder) :
    ff = pretrained_folder + "/" + f
    if ff not in model_files :
        print(ff)
```

And then :
```bash
surf_file=/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/checkpoints/epoch=699-val_accuracy=99.7875.ckpt_weights...
python loss_landscape/plot_2D.py --surf_file $surf_file --show
```
"""

import torch
import re
import os 
import shutil

from loss_landscape.utils import AttrDict, sorted_nicely
from loss_landscape.plot_hessian_eigen import plot_hessian_eigen, get_loss

from src.modeling import TrainableTransformer
from src.dataset import DataModule

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

# step 3 : define the parameters for the plot

args = AttrDict({ 
    
    'mpi' : True, # use mpi
    'cuda' : False, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # data parameters

    'raw_data' :False, # 'no data preprocessing'
    'data_split' : 1, #'the number of splits for the dataloader')
    'split_idx' : 0, # 'the index of data splits for the dataloader'

    # model parameters
    
    # parser.add_argument('--model', default='resnet56', help='model name')
    # parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    'model_file' : model_files[0], # path to the trained model file
    'model_file2' : "", # use (model_file2 - model_file) as the xdirection
    'model_file3' : "", # use (model_file3 - model_file) as the ydirection
    #'loss_name' : 'crossentropy', # help='loss functions: crossentropy | mse')

    # direction parameters

    'dir_file' : '',  # 'specify the name of direction file, or the path to an eisting direction file
    'dir_type' : 'weights', #'direction type: weights | states (including BN\'s running_mean/var)'
    'x' : '-1:1:51', #'A string with format xmin:x_max:xnum'
    'y' : None, #'A string with format ymin:ymax:ynum'
    #'y' : '-1:1:51', #'A string with format ymin:ymax:ynum'
    'xnorm' : '', # 'direction normalization: filter | layer | weight'
    'ynorm' : '', # 'direction normalization: filter | layer | weight'
    'xignore' : '', #'ignore bias and BN parameters: biasbn'
    'yignore' : '', #'ignore bias and BN parameters: biasbn'
    'same_dir' : False, # 'use the same random direction for both x-axis and y-axis'
    'idx' : 0, # 'the index for the repeatness experiment')
    'surf_file' : '', # 'customize the name of surface file, could be an existing file.'

    # plot parameters

    'show' : True, # help='show plotted figures')
    'plot' : True, #  help='plot figures after computation')
})

# step 4 : data

if True :
    dataloader = data_module.train_dataloader()
    data_size = len(data_module.train_dataset)
else :
    dataloader = data_module.val_dataloader()
    data_size = len(data_module.val_dataset)


# step 4 : lanscape points and params

args.model_file = model_files[-1]
args.model_file2 = ""

args.x='-1:1:51'
args.y=''

dir_file, surf_file = plot_hessian_eigen(args, lightning_module_class, dataloader, data_size, get_loss)