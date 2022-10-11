"""
```bash
python surface.py
```

This file illustrate how to plot the landscape as in the paper. 
It corresponds to `train.sh 90 +`, for 800 epochs.

You need to do this modification in `loss_landscape/model_loader.py` :
```python
logdir = "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+"
```

You can directly proceed like this :
```bash
model_file=/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/checkpoints/epoch=699-val_accuracy=99.7875.ckpt
python plot_surface.py --model_file $model_file
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
surf_file=/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/checkpoints/epoch=699-val_accuracy=99.7875.ckpt_weights.h5_[-1.0,1.0,51].h5
python loss_landscape/plot_1D.py --surf_file $surf_file --show
```
"""

import torch
import re
import os 
import shutil

from loss_landscape.utils import AttrDict, images_to_vidoe, sorted_nicely
from loss_landscape.plot_surface import plot_surface

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

phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
phases = [states[k] for k in phases_k]
print(phases)

# step 3 : select the epochs at which we can plot the landscape
"""
To see the slingshot point, check wandb if it was use, or check the tensorboard (it's is always use) :

```bash
%load_ext tensorboard
%tensorboard --logdir /content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/lightning_logs
```
"""

good_epochs = []
# start
for k in [2, 100, 200] : good_epochs.extend([k-1, k, k+1])
# phases
for p in phases : good_epochs.extend([p-1, p, p+1])
# slingshot
for k in [450, 578, 765] : good_epochs.extend([k-1, k, k+1])
# end
for k in [600, 700] : good_epochs.extend([k-1, k, k+1])
####
print(len(good_epochs), good_epochs)

# step 3 : define the parameters for the landscape plot

args = AttrDict({ 
    
    'mpi' : False, # use mpi
    'cuda' : True, # use cuda
    'threads' : 2, # 'number of threads'
    'ngpu' : 1, # 'number of GPUs to use for each rank, useful for data parallel evaluation

    # data parameters

    'raw_data' :False, # 'no data preprocessing'
    'data_split' : 1, #'the number of splits for the dataloader'
    'split_idx' : 0, # 'the index of data splits for the dataloader'

    # model parameters
    
    # parser.add_argument('--model', default='resnet56', help='model name')
    # parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    'model_file' : model_files[0], # path to the trained model file
    #'model_file2' : model_files[-1], # use (model_file2 - model_file) as the xdirection
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

    'proj_file' : '', # 'the .h5 file contains projected optimization trajectory.'
    'loss_max' : None, # 'Maximum value to show in 1D plot'
    'acc_max' :None, # 'ymax value (accuracy)')
    'vmax' : 10, # 'Maximum value to map'
    'vmin' : 0.1, # 'Miminum value to map'
    'vlevel' : 0.5, # 'plot contours every vlevel'
    'show' : True, # 'show plotted figures'
    'log' : False, # 'use log scale for loss values'
    'plot' : True, # 'plot figures after computation'
})

args.dir_type='weights'
#args.dir_type='states'

#args.xnorm="filter"
#args.xnorm="layer"
#args.xnorm="weight"

args.threads = 4


## step 4 : The epochs concerning by the plots (choose a single epoch for a single plot)
selected_epochs = good_epochs + []
for k in list(range(0, 150+1, 50)) + list(range(150, 750+1, 50)) : selected_epochs.extend([k-1, k, k+1])
selected_epochs = sorted(list(dict.fromkeys(selected_epochs)))
#selected_epochs = [s for s in selected_epochs if s > 651]
print(len(selected_epochs))


## step 5 : where to save the plots
save_to = "/content/surfaces_rand_dir"
os.makedirs(save_to, exist_ok=True)
#shutil.rmtree(save_to)


## step 6 : plots

dir_files, surf_files = {}, {}

#for epoch in [0, -1] :
#for epoch in [0, 1] :
for epoch in selected_epochs :
    args.model_file = model_files[epoch]
    #args.model_file2 = "" # random direction
    #args.model_file2 = model_files[epoch-1] # Epoch to Epoch
    #args.model_file2 =  model_files[epoch+1] # Epoch to Epoch
    #args.model_file2 =  model_files[0] # From the initialization point
    args.model_file2 =  model_files[-1] #  Until the end of the training

    #args.x='-2:2:41'
    args.x='-1.5:5.5:100'
    args.y='' # set the value to y for surface plots (take a lot of times)
    
    dir_files[epoch], surf_files[epoch] = plot_surface(
        args, lightning_module_class, metrics = ['val_loss', 'val_accuracy'],
        train_dataloader = data_module.train_dataloader(), 
        test_dataloader = data_module.val_dataloader(),
        save_to = f"{save_to}/{epoch}"
    )

## step 6 : images to vidoe animation / zip all

pattern = '^[0-9]+_1d_loss_acc.png$'
images_files = os.listdir(save_to)
images_files = [f for f in images_files if re.match(pattern, f)]
images_files= sorted_nicely(images_files)
images_files = [save_to + "/" + f for f in images_files]

#images_to_vidoe(video_path = "/content/rand.avi",  images_files = images_files[:10])

for repertoire, dossiers, fichiers in os.walk(save_to):
  for fichier in fichiers:
    if (not 'loss_acc' in fichier) or (not fichier.endswith(".png")) :
      filepath = os.path.join(repertoire, fichier)
      try:
        os.remove(filepath)
      except OSError as e:
          print(e)
      else:
          print("File " + filepath + " is deleted successfully")

"""
```bash
zip -r /content/surfaces_rand_dir.zip /content/surfaces_rand_dir
```
"""