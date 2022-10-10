import torch
import re
import os 

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from loss_landscape.andrews_curve import get_u, g_ft, get_new_cmap
from loss_landscape.cosine_sim import w_to_vec
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

# step 3 : params

thetas = [
    w_to_vec(model_file, lightning_module_class).detach().numpy()
    for i, model_file in enumerate(model_files[100:200]) if i%2==0
]

N=len(thetas)
print(N)

d = thetas[0].size
print(d, thetas[0].shape)

# step 4 : angles
L=100
alpha = np.linspace(start = -np.pi, stop = np.pi, num=L)

# step 5 : projection
u = get_u(alpha, d)
print(u.shape)

# step 6 : projection
f_t = g_ft(u, thetas)
print(f_t.shape)

# step 7 : plot
name='viridis'
#name="plasma"
#name='inferno'
#name='magma'
colors, new_cmap = get_new_cmap(name='viridis', N=N) 

fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [30, 1]})

ax1.set_xlim([-np.pi, np.pi])
ax1.set_ylim([f_t.min(), f_t.max()])

for i, tmp in enumerate(f_t):    
    ax1.plot(alpha, tmp, c=colors[i])
    
cb  = matplotlib.colorbar.ColorbarBase(ax2, cmap=new_cmap,
                                orientation='vertical',
                                ticks=[0,1])



# step  : animation
fig, ax = plt.subplots(1,1)

ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([f_t.min(), f_t.max()])

def animate(i):
    ax.clear()
    tmp = f_t[i]
    ax.plot(alpha, tmp, c=colors[i])
    ax.text(60, .025, str(i))
    ax.text(0.5, 0.5, str(i))
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([f_t.min(), f_t.max()])

ani = FuncAnimation(fig, animate, frames=N, interval=500, repeat=False)
plt.close()

# step 

from matplotlib.animation import PillowWriter
# Save the animation as an animated GIF
ani.save("/content/simple_animation.gif", dpi=300,
         writer=PillowWriter(fps=1))