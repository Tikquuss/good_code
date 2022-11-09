import torch
import re
import os 
from typing import Dict, List

from loss_landscape.cosine_sim import plt, consine_sim_weights_states, consine_sim_vec, consine_sim_vec_from_point#, plot_cosine_sim
from loss_landscape.utils import sorted_nicely

from src.modeling import TrainableTransformer
lightning_module_class = TrainableTransformer

def plot_cosine_sim(angles : List[Dict], ylabel=None, phases : Dict = None, save_to = None) :

    L, C = 1, len(angles)
    figsize = (7*C, 4*L)
    fig, axs = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)
    if C == 1 : axs = [axs]

    for ax, angle in zip(axs, angles) :
        ax.plot(angle["epochs"], angle['angles'], label=angle["label"])
    
    if phases is not None :
        colors = ["b", 'r', 'g', 'y']
        # labels = {
        #     'pre_memo_epoch' : 'pre_memorization_epoch (train_acc~5%)', 
        #     'pre_comp_epoch' : 'pre_comprehension_epoch (val_acc~5%)', 
        #     'memo_epoch' : 'memorization_epoch (train_acc~99%)', 
        #     'comp_epoch' : 'comprehension_epoch (val_acc~99%)'
        # }
        labels = {
            'pre_memo_epoch' : 'train_acc~5%', 
            'pre_comp_epoch' : 'val_acc~5%', 
            'memo_epoch' : 'train_acc~99%', 
            'comp_epoch' : 'val_acc~99%'
        }
        assert set(phases.keys()).issubset(set(labels.keys()))
        for i, k in enumerate(phases.keys()) :
            for ax in axs : ax.axvline(x = phases[k], color = colors[i], label = labels[k])
            #axs[0].axvline(x = phases[k], color = colors[i], label = labels[k])

    #axs[0].set(ylabel=ylabel)
    for ax in axs : 
      #ax.set(xlabel='epochs', ylabel=ylabel)
      ax.set(xlabel='epochs')
      #ax.set_title('title')
      ax.legend()

    if save_to is not None: fig.savefig(save_to, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
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


    ## step 3 : The epochs concerning by the plots 
    selected_epochs = list(range(0, 700+1, 1))
    tmp_model_files = [model_files[e] for e in selected_epochs]
    print(len(tmp_model_files))

    # step 5 : angles, weigths level

    print("==== angles, weigths level ====")

    dir_type = 'weights'
    #dir_type = 'states'

    #ignore = 'biasbn'
    ignore = ''
    angles1 = consine_sim_weights_states(tmp_model_files, dir_type, ignore, lightning_module_class)

    # dir_type = 'states'
    # angles11 = consine_sim_weights_states(tmp_model_files, dir_type, ignore, lightning_module_class)

    # step 6 : angles
    print("==== angles ====")
    angles2 = consine_sim_vec(tmp_model_files, lightning_module_class) 

    # step 7 : angles, from init
    print("==== angles, from init ====")
    angles3 = consine_sim_vec_from_point(model_files[0], tmp_model_files, lightning_module_class) 

    # step 8 : plot

    phases_k = ['pre_memo_epoch', 'pre_comp_epoch', 'memo_epoch', 'comp_epoch']
    plot_cosine_sim(
        angles = [
            {"angles" : angles1, "label" : "cos_%s (θ_{i+1}, θ_{i})"%dir_type, "epochs" : selected_epochs[:-1]},
            #{"angles" : angles2, "label" : "cos(θ_{i+1}, θ_{i})", "epochs" : selected_epochs[:-1]},
            {"angles" : angles3, "label" : "cos(theta_{i}, theta_0)", "epochs" : selected_epochs},
        ],
        ylabel="consine similarity", 
        phases = { k : states[k] for k in phases_k}, 
        save_to = "/content/angles.png"
    )