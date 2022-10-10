from typing import Dict, List
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

from IPython.display import clear_output

import matplotlib.pyplot as plt

from .net_plotter import get_weights, ignore_biasbn
from .model_loader import load
from .projection import tensorlist_to_tensor, cal_angle

def w_to_list(model_file, dir_type, ignore, lightning_module_class) :
    net = load(lightning_module_class, model_file = model_file)
    if dir_type == 'weights': w = get_weights(net)
    elif dir_type == 'states': w = [v for _, v in net.state_dict()]
    if ignore == 'biasbn': ignore_biasbn(w)
    w = tensorlist_to_tensor(w)
    return w

def consine_sim_weights_states(model_files, dir_type, ignore, lightning_module_class) :

    angles = []

    w1 = w_to_list(model_files[0], dir_type, ignore, lightning_module_class)

    for i in range(1, len(model_files)):
        if i%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)

        w2 = w_to_list(model_files[i], dir_type, ignore, lightning_module_class)

        alpha = cal_angle(w1, w2).item()
        angles.append(alpha)
        print(i-1, i, alpha)

        w1 = w2 + 0.0
    
    return angles

def consine_sim_weights_states_from_point(model_file, model_files, dir_type, ignore, lightning_module_class) :

    angles = []

    w1 = w_to_list(model_file, dir_type, ignore, lightning_module_class)

    for i in range(len(model_files)):
        if i%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)

        w2 = w_to_list(model_files[i], dir_type, ignore, lightning_module_class)

        alpha = cal_angle(w1, w2).item()
        angles.append(alpha)
        print(i-1, i, alpha)
    
    return angles

def w_to_vec(model_file, lightning_module_class) :
    net = load(lightning_module_class, model_file = model_file)
    w = parameters_to_vector(net.parameters())
    return w

def consine_sim_vec(model_files, lightning_module_class) :
    #args.dir_type
    angles = []

    w1 = w_to_vec(model_files[0], lightning_module_class)

    for i in range(1, len(model_files)):
        if i%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)

        w2 = w_to_vec(model_files[i], lightning_module_class)
    
        alpha = F.cosine_similarity(w1, w2, dim=0).item()
        angles.append(alpha)
        print(i-1, i, alpha)

        w1 = w2 + 0.0
    
    return angles


def consine_sim_vec_from_point(model_file, model_files, lightning_module_class) :
    #args.dir_type
    angles = []

    w1 = w_to_vec(model_file, lightning_module_class)

    for i in range(len(model_files)):
        if i%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)

        w2 = w_to_vec(model_files[i], lightning_module_class)
    
        alpha = F.cosine_similarity(w1, w2, dim=0).item()
        angles.append(alpha)
        print(i, alpha)
    
    return angles


def plot_cosine_sim(angles : List[Dict], ylabel=None, phases : Dict = None, save_to = None) :

    figsize=(6*3,4*2)
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize = figsize)
    #fig.suptitle("suptitle")

    for angle in angles :
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
            ax.axvline(x = phases[k], color = colors[i], label = labels[k])

    ax.set(xlabel='epochs', ylabel=ylabel)
    #ax.set_title('title')
    ax.legend()

    if save_to is not None: fig.savefig(save_to, dpi=300, bbox_inches='tight')