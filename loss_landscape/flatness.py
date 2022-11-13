"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""

# PyTorch Lightning
import pytorch_lightning as pl

# import argparse
import copy
import h5py
import torch
import time
import socket
import os
# import sys
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from IPython.display import clear_output
from sklearn.decomposition import PCA

from .mpi4pytorch import setup_MPI
from .evaluation import Evaluator
from .net_plotter import get_weights, set_weights, set_states, get_diff_weights, get_diff_states, ignore_biasbn
from .model_loader import load
from .projection import tensorlist_to_tensor, npvec_to_tensorlist
from .h5_util import write_list, read_list

def perform_PCA(args, w, s, dir_name, matrix, n_components):
    # Perform PCA on the optimization path matrix
    print ("Perform PCA on the models")
    pca = PCA(n_components=n_components)
    pca.fit(np.array(matrix))

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    directions = []
    if args.dir_type == 'weights':
        for pc in pca.components_ :
            pc = pc / np.linalg.norm(pc) # normalize
            directions.append(npvec_to_tensorlist(pc, w))
        
    elif args.dir_type == 'states':
        for pc in pca.components_ :
            pc = pc / np.linalg.norm(pc) # normalize
            directions.append(npvec_to_tensorlist(pc, s))

    if args.ignore == 'biasbn':
        for direction in directions :
            ignore_biasbn(direction)

    f = h5py.File(dir_name, 'w')
    for i, direction in enumerate(directions) :
        write_list(f, f'{i}_direction', direction)

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_
    f['n_components'] = n_components

    f.close()
    print ('PCA directions saved in: %s' % dir_name)

    return dir_name


def setup_PCA_directions_from_point(args, model_files, w, s, n_components, lightning_module_class = None):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    folder_name = args.model_folder + '/PCA_from_point' + args.dir_type
    if args.ignore:
        folder_name += '_ignore=' + args.ignore
    folder_name += '_save_epoch=' + str(args.save_epoch)
    #os.system('mkdir ' + folder_name)
    os.makedirs(folder_name, exist_ok=True)
    dir_name = folder_name + '/directions.h5'

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    s, w = None, None
    for count, model_file in enumerate(model_files):
        print(model_file)
        if count%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)
        net2 = load(lightning_module_class, model_file = model_file)
        d = None
        if args.dir_type == 'weights':
            w2 = get_weights(net2)
            d = get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = get_diff_states(s, s2)
        if args.ignore == 'biasbn':
            ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    return perform_PCA(args, w, s, dir_name, matrix, n_components)

def setup_PCA_directions(args, model_files, n_components, lightning_module_class = None):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    folder_name = args.model_folder + '/PCA_' + args.dir_type
    if args.ignore:
        folder_name += '_ignore=' + args.ignore
    folder_name += '_save_epoch=' + str(args.save_epoch)
    #os.system('mkdir ' + folder_name)
    os.makedirs(folder_name, exist_ok=True)
    dir_name = folder_name + '/directions.h5'

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    s, w = None, None
    for count, model_file in enumerate(model_files):
        print(model_file)
        if count%100 == 0 : 
            #os.system('cls')
            clear_output(wait=True)
        net = load(lightning_module_class, model_file = model_file)
        d = None
        if args.dir_type == 'weights':
            w = get_weights(net)
            d = get_diff_weights(0, w)
        elif args.dir_type == 'states':
            s = net.state_dict()
            d = get_diff_states(0, s)
        if args.ignore == 'biasbn':
            ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    return perform_PCA(args, w, s, dir_name, matrix, n_components)

def crunch(dir_file, net, w, s, dataloaders, loss_keys, acc_keys, args, evaluator, beta_loss, beta_acc):
    """
        Calculate the loss values and/or accuracies of modified models
    """
    assert len(dataloaders) == len(loss_keys) == len(acc_keys)
    coordinates1 = np.linspace(0, args.xmax, num=args.xnum)
    coordinates2 = np.linspace(0, args.xmin, num=args.xnum)

    L_0 = {}
    acc_0 = {}
    for dataloader, loss_key, acc_key in zip(dataloaders, loss_keys, acc_keys) :
        L_0[loss_key], acc_0[acc_key] = evaluator(net, dataloader)
    
    f = h5py.File(dir_file, 'r')
    """
    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_
    """
    results_loss = [L_0]
    results_acc = [acc_0]
    try :
        start_time_ = 0
        # https://groups.google.com/g/h5py/c/3la4BMAr8DE?pli=1
        n_components = f['n_components'][()]
        for i in range(n_components) :
            direction = [read_list(f, f'{i}_direction')]
            alpha_losses, alpha_accuracies = {}, {}
            for j, coordinates in enumerate([coordinates1, coordinates2]) :
                alpha_losses[j], alpha_accuracies[j] = {}, {}
                # is_done_losses, is_done_accuracies = {}, {}
                # for loss_key, acc_key in zip(loss_keys, acc_keys) :
                #     is_done_losses[loss_key] = False
                #     is_done_accuracies[acc_key] = False
                ####################################
                for loss_key, acc_key in zip(loss_keys, acc_keys) :
                    alpha_losses[j][loss_key] = {}
                    alpha_accuracies[j][acc_key] = {}
                ####################################
                L_coords = len(coordinates)
                for ind, coord in enumerate(coordinates):
                    start_time = time.time()
                    # Load the weights corresponding to those coordinates into the net
                    if args.dir_type == 'weights': 
                        set_weights(net.module if args.ngpu > 1 else net, w, direction, coord)
                    elif args.dir_type == 'states':
                        set_states(net.module if args.ngpu > 1 else net, s, direction, coord)

                    loss_compute_time = 0
                    for dataloader, loss_key, acc_key in zip(dataloaders, loss_keys, acc_keys) :
                        loss, acc = evaluator(net, dataloader)
                        alpha_losses[j][loss_key][coord] = loss
                        alpha_accuracies[j][acc_key][coord] = acc

                        # if loss >= beta_loss * L_0[loss_key] and not is_done_losses[loss_key] :
                        #     alpha_losses[j][loss_key] = coord
                        #     is_done_losses[loss_key] = True

                        # if acc <= acc_0[acc_key] / beta_acc and not is_done_accuracies[acc_key] :
                        #     alpha_accuracies[j][acc_key] = coord
                        #     is_done_accuracies[acc_key] = True

                    loss_compute_time += time.time() - start_time 
                    # TODO
                    print('%d/%d (%.1f%%) coord=%s %s=%.2f %s=%.2f \ttime=%.2f' % (
                        ind, L_coords, 100.0*ind/L_coords, str(coord), acc_key, acc, loss_key, loss, loss_compute_time))

                    # if all(is_done_losses.values()) and all(is_done_accuracies.values()) :
                    #     break

                    if ind%10 == 0 : 
                        #os.system('cls')
                        clear_output(wait=True)

                # for loss_key, acc_key in zip(loss_keys, acc_keys):
                #     if not is_done_losses[loss_key] : alpha_losses[j][loss_key] = str(coord) + '_last'
                #     if not is_done_accuracies[acc_key] : alpha_accuracies[j][acc_key] = str(coord) + '_last'

            results_loss.append(alpha_losses)
            results_acc.append(alpha_accuracies)

            total_time = time.time() - start_time_ 
            print('Done!  Total time: %.2f' % (total_time))
    except OSError as e : # Unable to open file (file is already open for read-only) 
        print(e)
    finally :
        f.flush()
        f.close()

    return results_loss, results_acc


def plot_components(n_components, loss_keys, acc_keys, results_loss, results_acc, beta_loss, beta_acc,
                    save_to = None, show = True):

    L_0 = results_loss[0]
    acc_0 = results_acc[0]

    L, C = n_components, len(loss_keys)
    figsize=(8*C, 5*L)
    fig, axs = plt.subplots(L, C, sharex=False, sharey=False, figsize = figsize)

    for i in range(1, n_components+1) :
        for j, (keys, results, r_0) in enumerate(zip([loss_keys, acc_keys], [results_loss, results_acc], [L_0, acc_0])) :
            for key in keys :
                alpha_1, alpha_2 = 0, 0
                x_left = list(results[i][1][key].keys())
                x_right = list(results[i][0][key].keys())
                x = x_left[::-1] + x_right
                y_left = list(results[i][1][key].values())
                y_right = list(results[i][0][key].values())
                y = y_left[::-1] + y_right
                label = key.split("_")[0]
                axs[i-1][j].plot(x, y, label=label)

                if 'loss' in key :
                    for k, loss_v in enumerate(y_left) :
                        if loss_v >= beta_loss * L_0[key] :
                            alpha_1 = x_left[k]
                            print(f"{label}_loss (alpha_1) = ", alpha_1)
                            plt.axvline(x = alpha_1, color = 'b', label = f'{label}_alpha_1')
                            break
                    for k, loss_v in enumerate(y_right) :
                        if loss_v >= beta_loss * L_0[key] :
                            alpha_2 = x_right[k]
                            print(f"{label}_loss (alpha_2) = ", alpha_2)
                            plt.axvline(x = alpha_2, color = 'b', label = f'{label}_alpha_2')
                            break
                    
                if 'acc' in key :
                    for k, loss_v in enumerate(y_left) :
                        if loss_v <= acc_0[key] / beta_acc : 
                            alpha_1 = x_left[k]
                            print(f"{label}_acc (alpha_1) = ", alpha_1)
                            plt.axvline(x = alpha_1, color = 'b', label = f'{label}_alpha_1')
                            break
                    for k, loss_v in enumerate(y_right) :
                        if loss_v <= acc_0[key] / beta_acc :
                            alpha_2 = x_right[k]
                            print(f"{label}_acc (alpha_2) = ", alpha_2)
                            plt.axvline(x = alpha_2, color = 'b', label = f'{label}_alpha_2')
                            break

    for i in range(L):
        for j in range(C) :
            axs[i][j].legend()
            if i==0 :
                axs[i][j].set_title([loss_keys, acc_keys][j][0].split("_")[1])
            if i==L-1:
                axs[i][j].set(xlabel='alpha')
            if j==0:
                axs[i][j].set(ylabel=f"direction_{i}")

    if save_to is not None :
        filename = save_to + '_flatness_plot' 
        plt.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
        fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: plt.show()

def flatness(args, lightning_module_class, metrics, model_files, model_file_PCA = None, n_components = 2,
                    train_dataloader = None, test_dataloader = None, save_to = None,
                    beta_loss = 2.71828, # e
                    beta_acc = 2.71828 # e
                    ) :

    assert train_dataloader or test_dataloader

    # Setting the seed
    pl.seed_everything(42)

    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' % (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.xnum = int(args.xnum)
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, 'You specified some arguments for the y axis, but not all'
            args.ynum = int(args.ynum)
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = load(lightning_module_class, model_file = args.model_file)
    w = get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------    
    if args.dir_file:
        dir_file = args.dir_file
    else:
        if model_file_PCA is None :
            dir_file = setup_PCA_directions(args, model_files, n_components, lightning_module_class)
        else :
            net_PCA = load(lightning_module_class, model_file = model_file_PCA)
            w_PCA = get_weights(net_PCA) # initial parameters
            s_PCA = net_PCA.state_dict()
            dir_file = setup_PCA_directions_from_point(args, model_files, w_PCA, s_PCA, n_components, lightning_module_class)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    evaluator = Evaluator(metrics = metrics)

    dataloaders, loss_keys, acc_keys = [], [], []
    if train_dataloader :
        dataloaders, loss_keys, acc_keys = [train_dataloader], ['train_loss'], ['train_acc']
    if test_dataloader :
        dataloaders.append(test_dataloader)
        loss_keys.append('test_loss')
        acc_keys.append('test_acc')

    results_loss, results_acc = crunch(dir_file, net, w, s, dataloaders, loss_keys, acc_keys, args, evaluator, beta_loss, beta_acc)

    plot_components(
        n_components, loss_keys, acc_keys, results_loss, results_acc, beta_loss, beta_acc,
                    save_to = args.get("save_to", None), show = args.get("show", True)
    )
    return results_loss, results_acc, dir_file