"""
    1D plotting routines
"""

import imp
from matplotlib import pyplot as pp
import h5py
import argparse
import numpy as np
import math

def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, loss_max=None, acc_max = None, log=False, show=False, save_to = None):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    y_loss_min, y_loss_max = min(train_loss), loss_max if loss_max else max(train_loss)
    y_acc_min, y_acc_max = min(train_acc), acc_max if acc_max else max(train_acc)

    save_to = save_to if save_to else surf_file

    # loss and accuracy map
    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
    tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        y_loss_min, y_loss_max = min(y_loss_min, min(test_loss)), max(y_loss_max, max(test_loss))
        y_acc_min, y_acc_max = min(y_acc_min, min(test_acc)), max(y_acc_max, max(test_acc))
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=1)

    # err_loss = y_loss_max - y_loss_min
    # y_loss_min -= err_loss / y_loss_max
    # y_loss_max += err_loss / y_loss_max

    # err_acc = y_acc_max - y_acc_min
    # y_acc_min -= err_acc / 100
    # y_acc_max += err_acc / 100

    #if log: y_loss_min, y_loss_max = math.log(y_loss_min, math.e), math.log(y_loss_max, math.e)


    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    #ax1.set_ylim(0, loss_max)
    ax1.set_ylim(y_loss_min, y_loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    #ax2.set_ylim(0, acc_max)
    ax2.set_ylim(y_acc_min, y_acc_max)
    filename = save_to + '_1d_loss_acc' + ('_log' if log else '')
    pp.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    # train_loss curve
    fig = pp.figure()
    if log:
        pp.semilogy(x, train_loss)
    else:
        pp.plot(x, train_loss)
    pp.ylabel('Training Loss', fontsize='xx-large')
    pp.xlim(xmin, xmax)
    #pp.ylim(0, loss_max)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_train_loss' + ('_log' if log else '')
    pp.savefig(filename + '.pdf',
                dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    # train_err curve
    fig = pp.figure()
    tmp = 100.0 - train_acc
    pp.plot(x, tmp)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, acc_max)
    pp.ylim(min(tmp), max(tmp))
    pp.ylabel('Training Error', fontsize='xx-large')
    filename = save_to + '_1d_train_err'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: pp.show()
    f.close()


def plot_1d_loss_err_repeat(prefix, idx_min=1, idx_max=10, xmin=-1.0, xmax=1.0,
                            loss_max=None, acc_max = 100, show=False, save_to = None):
    """
        Plotting multiple 1D loss surface with different directions in one figure.
    """

    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()

    y_loss_min, y_loss_max = 0.0, 1.0e9
    y_acc_min, y_acc_max = 0.0, 100.0
    save_to = save_to if save_to else prefix

    for idx in range(idx_min, idx_max + 1):
        # The file format should be prefix_{idx}.h5
        f = h5py.File(prefix + '_' + str(idx) + '.h5','r')

        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        y_loss_min, y_loss_max = min(y_loss_min, min(test_loss)), max(y_loss_max, max(test_loss))
        y_acc_min, y_acc_max = min(y_acc_min, min(test_acc)), max(y_acc_max, max(test_acc))

        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        te_loss, = ax1.plot(x, test_loss, 'b--', label='Testing loss', linewidth=1)
        tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Testing accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    #ax1.set_ylim(0, loss_max)
    ax1.set_ylim(y_loss_min, y_loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    #ax2.set_ylim(0, acc_max)
    ax2.set_ylim(y_acc_min, y_acc_max)
    filename = save_to + '_1d_loss_err_repeat'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show: pp.show()


def plot_1d_eig_ratio(surf_file, xmin=-1.0, xmax=1.0, val_1='min_eig', val_2='max_eig', ymax=None, show=False, save_to = None):
    print('------------------------------------------------------------------')
    print('plot_1d_eig_ratio')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    x = f['xcoordinates'][:]

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])
    abs_ratio = np.absolute(np.divide(Z1, Z2))

    y_loss_min, y_loss_max = min(abs_ratio), ymax if ymax else max(abs_ratio)
    save_to = save_to if save_to else surf_file

    pp.plot(x, abs_ratio)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, ymax)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_eig_abs_ratio'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    ratio = np.divide(Z1, Z2)
    y_loss_min, y_loss_max = min(ratio), ymax if ymax else max(ratio)
    pp.plot(x, ratio)
    pp.xlim(xmin, xmax)
    #pp.ylim(0, ymax)
    pp.ylim(y_loss_min, y_loss_max)
    filename = save_to + '_1d_eig_ratio'
    pp.savefig(filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    pp.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    f.close()
    if show: pp.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plott 1D loss and error curves')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file contains loss values')
    parser.add_argument('--log', action='store_true', default=False, help='logarithm plot')
    parser.add_argument('--xmin', default=-1, type=float, help='xmin value')
    parser.add_argument('--xmax', default=1, type=float, help='xmax value')
    parser.add_argument('--loss_max', default=5, type=float, help='ymax value (loss)')
    parser.add_argument('--acc_max', default=100, type=float, help='ymax value (accuracy)')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--prefix', default='', help='The common prefix for surface files')
    parser.add_argument('--idx_min', default=1, type=int, help='min index for the surface file')
    parser.add_argument('--idx_max', default=10, type=int, help='max index for the surface file')

    args = parser.parse_args()

    if args.prefix:
        plot_1d_loss_err_repeat(args.prefix, args.idx_min, args.idx_max,
                                args.xmin, args.xmax, args.loss_max, args.acc_max, args.show)
    else:
        plot_1d_loss_err(args.surf_file, args.xmin, args.xmax, args.loss_max, args.acc_max, args.log, args.show)
