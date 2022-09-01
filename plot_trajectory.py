import torch
import argparse

from loss_landscape.plot_trajectory import plot_trajectory
from src.modeling import TrainableTransformer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    args = parser.parse_args()

    lightning_module_class = TrainableTransformer

    ###########
    import re
    import os 

    def sorted_nicely(l): 
        """ Sort the given iterable in the way that humans expect.
        https://stackoverflow.com/a/2669120/11814682
        """ 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    pretrained_folder = "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+/checkpoints"
    #pattern = '^epoch_[0-9]+.ckpt$'
    pattern = '^epoch=[0-9]+-val_accuracy=[0-9]+\.[0-9]+.ckpt$'

    model_files = os.listdir(pretrained_folder)
    model_files = [f for f in model_files if re.match(pattern, f)]
    model_files = sorted_nicely(model_files)
    model_files = ["init.ckpt"] + model_files
    model_files = [pretrained_folder + "/" + f for f in model_files]
    ###########
    
    proj_file, dir_file = plot_trajectory(args, model_files, lightning_module_class)