"""
Load a pytorch lightning module checkpoint
"""

import torch

logdir = ""
hparams = torch.load(logdir + "/hparams.pt")

def load(lightning_module_class, model_file):
    #return lightning_module_class.load_from_checkpoint(model_file)
    return lightning_module_class.load_from_checkpoint(hparams = hparams, checkpoint_path = model_file).float()