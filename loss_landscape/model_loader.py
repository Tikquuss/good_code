"""
Load a pytorch lightning module checkpoint
"""

"""
Load a pytorch lightning module checkpoint
"""

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
# hparams = AttrDict({
#     "random_seed" : 0,
#     "gpu" : -1,
#     "max_epochs" : None,
#     "max_steps" : 10,
#     "use_cuda" : True,
#     "batchsize" : -1.0,
#     "n_layers" : 2,
#     "n_heads" : 4,
#     "d_model" : 128,
#     "dropout" : 0.0,
#     "weight_noise" : 0.0,
#     "non_linearity" : "relu",
#     "max_context_len" : 50,
#     "math_operator" : "+",
#     "operand_length" : None,
#     "train_data_pct" : 90.0,
#     "warmup_steps" : 10,
#     "anneal_lr_steps" : 100000,
#     "anneal_lr" : False,
#     "max_lr" : 0.001,
#     "weight_decay" : 1.0,
#     "weight_decay_kind" : "to_zero",
#     "noise_factor" : 0.0,
#     "save_activations" : False,
#     "save_outputs" : False,
#     "logdir" : "/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+",
#     "datadir" : "/content/data/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+",
#     "save_checkpoint" : True,
#     "load_from_ckpt" : None,
#     "opt" : "adamw",
#     "momentum" : 0.9,
#     "use_wandb" : False,
#     "group_name" : "tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+",
#     "wandb_entity" : "grokking_ppsp",
#     "wandb_project" : "grokking_operator=+",
#     "early_stopping_patience" : 5000,
#     "patience_metric" : "val_accuracy",
#     "early_stopping_step_val_acc_threshold" : 90.0,
# })


hparams = AttrDict({
    'random_seed': 0,
    'gpu': -1,
    'max_epochs': None,
    'max_steps': 10,
    'use_cuda': True,
    'batchsize': -1.0,
    'n_layers': 2,
    'n_heads': 4,
    'd_model': 128,
    'dropout': 0.0,
    'weight_noise': 0.0,
    'non_linearity': 'relu',
    'max_context_len': 50,
    'math_operator': '+',
    'operand_length': None,
    'train_data_pct': 90.0,
    'warmup_steps': 10,
    'anneal_lr_steps': 100000,
    'anneal_lr': False,
    'max_lr': 0.001,
    'weight_decay': 1.0,
    'weight_decay_kind': 'to_zero',
    'noise_factor': 0.0,
    'save_activations': False,
    'save_outputs': False,
    'logdir': '/content/logs/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+',
    'datadir': '/content/data/tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+',
    'save_checkpoint': True,
    'load_from_ckpt': None,
    'opt': 'adamw',
    'momentum': 0.9,
    'use_wandb': False,
    'group_name': 'tdp=90-wd=1-d=0.0-opt=adamw-mlr=0.001-mo+',
    'wandb_entity': 'grokking_ppsp',
    'wandb_project': 'grokking_operator=+',
    'early_stopping_patience': 5000,
    'patience_metric': 'val_accuracy',
    'early_stopping_step_val_acc_threshold': 90.0,
    'data_module_params': AttrDict({
        'vocab_len': 239,
        'eq_token_index': 1,
        'base_length': 9409,
        'train_data_size': 8468,
        'train_batchsize': 8468,
        'batches_per_epoch_train': 1,
        'val_data_size': 941,
        'val_batchsize': 941,
        'batches_per_epoch_val': 1,
        "data_flag" : True
    }),
    'early_stopping_grokking': 1000000000.0
})

def load(lightning_module_class, model_file):
    #return lightning_module_class.load_from_checkpoint(model_file)
    return lightning_module_class.load_from_checkpoint(hparams = hparams, checkpoint_path = model_file).float()