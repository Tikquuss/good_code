import os
from argparse import ArgumentParser, Namespace

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.utils import bool_flag, str2dic_all, init_wandb, AttrDict, intorstr
from src.data import DEFAULT_DATA_DIR
from src.dataset import DataModule
from src.modeling import TrainableTransformer

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = ArgumentParser(description="")

    # Main parameters
    parser.add_argument("--validation_metric", type=str, default="val_accuracy", help="Validation metrics : val_accuracy, val_loss ...")

    # Devices & Seed
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator types : cpu, gpu, tpu, ipu, auto") 
    parser.add_argument("--devices", type=intorstr, default="auto", help="number of cpu processes, of gpu/tpu cores ...")
    parser.add_argument("--random_seed", type=int, default=-1)

    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)

    # Dataset params
    parser.add_argument("--math_operator", type=str, default="+")
    parser.add_argument("--operand_length", type=int, help="for list operations, the length of the lists")
    parser.add_argument("--train_data_pct", type=float, default=5)
    parser.add_argument(
        "--batchsize",
        type=float,
        # default=0.25,
        default=0,
        help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    # Model params    
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_noise", type=float, default=0.0)
    parser.add_argument("--non_linearity", type=str, default="relu")
    parser.add_argument("--max_context_len", type=int, default=50)

    # Training params
    parser.add_argument("--save_activations", type=bool_flag, default=True)
    parser.add_argument("--save_outputs", type=bool_flag, default=False)
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
    )
    parser.add_argument("--save_checkpoint", type=bool_flag, default=True)     
    parser.add_argument("--load_from_ckpt", type=str, default=None)

    # Optimizer
    parser.add_argument("--opt", type=str, default="adamw", choices=("sgd", "adamw"))
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--anneal_lr_steps", type=int, default=100000)
    parser.add_argument("--anneal_lr", type=bool_flag, default=False)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
    parser.add_argument("--noise_factor", type=float, default=0)

    # wandb
    parser.add_argument("--use_wandb", type=bool_flag, default=False)
    parser.add_argument("--group_name", type=str, default="base")
    parser.add_argument("--wandb_entity", type=str, default=None, help="name of the team on wandb and is optional")
    parser.add_argument("--wandb_project", type=str, default=None, help="name of the project")

    # Early stopping 
    parser.add_argument("--early_stopping_patience", type=int, default=1e9)
    parser.add_argument("--patience_metric", type=str, default="val_accuracy", 
                help="train_loss, train_accuracy, val_loss, val_accuracy, ...")
    parser.add_argument("--early_stopping_step_val_acc_threshold", type=float, default=90.0)

    # Early_stopping (stop training after grokking)
    parser.add_argument("--early_stopping_grokking", type=str2dic_all, default="", help="""
        * eg. : "patience=int(1000),metric=str(val_accuracy),metric_threshold=float(90.0)"
        * Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`"
        """)
    
    return parser

class StopTrainingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.early_stopping_step >= pl_module.hparams.early_stopping_patience :
            #exit()
            raise KeyboardInterrupt


class GenerateCallback(pl.Callback):
    """Use to plot the learned input embeddings at different training stages"""
    
    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
    #def on_train_epoch_end(self, trainer, pl_module) :
    #def on_validation_epoch_end(self, trainer, pl_module) :
        pass
        #current_epoch = trainer.current_epoch
        #if current_epoch % self.every_n_epochs == 0 :
         #   pass

def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """
    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        pl.seed_everything(hparams.random_seed, workers=True)

    # set up wandb
    init_wandb(hparams)  
    data_flag = False
    device = "cpu"
    data_module = DataModule(
        train_data_pct = hparams.train_data_pct,  
        math_operator = hparams.math_operator,
        operand_length = hparams.operand_length,
        data_dir = hparams.datadir,
        batch_size = hparams.batchsize,
        device = device,
        flag=data_flag
    )

    train_dataset = data_module.train_dataset
    data_module.train_dataloader()
    data_module.val_dataloader()
    hparams.data_module_params = AttrDict({
        "vocab_len" : len(data_module.tokenizer),
        "eq_token_index" : data_module.tokenizer.stoi["="],
        "base_length" : data_module.base_length,

        "train_data_size" : len(train_dataset),
        "train_batchsize" : data_module.train_batchsize,
        "batches_per_epoch_train" : data_module.batches_per_epoch_train,

        "val_data_size" : len(data_module.val_dataset),
        "val_batchsize" : data_module.val_batchsize,
        "batches_per_epoch_val" : data_module.batches_per_epoch_val,
    })

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    ## 
    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    model = TrainableTransformer(hparams).float()

    logger = CSVLogger(hparams.logdir)

    root_dir = hparams.logdir
    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": 5, # hparams.max_epochs, 

        "val_check_interval": 1.0,
        #"profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        #"log_every_n_steps": 1,
        #"flush_logs_every_n_steps": 1000,

        "default_root_dir" : root_dir,

        "accelerator" : hparams.accelerator,
        "devices" : hparams.devices,
        #"reload_dataloaders_every_n_epochs" : True,
        "weights_summary":"full", # "top", None,
    }

    callbacks = []
    if not data_flag and False :
        patience_metric = hparams.patience_metric
        mode = (lambda s : "min" if 'loss' in s else 'max')(patience_metric)
        early_stopping_callback = EarlyStopping(
            monitor=patience_metric, patience=hparams.early_stopping_patience, verbose=False, strict=True,
            mode = mode
        )

        validation_metric = hparams.validation_metric
        mode = (lambda s : "min" if 'loss' in s else 'max')(validation_metric)
        hparams.save_top_k = -1
        model_checkpoint_callback = ModelCheckpoint(
                dirpath=root_dir,
                save_weights_only=True,
                filename="{epoch}-{%s:.4f}"%validation_metric,
                mode = mode,
                monitor=validation_metric,
                save_top_k=hparams.save_top_k,
        )

        callbacks += [early_stopping_callback, model_checkpoint_callback]
    
    callbacks += [
        #GenerateCallback(), 
        #pl.callbacks.LearningRateMonitor("epoch");
        #StopTrainingCallback()
    ]

    trainer_args["callbacks"] = callbacks
    
    trainer = Trainer(**trainer_args) #, progress_bar_refresh_rate=0
 
    # #torch.save(model, os.path.join(checkpoint_path, "init.pt"))
    # trainer.save_checkpoint(
    #     os.path.join(
    #         model.hparams.checkpoint_path ,
    #         "init.ckpt",
    #     )
    # )

    hparams.eval_only = False
    if not hparams.eval_only :
        # Training
        print("Training starts...")
        model.train()
        trainer.fit(model, datamodule=data_module, ckpt_path=hparams.load_from_ckpt)
        print("Training completed.")
        if not data_flag :
            print("Testing starts....")
            model.eval()
            r = trainer.test(model, datamodule=data_module)
            print(r)
            print("Testing completed.")
    else :
        hparams.eval_split = "validation"
        if not data_flag :
            # Evaluation
            print("Evaluation starts....")
            if hparams.eval_split == "train":
                data_module.test_dataloader = data_module.train_dataloader
            elif hparams.eval_split == "validation" :
                data_module.test_dataloader = data_module.val_dataloader
            model.eval()
            #r = trainer.test(model, datamodule=data_module, ckpt_path=hparams.load_from_ckpt)
            r = trainer.validate(model, datamodule=data_module, ckpt_path=hparams.load_from_ckpt)
            print(r)
            print("Evaluation completed.")

    return hparams.logdir

if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_parser().parse_args()
    print()
    for k, v in vars(params).items() : print(k, " --> ", v)
    print()

    # run experiment
    train(params)