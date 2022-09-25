
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import math

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb

from intrinsics_dimension import mle_id, twonn_pytorch
ID_functions = {"twonn" : twonn_pytorch, "mle" : mle_id}

import itertools
possible_metrics = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["accuracy", "loss"])]

from .transformer import Transformer
from .optim import CustomAdamW

class TrainableTransformer(LightningModule):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()

        self.save_hyperparameters(hparams)

        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            hparams.data_module_params.vocab_len,
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
            # skip_layer_norm = self.hparams.freeze_norm
        )

        # Intrinsic dimension params
        if self.hparams.get("ID_params", None) is None : self.hparams["ID_params"] = {}
        ID_params = {**{}, **self.hparams.get("ID_params", {"method" : "mle", "k":2})}
        id_funct = ID_functions.get(ID_params.pop("method", None), None)
        setattr(self.hparams, "ID_for_attention_weigths_and_values", ID_params.pop("attention_weigths_and_values", False))
        self.ID_function = id_funct
        self.hparams.ID = id_funct is not None
        self.ID_params = ID_params

        ####
        self.use_wandb = self.hparams.use_wandb

        # State
        self.grok = False
        self.comprehension = False
        self.memorization = False
        self.confusion = True
        self.pre_comp_epoch = float("inf") # val_accuracy > 05.0%
        self.pre_memo_epoch = float("inf") # train_accuracy > 05.0%
        self.comp_epoch = float("inf") # val_accuracy > 99.0%
        self.memo_epoch = float("inf") # train_accuracy > 99.0%

        self.set_states()

        # Early stopping grokking : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`
        early_stopping_grokking = self.hparams.early_stopping_grokking
        if type(early_stopping_grokking) != dict : early_stopping_grokking = {} 
        self.es_patience = early_stopping_grokking.get("patience", self.hparams.max_epochs)
        self.es_metric = early_stopping_grokking.get("metric", "val_accuracy") 
        assert self.es_metric in possible_metrics
        self.es_metric_threshold = early_stopping_grokking.get("metric_threshold", 0.0 if 'loss' in self.es_metric else 99.0) 
        self.es_mode = (lambda s : "min" if 'loss' in s else 'max')(self.es_metric)
        self.es_step = 0
        self.reached_limit = False

    def _scheduler_lr(self, step: int) -> float:
        """
        Used by pytorch_lighting

        :returns: the learning_rate for this training step
        """
        max_lr = self.hparams.max_lr  # type: ignore
        min_lr = self.hparams.max_lr / 10  # type: ignore
        warmup_steps = self.hparams.warmup_steps  # type: ignore
        if not self.hparams.anneal_lr:
            if step < warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step < warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step < self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
                # lr = max_lr - ((effective_step / max_effective_step) * (max_lr - min_lr))
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        """
        Used by pytorch_lighting

        :returns: optimizers and schedulers.
        """
        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=1, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        elif self.hparams.opt == 'adamw':
            optimizer = CustomAdamW(
                self.parameters(),
                betas=(0.9, 0.98),
                eps=1e-8,
                lr=1,
                weight_decay=self.hparams.weight_decay,
                noise_factor=self.hparams.noise_factor,
                weight_decay_form=self.hparams.weight_decay_kind,
            )
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def forward(self, *args, **kwargs) -> Any:
        """Passes all arguments directly to Tranformer.forward()"""
        return self.transformer(*args, **kwargs)
        
    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """

        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-2).indices  # batchsize x num_rhs_tokens
        row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = row_accuracy.float() * 100  # shape: batchsize
        return accuracy

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        data_size: int,
        reduction: str = "mean",
        grads: bool = False,
    ) :
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probabilities for the solutions to the equations
                  A list lists of hidden states by layer (including embedding layer)
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
                  Margin for this batch
        """
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        y_hat, hidden_states, attentions, values = self(
            x=x, save_activations=self.hparams.save_activations or self.hparams.ID_for_attention_weigths_and_values  # type: ignore
        )  # shape = batchsize * context_len * vocab_size
        y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len

        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.hparams.data_module_params.eq_token_index 
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_rhs = y[..., eq_position + 1 :]
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        x_lhs = x[..., : eq_position + 1]

        coeff = float(batch["target"].shape[0]) / data_size

        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        with torch.no_grad():
            acc = self._accuracy(y_hat_rhs, y_rhs)
            if reduction == "mean":
                acc = acc.mean()

        grad_vec = None
        if grads:
            loss.backward()
            for p in self.parameters():
                p.grad.data.div_(batch["text"].shape[0])
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
            return loss, grad_vec

        return loss, acc, coeff, x_lhs, y_hat_rhs, hidden_states, attentions, values

    def training_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward training pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions, hidden states,
                  attentions, and values
        """
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0

        start = time.time()
        data_size = self.hparams.data_module_params.train_data_size
        #data_size = len(self.trainer.train_dataloaders[0].dataset)
        loss, accuracy, coeff, x_lhs, y_hat_rhs, hidden_states, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, data_size=data_size
        )

        self.fwd_time_in_epoch += time.time() - start

        output = {
            "loss": loss,
            "partial_train_loss": coeff * loss,
            "partial_train_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "hidden_states" : hidden_states,
            "partial_attentions": attentions,
            "partial_values": values
        }

        #schedulers = self.trainer.lr_schedulers[0]
        #lr = 0 #schedulers["scheduler"].optimizer.param_groups[0]["lr"]
        schedulers = self.lr_schedulers()
        if schedulers is not None :
            try : scheduler = schedulers[0]
            except TypeError: scheduler = schedulers # 'xxx' object is not subscriptable
            param_groups = scheduler.optimizer.param_groups
            lr = param_groups[0]["lr"]
            output["learning_rate"] = torch.tensor([lr])

        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def validation_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """    
        with torch.no_grad():
            #data_size = self.hparams.data_module_params.val_data_size
            data_size = len(self.trainer.val_dataloaders[0].dataset)
            loss, accuracy, coeff, x_lhs, y_hat_rhs, hidden_states, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, data_size=data_size
            )
        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
            "hidden_states" : hidden_states
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def test_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """
        data_size = self.hparams.data_module_params.val_data_size
        #data_size = len(self.trainer.val_dataloaders[0].dataset)
        loss, accuracy, coeff, x_lhs, y_hat_rhs, hidden_states, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, data_size=data_size, reduction="none"
        )
        output = {
            "partial_test_loss": coeff * loss,
            "partial_test_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
            "hidden_states" : hidden_states
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def _save_inputs(self, outputs: Dict, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        :param ds: a string ('train' or 'val') naming which dataset
                   these inputs are from.
        :param train: True is this is a training batch, false otherwise
        """
        logdir = self.hparams.logdir + "/inputs/" + ds  # type: ignore
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = torch.cat([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        # num_batches = len(partial_activations)
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    # # print(f"head_attn = {head_attn}")
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: Dict, ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        """

        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:  # type: ignore
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:  # type: ignore
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:  # type: ignore
            logdir = self.hparams.logdir + "/outputs/" + ds  # type: ignore
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)
        return output.get("attentions", None), output.get("values", None)

    def _group_hidden_states(self, outputs):
        """
        Merges the hidden states from all batches in this epoch.

        :param partial_activations: A list of (lists of hidden states by layer)
        :returns: A lists hiddens states by layer

        hidden_states : (nlayers+1)x(batch_size, seq_len, dim)
        """ 
        hidden_states = [
            torch.cat([output["hidden_states"][l] for output in outputs], dim=0) # (batch_size, seq_len, dim)
            for l in range(len(outputs[0]["hidden_states"]))
        ]
        return hidden_states

    def intrinsic_dimension(self, outputs, prefix, attentions = None, values = None):
        """
        Estimate intrinsic dimensions using all hidden states collected across one epoch
        hidden_states : (nlayers+1)x(batch_size, seq_len, dim)
        
        """
        result = {}
        hidden_states = self._group_hidden_states(outputs)
        batch_size = hidden_states[0].size(0)
        for l in range(len(hidden_states)): 
            h = hidden_states[l] # (batch_size, seq_len, dim)
            h = h.view(batch_size, -1) # (batch_size, seq_len*dim)
            result[f"{prefix}ID_layer_{l}"] = self.ID_function(data=h, **self.hparams.ID_params)
        if self.hparams.ID_for_attention_weigths_and_values:
            if attentions is None : attentions = self._merge_batch_activations(list([o["partial_attentions"] for o in outputs]))
            if values is None : values = self._merge_batch_activations(list([o["partial_values"] for o in outputs]))
            num_layers = len(attentions)
            num_heads = len(attentions[0])
            for l in range(num_layers):
                for h in range(num_heads):
                    result[f"{prefix}ID_attn_layer_{l}_head_{h}"] = self.ID_function(data=attentions[l][h].view(batch_size, -1), **self.hparams.ID_params)
                    result[f"{prefix}ID_value_layer_{l}_head_{h}"] = self.ID_function(data=values[l][h].view(batch_size, -1), **self.hparams.ID_params)            
        return result

    def increase_es_limit(self, logs):
        es_metric = logs[self.es_metric]
        self.reached_limit = self.reached_limit or (es_metric >= self.es_metric_threshold if self.es_mode == "max" 
                                                    else es_metric <= self.es_metric_threshold)
        if self.reached_limit : self.es_step+=1
        return self.es_step

    def set_states(self, states : dict = None):
        if states is None :
            self.states = {
                "grok":int(self.grok), "comprehension":int(self.comprehension), 
                "memorization":int(self.memorization), "confusion":int(self.confusion),
                "pre_comp_epoch":self.pre_comp_epoch, "pre_memo_epoch":self.pre_memo_epoch,
                "comp_epoch":self.comp_epoch, "memo_epoch":self.memo_epoch,
            }
        else :
            for k, v in states.items() :
                assert k in self.states
                setattr(self, k, v)
            self.set_states()

    def is_grok(self, delay : int) :
        diff_epoch = self.comp_epoch - self.memo_epoch
        if not math.isnan(diff_epoch) : return diff_epoch >= delay
        return False

    def training_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward training passes in this epoch

        :param outputs: a list of dicts from self.training_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions, hidden states,
                  attentions, and values
        """
        epoch_is_to_be_logged = True
        if epoch_is_to_be_logged:   
            with torch.no_grad():
                loss = torch.stack([x["partial_train_loss"] for x in outputs])
                perplexity = torch.exp(loss.sum())
                loss = loss.mean()
                accuracy = torch.stack(
                    [x["partial_train_accuracy"] for x in outputs]
                ).mean()
            # avg_lr = torch.stack([x["learning_rate"] for x in outputs]).mean()
            # max_lr = torch.stack([x["learning_rate"] for x in outputs]).max()
            # last_lr = outputs[-1]["learning_rate"]
            first_lr = outputs[0]["learning_rate"]

            attentions, values = None, None
            if self.hparams.save_activations or self.hparams.save_outputs or self.hparams.save_checkpoint:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                if self.hparams.save_activations or self.hparams.save_outputs:
                    attentions, values = self._save_activations(outputs, ds="train")
            
            id_output = {}
            if self.hparams.ID : 
                id_output = self.intrinsic_dimension(outputs, "train_", attentions, values)
                # TOCHANGE
                id_output[f"ID_last_layer_weights_train"] = self.ID_function(data=self.transformer.linear.weight, **self.hparams.ID_params)

            logs = {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_perplexity": perplexity,
                "learning_rate": first_lr,
                "len_train_ds": self.hparams.data_module_params.train_data_size,
                "len_val_ds": self.hparams.data_module_params.val_data_size,
                "batches_per_epoch_train": self.hparams.data_module_params.batches_per_epoch_train,
                "time_per_epoch": time.time() - self.training_epoch_start_time,
                "fwd_time_in_epoch": self.fwd_time_in_epoch,
            }
            logs = {**id_output, **logs}

            ##
            if 'train' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

            if accuracy >= 5.0 : self.pre_memo_epoch = min(self.current_epoch, self.pre_memo_epoch)

            memo_condition = accuracy >= 99.0
            self.memorization = self.memorization or memo_condition
            if memo_condition : self.memo_epoch = min(self.current_epoch, self.memo_epoch)
            ##

            for k, v in logs.items():
                self.log(k, v, prog_bar="loss" in k or "accuracy" in k)

            if self.hparams.use_wandb:
                db_data = {"epoch": self.current_epoch, "train loss": loss.detach(), "train accuracy": accuracy, 'lr': first_lr}
                db_data = {**db_data, **id_output}
                wandb.log(db_data)

        if self.hparams.data_module_params.data_flag and self.current_epoch % 1 == 0:
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.checkpoint_path,
                    "epoch_" + str(self.current_epoch) + ".ckpt",
                )
            )
        
    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        validation_is_real = True
        if not outputs : validation_is_real = False

        if validation_is_real:

            loss = torch.stack([x["partial_val_loss"] for x in outputs])
            perplexity = torch.exp(loss.sum())
            loss = loss.mean()
            accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).mean()

            attentions, values = None, None
            if self.hparams.save_activations or self.hparams.save_outputs or self.hparams.save_checkpoint:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                if self.hparams.save_activations or self.hparams.save_outputs:
                    attentions, values = self._save_activations(outputs, ds="val")

            id_output = {}
            if self.hparams.ID : 
                id_output = self.intrinsic_dimension(outputs, "val_", attentions, values)
                # TOCHANGE
                id_output[f"ID_last_layer_weights_val"] = self.ID_function(data=self.transformer.linear.weight, **self.hparams.ID_params)
            
            logs = {
                "val_loss": loss,
                "val_accuracy": accuracy,
                "val_perplexity": perplexity
            }
            logs = {**id_output, **logs}

            ##
            if 'val' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

            if accuracy >= 5.0 : self.pre_comp_epoch = min(self.current_epoch, self.pre_comp_epoch)

            comp_condition = accuracy >= 99.0
            self.comprehension = self.comprehension or comp_condition
            if comp_condition : self.comp_epoch = min(self.current_epoch, self.comp_epoch)
            
            self.grok = self.comprehension and True # and long step of training
            self.memorization = (not self.comprehension) and self.memorization
            self.confusion = (not self.comprehension) and (not self.memorization)

            #self.grok = self.is_grok(delay = 100)
            #self.comprehension = not self.grok

            self.set_states()
            torch.save(self.states, self.hparams.logdir + "/states.pt")
            ##

            for k, v in logs.items():
                self.log(k, v, prog_bar="loss" in k or "accuracy" in k)

            if self.hparams.use_wandb:
                db_data = {"epoch": self.current_epoch, "val loss": loss.detach(), "val accuracy": accuracy,
                           "es_step" : self.es_step}
                db_data = {**db_data, **id_output}
                wandb.log(db_data)
  
        if validation_is_real:
            return logs

    def test_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        loss = torch.cat([x["partial_test_loss"] for x in outputs], dim=0)  # .sum()
        # loss = list([x["partial_test_loss"] for x in outputs])  # .sum()
        perplexity = torch.exp(loss)
        accuracy = torch.cat([x["partial_test_accuracy"] for x in outputs], dim=0)

        id_output = {}
        # if self.hparams.ID : id_output = self.intrinsic_dimension(outputs, prefix="test_")

        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_perplexity": perplexity,
        }
        logs = {**id_output, **logs}
        if self.hparams.use_wandb:
            db_data = {"epoch": self.current_epoch, "test loss": loss.detach(), "test accuracy": accuracy}
            db_data = {**db_data, **id_output}
            wandb.log(db_data) 

        return logs

    def send_dict_to_wandb(self, data, label, title) :
        if self.hparams.use_wandb:  
            labels = data.keys()
            values = data.values()
            data = [[label, val] for (label, val) in zip(labels, values)]
            table = wandb.Table(data=data, columns = ["label", "value"])
            wandb.log({label : wandb.plot.bar(table, "label", "value", title=title)})

    def on_train_start(self):
        self.trainer.save_checkpoint(
            os.path.join(
                self.hparams.checkpoint_path,
                "init.ckpt",
            )
        )
        if self.hparams.use_wandb:
            db_data = {
                "base_length" : self.hparams.data_module_params.base_length,

                "train_batchsize" : self.hparams.data_module_params.train_batchsize,
                "batches_per_epoch_train" : self.hparams.data_module_params.batches_per_epoch_train,
                "len_train_data": self.hparams.data_module_params.train_data_size,
                    
                "val_batchsize" : self.hparams.data_module_params.val_batchsize,
                "batches_per_epoch_val" : self.hparams.data_module_params.batches_per_epoch_val,
                "len_val_data": self.hparams.data_module_params.val_data_size,
            }   
            self.send_dict_to_wandb(db_data, label = "data_info", title="Dataset Informations")
            
            if self.hparams.watch:
                wandb.watch(
                    self.transformer,
                    #criterion=None,
                    log = self.hparams.watch.log,
                    log_freq = self.hparams.watch.log_freq,
                    #idx: = None,
                    #log_graph = False
                )

    def on_train_end(self) :
        # self.grok = self.is_grok(delay = 100)
        # self.comprehension = not self.grok

        self.set_states()
        print("="*10)
        print(self.states)
        print("="*10)
        self.send_dict_to_wandb(self.states, label = "states_info", title="Phase Informations")

    # def on_after_backward(self):
    #     # example to inspect gradient information in tensorboard
    #     if self.trainer.global_step % 1e9 == 0:  # don't make the tf file huge
    #         grad_vec = None
    #         for p in self.parameters():
    #             #p.grad.data.div_(batch["text"].shape[0])
    #             if grad_vec is None:
    #                 grad_vec = p.grad.data.view(-1)
    #             else:
    #                 grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
        
    #     print(grad_vec.shape)