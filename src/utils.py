import argparse
import wandb

wandb_id = None

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def init_wandb(hparams, resume=False):
    global wandb_id
    print('='*5, "init wandb", '='*5)
    if hparams.use_wandb:
        group_vars = ["d_model", "n_heads", "random_seed", "max_steps", "max_epochs", "n_layers", "dropout", "max_context_len", "weight_noise", "train_data_pct", "math_operator", "operand_length", "weight_decay", "noise_factor", "weight_decay_kind", "batchsize", "max_lr", "warmup_steps", "anneal_lr", "anneal_lr_steps", "opt", "momentum"]

        group_name = ''
        for var in group_vars:
            group_name = group_name + '_' + var + str(getattr(hparams, var))
        run = wandb.init(
            project=hparams.wandb_project,
            entity=hparams.wandb_entity,
            group=hparams.group_name,
            #name=group_name, # too_long_for_that
            notes=group_name,
            resume=True if resume else None,
            id = wandb_id if resume else None
        )
        if wandb_id is None : wandb_id = run.id
        for var in group_vars:
            wandb.config.update({var:getattr(hparams, var)}) 

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def intorstr(s):
    try : return int(s)
    except ValueError : return s

def to_none(a):
    return None if not a or a == "_None_" else a

def str2dic(s, _class = str):
    """`a=x,b=y` to {a:x, b:y}"""
    s = to_none(s)
    if s is None :
        return s
    if s:
        params = {}
        for x in s.split(","):
            split = x.split('=')
            assert len(split) == 2
            params[split[0]] = _class(split[1])
    else:
        params = {}

    return AttrDict(params)

def str2dic_int(s):
    return str2dic(s, int)
        
def str2dic_all(s) :
    """`a=int(0),b=str(y),c=float(0.1)` to {a:0, b:y, c:0.1}"""
    all_class = {"int" : int, "str" : str, "float" : float, 'bool' : bool_flag}
    s = to_none(s)
    if s is None :
        return s
    if s:
        params = {}
        for x in s.split(","):
            split = x.split('=')
            assert len(split) == 2
            val = split[1].split("(")
            _class = val[0]
            val = split[1][len(_class)+1:][:-1]
            params[split[0]] = all_class[_class](val)
    else:
        params = {}

    return AttrDict(params)

def str2list(s):
    """`a,b` to [a, b]"""
    s = to_none(s)
    return s if s is None else s.split(",")