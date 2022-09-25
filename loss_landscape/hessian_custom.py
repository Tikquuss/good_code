"""
https://modelzoo.co/model/pytorch-hessian-eigenthings
https://github.com/noahgolmant/pytorch-hessian-eigenthings

pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings

######

https://github.com/amirgholami/PyHessian

#pip install --upgrade git+https://amirgholami/PyHessian.git@master#egg=PyHessian
git clone https://github.com/amirgholami/PyHessian.git
cd PyHessian
python setup.py install
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

flag = True
if flag :
    from hessian_eigenthings import compute_hessian_eigenthings
else:

    from pyhessian import hessian
    from density_plot import get_esd_plot
    import numpy as np

def get_dataloader(data_module, train : bool, hparams) :

    if train :
        dataset = data_module.train_dataset
        batch_size = data_module.train_batchsize
    else :
        dataset = data_module.val_dataset
        batch_size = data_module.val_batchsize

    x = dataset.text
    y = dataset.target

    # Note: each sample must have exactly one '=' and all of them must have it in the same position.
    eq_token_index = hparams.data_module_params.eq_token_index 
    eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
    eq_position = int(eq_position_t.squeeze())

    # only calculate loss/accuracy on right hand side of the equation
    y_rhs = y[..., eq_position + 1 :]

    dataloader  = DataLoader(
        TensorDataset(x, y_rhs), 
        batch_size=batch_size, shuffle=True, drop_last=False
    )

    return dataloader, y


class Model4Hessian(nn.Module):
    """costomized linear layer"""
    def __init__(self, pl_module):
        super(Model4Hessian, self).__init__()
        self.pl_module = pl_module

    def forward(self, x):
        y_hat, _, _, _ = self.pl_module(x=x, save_activations=False)  # shape = batchsize * context_len * vocab_size
        y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size * context_len

        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.pl_module.hparams.data_module_params.eq_token_index 
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_hat_rhs = y_hat[..., eq_position + 1 :]
        
        return y_hat_rhs

############################################
############################################

torch.manual_seed(0)

N = 32
batch_size = 32
x = torch.randn(batch_size, 3)
y = torch.zeros(N, dtype=torch.long)
dataset = TensorDataset(x, y)
dataloader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

model = nn.Linear(3,4)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.CrossEntropyLoss()

flag = True

if flag :
    num_eigenthings = 20  # compute top 20 eigenvalues/eigenvectors
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model, dataloader, criterion, num_eigenthings,
        full_dataset=True,
        mode="power_iter",
        use_gpu=False,
        fp16=False,
        #max_possible_gpu_samples=2**16,
    )
else :
    hessian_dataloader = [x, y]

    # turn model to eval mode
    model.eval()

    hessian_comp = hessian(model,
                           criterion,
                           data=hessian_dataloader,
                           cuda=False)

    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = hessian_comp.trace()
    density_eigen, density_weight = hessian_comp.density()

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))

    get_esd_plot(density_eigen, density_weight)


############################################
############################################

# TODO
model = None 
data_module = None

model_4_hessian = Model4Hessian(pl_module=model)
model_4_hessian = torch.nn.DataParallel(model_4_hessian)

dataloader, y = get_dataloader(data_module, train = True, hparams = model.hparams)

criterion = nn.CrossEntropyLoss()

if flag :
    num_eigenthings = 20  # compute top 20 eigenvalues/eigenvectors
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model_4_hessian, dataloader, criterion, num_eigenthings,
        full_dataset=True,
        mode="power_iter",
        use_gpu=False,
        fp16=False,
        #max_possible_gpu_samples=2**16,
    )

else :
    batch_num = len(dataloader)

    if batch_num == 1:
        for inputs, labels in dataloader:
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(dataloader):
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1: break

    # turn model to eval mode
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model_4_hessian,
                            criterion,
                            data=hessian_dataloader,
                            cuda=False)
    else:
        hessian_comp = hessian(model_4_hessian,
                            criterion,
                            dataloader=hessian_dataloader,
                            cuda=False)

    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = hessian_comp.trace()
    density_eigen, density_weight = hessian_comp.density()

    print('\n***Top Eigenvalues: ', top_eigenvalues)
    print('\n***Trace: ', np.mean(trace))

    get_esd_plot(density_eigen, density_weight)