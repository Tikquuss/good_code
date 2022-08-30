import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from .data import (
    ArithmeticDataset,
    ArithmeticIterator,
)

class ArithmeticDataset4Loader(Dataset):
    def __init__(self, data):
        self.text = data[:,:-1]
        self.target = data[:,1:]
        
    def __getitem__(self, index):
        x = self.text[index]
        y = self.target[index]
        return {'text': x, 'target': y}
    
    def __len__(self):
        return len(self.text)
        
class DataModule(pl.LightningDataModule):
    """Dataset"""
    def __init__(
        self,
        train_data_pct : int,  
        math_operator : str,
        operand_length : int,
        data_dir : str,
        batch_size : int,
        device,
        flag = True
    ):
        """
        Params :
        """
        super(DataModule, self).__init__()

        self.train_data_pct = train_data_pct  
        self.math_operator = math_operator
        self.operand_length = operand_length
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device

        self.flag = flag

        self.prepare_data()

    def prepare_data(self) -> None:
        """
        Used by pytorch_lighting

        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        self.train_dataset, self.val_dataset, eqs, self.base_length = ArithmeticDataset.splits(
            train_pct=self.train_data_pct,  
            operator=self.math_operator,  
            operand_length=self.operand_length,
            data_dir=self.data_dir, 
        )
        self.tokenizer = self.train_dataset.tokenizer
        if not self.flag :
            if True :
                #data = torch.cat([self.train_dataset.data, self.val_dataset.data], dim=0) # (n_train + n_val, 7)
                #x = data[:,:-1]
                #y = data[:,1:]
                #dataset = TensorDataset(x, y)
                self.train_dataset = ArithmeticDataset4Loader(self.train_dataset.data)
                self.val_dataset = ArithmeticDataset4Loader(self.val_dataset.data)
            else :
                dataset = eqs
                train_pct = self.train_data_pct
                n = len(dataset)
                train_size = train_pct * n // 100
                val_size = n - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        if self.flag :
            iterator = ArithmeticIterator(
                self.train_dataset,
                self.device,
                batchsize_hint=self.batch_size,  # type: ignore
            )
            self.train_batchsize = iterator.batchsize
        else :
            batch_size = self.batch_size
            batch_size = ArithmeticIterator.calculate_batchsize(
                len(self.train_dataset), batchsize_hint=batch_size
            )
            train_dataset = self.train_dataset
            self.train_batchsize = min(batch_size, len(train_dataset))
            iterator = DataLoader(
                self.train_dataset, 
                batch_size=self.train_batchsize, 
                shuffle=True, 
                drop_last=False, 
                pin_memory=True, 
                num_workers=4
            )

        self.batches_per_epoch_train = len(iterator)
        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        if self.flag :
            iterator = ArithmeticIterator(
                self.val_dataset,
                self.device,
                batchsize_hint=-1,  # no need to batch validation data
            )
            self.val_batchsize = iterator.batchsize
        else :
            batch_size = self.batch_size
            batch_size = ArithmeticIterator.calculate_batchsize(
                len(self.val_dataset), batchsize_hint=batch_size
            )
            self.val_batchsize = min(batch_size, len(self.val_dataset))
            iterator = DataLoader(
                self.val_dataset, 
                batch_size=self.val_batchsize, 
                shuffle=True, 
                drop_last=False, 
                pin_memory=True, 
                num_workers=4
            )
        self.batches_per_epoch_val = len(iterator)
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        if self.flag :
            iterator = ArithmeticIterator(
                self.val_dataset, self.device, batchsize_hint=-1  # type: ignore
            )
            self.test_batchsize = iterator.batchsize
        else :
            batch_size = self.batch_size
            batch_size = ArithmeticIterator.calculate_batchsize(
                len(self.val_dataset), batchsize_hint=batch_size
            )
            self.test_batchsize = min(batch_size, len(self.val_dataset))
            iterator = DataLoader(
                self.val_dataset, 
                batch_size=self.test_batchsize, 
                shuffle=True, 
                drop_last=False, 
                pin_memory=True, 
                num_workers=4
            )
        self.batches_per_epoch_test = len(iterator)
        return iterator

    def predict_dataloader(self) -> ArithmeticIterator: 
        iterator = ArithmeticIterator(
            self.val_dataset, self.device, batchsize_hint=-1  # type: ignore
        )
        return iterator


if __name__ == '__main__':
    data_module = DataModule(
        train_data_pct = 80,  
        math_operator = "+",
        operand_length = 50,
        data_dir = "",
        #batch_size = -1,
        batch_size = 12,
        device = "cuda",
        flag  = True
    )

    #print(data_module.train_dataset)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    for batch in train_dataloader :
        print(batch)
        break