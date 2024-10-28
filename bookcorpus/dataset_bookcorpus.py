import os
import torch
import pandas as pd
import numpy as numpy
from torch.utils.data import Dataset, DataLoader

from memory_transformer.bookcorpus.bookcorpus import Bookcorpus

def download_bookcorpus():
    bookcorpus = Bookcorpus()
    bookcorpus.download_and_prepare()

    return bookcorpus

def get_train_ds(bookcorpus):
    return bookcorpus.as_dataset(split="train")

class BookCorpusDataset(Dataset):

    def __init__(self, bookcorpus: Bookcorpus, transform = None):
        self.bookcorpus = bookcorpus
        self.transform = transform
        self.bookcorpus_train_data = get_train_ds(self.bookcorpus)
        self.pd_data = pd.DataFrame(self.bookcorpus_train_data)
        
    def __len__(self):
        return len(self.bookcorpus_train_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pd_sample = self.pd_data.iloc[idx]

        sample = pd_sample["text"].to_list()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
        
        


