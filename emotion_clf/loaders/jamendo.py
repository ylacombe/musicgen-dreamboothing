"""Dataset and dataloader constructor for jamendo"""
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pdb

class AudioFolder(torch.utils.data.Dataset):
    """Audio dataset"""
    def __init__(self,
                 root: str,
                 directory: str,
                 trackfile: str,
                 input_length: Optional[int] = None,
                 random_chunk: Optional[bool] = True) -> None:
        self.root = root
        self.directory = directory
        self.input_length = input_length
        self.random_chunk = random_chunk
        filename = os.path.join(self.root, trackfile)
        self.get_dictionary(filename)

    def __getitem__(self, index):
        filename = os.path.join(self.directory, self.dictionary[index]['path'][:-3] + 'npy')
                   #os.path.join(self.root, self.directory, self.dictionary[index]['path'][:-3] + 'npy')

        if self.input_length is None:
            audio = np.load(filename)
        else:
            memmap = np.load(filename, mmap_mode='r', allow_pickle=True)
            length = memmap.shape[-1]
            pos = np.random.random_sample() if self.random_chunk else 0.5
            idx = int(pos * (length - self.input_length))
            audio = np.array(memmap[:, idx:idx + self.input_length])

        tags = self.dictionary[index]['tags']
        return audio.astype('float32'), tags.astype('float32')

    def get_dictionary(self, filename: str) -> None:
        with open(filename, 'rb') as file:
            dictionary = pickle.load(file)
            self.dictionary = dictionary

    def __len__(self) -> int:
        return len(self.dictionary)


def get_dataloader(root: str,
                   dataset_params: Dict[str, Any],
                   dataloader_params: Dict[str, Any],
                   num_workers: int = 0) -> DataLoader:
    dataset = AudioFolder(root, **dataset_params)

    return torch.utils.data.DataLoader(dataset,
                                       pin_memory=True,
                                       num_workers=num_workers,
                                       **dataloader_params)
