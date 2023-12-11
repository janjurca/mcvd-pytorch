# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from .h5 import HDF5Dataset
import glob
from os import path

class MeteoDataset(Dataset):
    FILE_APPEND = ".png"
    def __init__(self, data_dir, frames_per_sample=12, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, with_target=True, start_at=0, image_size=64):
        self.root = data_dir
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        
        tiles_dirs = glob.glob(path.join(self.root, "*"))
        tiles = {} 
        for tile_dir in tiles_dirs:
            if path.isdir(tile_dir):
                tile = path.basename(tile_dir)
                tiles[tile] = []
                for file in glob.glob(path.join(tile_dir, f"*{self.FILE_APPEND}")):
                    tiles[tile].append(file)
                tiles[tile].sort()
                
        self.sequences = []
        diference = 600
        for tile in tiles:
            for i, file in enumerate(tiles[tile]):
                sequence = []
                file_timestamp = int(path.basename(file).split(".")[0])
                for j, file2 in enumerate(tiles[tile][i+1:]):
                    file2_timestamp = int(path.basename(file2).split(".")[0])
                    if file2_timestamp == file_timestamp+(j+1)*diference:
                        sequence.append(file2)
                    else:
                        break
                    if len(sequence) == self.frames_per_sample:
                        break
                if len(sequence) == self.frames_per_sample:
                    self.sequences.append(tuple(sequence))

        self.train = train
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.start_at = start_at

        if self.train:
            portion = 0.9
        else:
            portion = -0.1

        if portion < 0:
            self.sequences = self.sequences[int(len(self.sequences)*(1+portion)):]
        else:
            self.sequences = self.sequences[:int(len(self.sequences)*portion)]
        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index, time_idx=0):
        sequence = self.sequences[index]
        query = []
        for i, file in enumerate(sequence):
            with Image.open(file) as img:
                img = img.convert("L")
                img = img.resize((self.image_size,self.image_size))
                img = transforms.ToTensor()(img)
                query.append(img)
        query = torch.stack(query)
        if self.with_target:
            #target = query[-1]
            #target = target.view(1, self.image_size, self.image_size)
            #target = target.repeat(self.frames_per_sample, 1, 1)
            return query, torch.tensor(1)
        return query

