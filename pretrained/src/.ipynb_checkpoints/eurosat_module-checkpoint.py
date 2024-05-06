#!/usr/bin/env python

import torch
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import lightning as L
from torch.utils.data import DataLoader, Subset
import rasterio as rio
from rasterio.plot import reshape_as_image

def load_img(img_path:str) -> np.ndarray:
    if img_path.split('.')[-1] == "tif":
        with rio.open(img_path, "r") as d:
            img = d.read([1,2,3,4,5,6,7,8,9,10,11,12,13])
            img = reshape_as_image(img)
    else:
        img = np.load(img_path)
    return img.astype("float32")


# TODO: improvement -> find global max / min
def l2a_approx(img):
    l2a_bands = img[:,:,[0,1,2,3,4,5,6,7,12,8,10,11]]
    return l2a_bands
    # band_min = np.min(l2a_bands, (0,1)) # minimal value per band
    # return l2a_bands - band_min # dark object subtraction algo approximation

def bandselect(img):
    return img[:, :, [3,2,1]]


class EuroSAT_RGB_DataModule(L.LightningDataModule):
    '''
    Lightning datamodule for the Country211 dataset

    '''

    def __init__(self, data_root, batch_size, valid_size=2700):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size

        self.num_workers = 8
        self.valid_size = valid_size

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        '''
        Download the data manually

        https://zenodo.org/records/7711810#.ZAm3k-zMKEA
        '''
        print(self.data_root)
        assert os.path.exists(self.data_root), print('Download URL: https://zenodo.org/records/7711810#.ZAm3k-zMKEA')

    def setup(self):
        '''
        Setup the dataset

        '''

        # define the transforms
        # - resize to (224, 224) as expected for ViT
        # - scale to [0,1] and transform to float32
        # - normalize with ViT mean/std

        transforms = v2.Compose([l2a_approx,
                                 bandselect,
                                 v2.ToImage(),
                                 v2.Resize(size=(224,224), interpolation=2, antialias=True),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])

        data = ImageFolder(self.data_root, transform=transforms, loader=load_img)
        targets = np.asarray(data.targets)

        print(data)
        print('Total number of samples: ', len(targets))

        tmp_ix, test_ix = train_test_split(np.arange(len(targets)), test_size=5400, stratify=targets)
        train_ix, valid_ix = train_test_split(tmp_ix, test_size=self.valid_size, stratify=targets[tmp_ix])
                                
        self.train_data = Subset(data, train_ix)
        self.valid_data = Subset(data, valid_ix)
        self.test_data = Subset(data, test_ix)

        print(f'Training samples: {len(self.train_data)}')
        print(f'Validation samples: {len(self.valid_data)}')
        print(f'Test samples: {len(self.test_data)}')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
def get_id(img_path):
    return img_path.split("/")[-1].split("_")[-1].split(".")[0]

class SentinelTest():

    def __init__(self, data_root, batch_size, transformations=None):
        self.img_paths = [path.replace("\\","/") for path in glob.glob(os.path.join(data_root,  f"*.npy"))]
        self.transformations = transformations
        self.current_index = 0
        self.batch_size = batch_size
        self.num_workers = 8
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = load_img(img_path).astype(np.float32)
        image_id = get_id(img_path)

        if self.transformations:
            image = self.transformations(image)
        return image, image_id

    def __next__(self):
        image, image_id = self.__getitem__(self.current_index)
        self.current_index += 1
        return image

    def __iter__(self):
        return self

    def test_dataloader(self):
        return DataLoader(dataset=self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)