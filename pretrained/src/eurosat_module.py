#!/usr/bin/env python

import torch
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision import transforms
from PIL import Image

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


BANDS = [3,2,1]

def bandselect(img):
    return img[:, :, [3,2,1]]


train_mean = [1353.7283, 1117.2009, 1041.8888,  946.5547, 1199.1866, 2003.0106,
        2374.0134, 2301.2244, 2599.7827,  732.1823, 1820.6930, 1118.2052]
train_std = [ 65.2964, 153.7740, 187.6989, 278.1234, 227.9242, 355.9332, 455.1324,
        530.7811, 502.1637,  98.9300, 378.1612, 303.1070]
test_mean = [380.1732,  400.1498,  628.8646,  578.8707,  943.4276, 1826.2419,
        2116.6646, 2205.9729, 2281.1836, 2266.9331, 1487.6902,  959.2352]
test_std = [115.1743, 209.1482, 241.2069, 301.1053, 269.5137, 420.2494, 503.8187,
        598.0409, 529.4133, 403.9382, 398.1438, 342.4408]
# train_mean = [946.5547, 1041.8888, 1117.2009]
# train_std = [278.1234,187.6989, 153.7740]
# test_mean = [ 578.8707,628.8646, 400.1498]
# test_std = [301.1053,241.2069, 209.1482]


def wiggle_them_bands(x):
    num_bands = x.shape[0]

    for i in range(num_bands):
        x[i] = transforms.ColorJitter(brightness=(0.2,1.2),contrast=(1),saturation=(0.2,1.2),hue=(-0.1,0.1))(x[i].unsqueeze(0))
    return x

def color_jitter(x):
    # jitter rgb bands only (order doesn't matter)
    pre = x[0, :, :].unsqueeze(0)
    jittered = transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))(x[[1,2,3], : , :])
    post = x[4:, : ,:]
    return torch.cat((pre, jittered, post), 0)

train_transforms  = transforms.Compose([
    l2a_approx,
    # bandselect,
    v2.ToImage(),
    v2.Resize(size=(224,224), interpolation=2, antialias=True),
    v2.RandomResizedCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(test_mean, test_std),
    # wiggle_them_bands,
    color_jitter,
    transforms.GaussianBlur(1),
])
test_transforms  = transforms.Compose([
    # bandselect,
    v2.ToImage(),
    v2.Resize(size=(224,224), interpolation=2, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(test_mean, test_std),
    transforms.GaussianBlur(1),
])



class EuroSAT_RGB_DataModule(L.LightningDataModule):
    '''
    Lightning datamodule for the EuroSAT dataset

    '''

    def __init__(self, data_root, batch_size, valid_size=2700):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size

        self.num_workers = 2
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

        transforms = train_transforms

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
        self.transformations = test_transforms
        self.current_index = 0
    
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
        print(len(self.img_paths))
        return DataLoader(dataset=self, batch_size=2, num_workers=1, shuffle=False)