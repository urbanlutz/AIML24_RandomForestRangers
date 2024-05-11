#!/usr/bin/python
import sys
import torch
from torch.nn import functional as F
from torch import optim
from torch import nn

from transformers import ViTForImageClassification
import torchmetrics

sys.path.append('./src/')
from country211_module import Country211DataModule
from eurosat_module import EuroSAT_RGB_DataModule, SentinelTest


import lightning as L

class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential([
        nn.Conv2d(12, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
 
        nn.Conv2d(32, 3, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        ])

        self.decoder = nn.Sequential([
        nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
 
        nn.Conv2d(32, 12, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EncodeMultiChannel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 3, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        
 
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        return self.act2(self.conv2(x)) # torch.Size([1, 32, 224, 224])

class ViTForImageClassificationMultiChannelAuto(L.LightningModule):
    def __init__(self, auto_encoder: AutoEncoder, num_classes, *args, **kwargs):
        super().__init__()
        self.encode = auto_encoder.encoder
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
  
    def forward(self, x):
        x = self.encode.forward(x)
        return self.vit(x)

class ViTForImageClassificationMultiChannel(L.LightningModule):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()
        self.encode = EncodeMultiChannel()
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
  
    def forward(self, x):
        x = self.encode.forward(x)
        return self.vit(x)
    


class VisionTransformerPretrained(L.LightningModule):
    '''
    Wrapper for the torchvision pretrained Vision Transformers

    Args:
      model (str)       : specifies which flavor of ViT to use
      num_classes (int) : number of output classes
      learning_rate (float) : optimizer learning rate

    '''

    def __init__(self, model, num_classes, learning_rate):

        super().__init__()

        if model == 'vit_b_16':
            vit = ViTForImageClassificationMultiChannel(num_classes)
        else:
            raise ValueError(model)

        self.backbone = vit

        # metrics
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)

        # other
        self.learning_rate = learning_rate

    def forward(self, x):
        try:
            return self.backbone(x)
        except:
            return self.backbone(x[0]), x[1]

    def step(self, batch):
       '''
       Any step processes batch to return loss and predictions
       '''

       x, y = batch
       prediction = self.backbone(x)
       y_hat = torch.argmax(prediction.logits, dim=-1)

       loss = F.cross_entropy(prediction.logits, y)
       acc = self.acc(y_hat, y)
       
       return loss, acc, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        # Return your dataloader here
        path_to_data = "../drive/MyDrive/AIML24/testset/testset/testset"
        test_dataset = SentinelTest(path_to_data, 2)
        test_dataloader = test_dataset.test_dataloader()
        return test_dataloader
