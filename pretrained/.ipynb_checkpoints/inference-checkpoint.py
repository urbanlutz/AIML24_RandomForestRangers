# import os
# os.chdir("pretrained-vision-transformer")

import sys
import yaml
import lightning as L

sys.path.append('./pretrained/src/')
from eurosat_module import EuroSAT_RGB_DataModule, SentinelTest
from vision_transformer import VisionTransformerPretrained
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import sys
import torch
import yaml
import lightning as L
import numpy as np
sys.path.append('./pretrained/src/')
from eurosat_module import EuroSAT_RGB_DataModule, SentinelTest
from vision_transformer import VisionTransformerPretrained
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
with open('./pretrained/configs/default.yaml', encoding="utf-8") as cf_file:
    config = yaml.safe_load(cf_file.read())




datamodule = EuroSAT_RGB_DataModule(config["data_root"], config['batch_size'], config['valid_size'])
datamodule.prepare_data()
datamodule.setup()



test_dataset = SentinelTest(config["test_data_path"], 2)
test_dataloader = test_dataset.test_dataloader()

# TODO: try https://pytorch-lightning.readthedocs.io/en/1.6.5/common/checkpointing.html#restoring-training-state
model = VisionTransformerPretrained(
    config['model'], 10, config['learning_rate']
)

early_stopping = EarlyStopping(monitor='valid_acc', patience=config['early_stopping_patience'], mode='max')

logger = TensorBoardLogger("pretrained/tensorboard_logs", name=config['run_id'])

trainer = L.Trainer( max_epochs=config['max_epochs'],
                        devices=1,
                        num_nodes=config['num_nodes'],
                        strategy='ddp',
                        callbacks=[early_stopping],
                        logger=logger,
                        enable_progress_bar=True)

chkpt_path = "pretrained/tensorboard_logs/alpha/version_36/checkpoints/epoch=7-step=1576.ckpt"

# trainer.fit(model, ckpt_path=chkpt_path)

preds = trainer.predict(model=model, dataloaders=test_dataloader, ckpt_path=chkpt_path)

print(f"len preds {len(preds)}")
# import os
# os.chdir("pretrained-vision-transformer")



# with open('./pretrained/configs/default.yaml', encoding="utf-8") as cf_file:
#     config = yaml.safe_load(cf_file.read())
    
    
# def get_id(img_path):
#     return img_path.split("/")[-1].split("_")[-1].split(".")[0]


# datamodule = EuroSAT_RGB_DataModule(config["data_root"], config['batch_size'], config['valid_size'])
# datamodule.prepare_data()
# datamodule.setup()

labels2ids = datamodule.train_data.dataset.class_to_idx
ids2labels = {v: k for k, v in labels2ids.items()}


def oh2text(one_hot):
    idx = torch.argmax(one_hot).item()
    return ids2labels[idx]




import torch

class_array = []
for pred, idx in preds:
    class_array = [*class_array, *[( idx[i], oh2text(pred.loss['logits'][i]))for i in range(pred.loss['logits'].shape[0])]]

from datetime import datetime
execution_start = datetime.now().strftime("%m%d%Y-%H%M%S")
import csv
with open(f'{execution_start}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['test_id', 'label'])

    # Write each string to a row with its corresponding index as the test_id
    for i, label in class_array:
        writer.writerow([i, label])

# import pickle
# with open('model_output.pkl', 'wb') as f:
#     pickle.dump(preds, f)
# disable randomness, dropout, etc...
# model.eval()

# # predict with the model
# y_hat = model(x)