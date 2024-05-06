# import os
# os.chdir("pretrained-vision-transformer")

import sys
import yaml
import lightning as L

sys.path.append('./src/')
from country211_module import Country211DataModule
from eurosat_module import EuroSAT_RGB_DataModule, SentinelTest
from vision_transformer import VisionTransformerPretrained
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

with open('./configs/default.yaml') as cf_file:
        config = yaml.safe_load(cf_file.read())



# setup data
# datamodule = EuroSAT_RGB_DataModule(path_to_data, config['batch_size'], config['valid_size'])
# datamodule.prepare_data()
# datamodule.setup()
# train_dataloader = datamodule.train_dataloader()
# valid_dataloader = datamodule.valid_dataloader()
# test_dataloader = datamodule.test_dataloader()



path_to_data = "../drive/MyDrive/AIML24/testset/testset/testset"
test_dataset = SentinelTest(path_to_data, config['batch_size'])
test_dataloader = test_dataset.test_dataloader()

# TODO: try https://pytorch-lightning.readthedocs.io/en/1.6.5/common/checkpointing.html#restoring-training-state
model = VisionTransformerPretrained(
    config['model'], 10, config['learning_rate']
)

early_stopping = EarlyStopping(monitor='valid_acc', patience=config['early_stopping_patience'], mode='max')

logger = TensorBoardLogger("tensorboard_logs", name=config['run_id'])

trainer = L.Trainer(max_epochs=config['max_epochs'],
                        devices=config['devices'], 
                        num_nodes=config['num_nodes'],
                        strategy='ddp',
                        callbacks=[early_stopping], 
                        logger=logger,
                        enable_progress_bar=False)

chkpt_path = "tensorboard_logs/alpha/version_1/checkpoints/epoch=14-step=2955.ckpt"

# trainer.fit(model, ckpt_path=chkpt_path)
trainer.predict(model=model, dataloaders=test_dataloader, ckpt_path=chkpt_path)
# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)