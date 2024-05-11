#!/usr/bin/env python

import sys
import yaml
import torch
import csv
import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append('./pretrained/src/')
from eurosat_module import EuroSAT_RGB_DataModule, SentinelTest
from vision_transformer import VisionTransformerPretrained

import utils


def main(arg):
    L.seed_everything(1312)
    print('Seeed everything')

    with open('./pretrained/configs/default.yaml') as cf_file:
        default_config = yaml.safe_load(cf_file.read())

    if len(arg) == 1:
        with open(arg[0]) as cf_file:
            config = yaml.safe_load( cf_file.read() )

        config = utils.merge_dictionaries_recursively(default_config, config)
        print(f'Configuration read from {arg[0]}')
    else:
        config = default_config
        print('Configuration read from ./configs/default.yaml')

    if config['data_set'] == 'tif':
        # setup data
        path_to_data = config['data_root']
        datamodule = EuroSAT_RGB_DataModule(path_to_data, config['batch_size'], config['valid_size'])
        datamodule.prepare_data()

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    print('Finished data loading')
    print('Number of classes:', datamodule.num_classes)

    # setup model
    model = VisionTransformerPretrained(config['model'], datamodule.num_classes, config['learning_rate'])
    print('Finished model loading')

    # setup callbacks
    early_stopping = EarlyStopping(monitor='valid_acc', patience=config['early_stopping_patience'], mode='max')

    # logger
    logger = TensorBoardLogger("pretrained/tensorboard_logs", name=config['run_id'])

    # train
    trainer = L.Trainer(max_epochs=config['max_epochs'],
                        devices=config['devices'], 
                        num_nodes=config['num_nodes'],
                        strategy='ddp',
                        callbacks=[early_stopping], 
                        logger=logger,
                        enable_progress_bar=True)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # test
    print('Starting model test')
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)
    print('Finished model test')
    from datetime import datetime
    execution_start = datetime.now().strftime("%m%d%Y-%H%M%S")
    trainer.save_checkpoint(f"final_checkpoint/version_{logger.version}-{execution_start}.chkpt")


    # Generate Kaggle Submission
    test_dataset = SentinelTest(config["test_data_path"], 2)
    test_dataloader = test_dataset.test_dataloader()

    # predict testset samples
    preds = trainer.predict(model=model, dataloaders=test_dataloader)

    
    # resolve predictions
    labels2ids = datamodule.train_data.dataset.class_to_idx
    ids2labels = {v: k for k, v in labels2ids.items()}
    
    def oh2text(one_hot):
        idx = torch.argmax(one_hot).item()
        return ids2labels[idx]
        
    class_array = []
    for pred, idx in preds:
        class_array = [*class_array, *[( idx[i], oh2text(pred.loss['logits'][i]))for i in range(pred.loss['logits'].shape[0])]]
    
    # Save as CSV
    with open(f'results/version_{logger.version}-{execution_start}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['test_id', 'label'])
    
        # Write each string to a row with its corresponding index as the test_id
        for i, label in class_array:
            writer.writerow([i, label])

if __name__=='__main__':
    main(sys.argv[1:])
