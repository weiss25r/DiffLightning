import torch
import yaml
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from .dataset import Cifar10DataModule
from .model import DiffusionModel, DiffusionLoggingCallback
from .backbone import Backbone
from .noise_scheduler import NoiseScheduler
import wandb

class DiffusionModelTrainer():
    def __init__(self, train_config_path='../config/training_config.yaml'):
        try:
            with open(train_config_path, 'r') as f:
                config_file = yaml.safe_load(f)
                train_config = config_file['training']
                data_config = config_file['data']
                augmentation_config = config_file['augmentation']
                logging_config = config_file['logging']
                backbone_config = config_file['backbone']

        except Exception as e:
            print("Error loading config file:", e)
            return
        
        self.dm = Cifar10DataModule(
            batch_size = data_config['batch_size'], 
            val_split = data_config['val_split'],
            num_workers = data_config['num_workers'],
            aug_settings = augmentation_config,
            seed=data_config['seed']
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=logging_config['checkpoint_dir'],
            filename=logging_config['checkpoint_name'],
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        
        image_callback = DiffusionLoggingCallback()

        if logging_config['logger'] == 'mlflow':
            logger = MLFlowLogger(
                experiment_name=logging_config['experiment_name'],
                tracking_uri="file:./mlruns" 
            )
        elif logging_config['logger'] == 'wandb':
            logger = WandbLogger(
                name = logging_config['experiment_name'],
                version = logging_config['version'],
                dir = logging_config['logging_dir'],
                log_model=True,
            )

        backbone_params = {
            'in_channels': backbone_config['in_channels'], 
            'base_channels': backbone_config['base_channels'],
            'multipliers': backbone_config['multipliers']
        }
        
        self.model = DiffusionModel(
            backbone_class=Backbone,
            backbone_params=backbone_params,
            noise_scheduler_class=NoiseScheduler,
            T=train_config['T'],
            lr=train_config['lr']
        )

        self.trainer = Trainer(
            max_epochs=train_config['epochs'],
            accelerator="auto", 
            devices=train_config['devices'],
            logger=logger,
            callbacks=[checkpoint_callback, image_callback],
            precision="16-mixed" if torch.cuda.is_available() else "32", 
            log_every_n_steps=10
        )

    
    def train(self):
        self.trainer.fit(model=self.model, datamodule=self.dm)

    def test(self, checkpoint_path=None):
        self.trainer.test(self.model, datamodule=self.dm, ckpt_path=checkpoint_path, weights_only=False)
