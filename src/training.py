import torch
import yaml
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from .dataset import DiffusionModelDataModule
from .model import DiffusionModel
from .backbone import Backbone
from .noise_scheduler import NoiseScheduler

class DiffusionModelTrainer():
    def __init__(self, train_config_path='../config/training_config.yaml'):
        """
        Initialize a DiffusionModelTrainer object with a given configuration file.

        Args:
            train_config_path (str): Path to the YAML configuration file for training.

        Returns:
            None

        Raises:
            Exception: If there is an error loading the config file.
        """
        
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
        
        self.dm = DiffusionModelDataModule(
            dataset = data_config["dataset"],
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
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience = train_config['patience'],
            verbose = train_config['verbose']
        )

        logger = WandbLogger(
            name = logging_config['experiment_name'],
            version = logging_config['version'],
            dir = logging_config['logging_dir'],
            log_model=True,
        )

        backbone_params = {
            'in_channels': backbone_config['in_channels'], 
            'base_channels': backbone_config['base_channels'],
            'multipliers': backbone_config['multipliers'],
            'attention_res': backbone_config['attention_resolutions']
        }
        
        self.model = DiffusionModel(
            backbone_class=Backbone,
            backbone_params=backbone_params,
            noise_scheduler_class=NoiseScheduler,
            T=train_config['T'],
            lr=train_config['lr'],
            compute_fid_every_n_epochs = logging_config["compute_fid_every_n_epochs"]
        )

        self.trainer = Trainer(
            max_epochs=train_config['epochs'],
            accelerator="auto", 
            devices=train_config['devices'],
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            precision="16-mixed" if torch.cuda.is_available() else "32", 
            log_every_n_steps=logging_config['log_every_n_steps']
        )

    
    def train(self, checkpoint_path=None):
        """
        Train a diffusion model

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file of the DiffusionModel to resume training from. Defaults to None.

        Returns:
            None
        """
        self.trainer.fit(model=self.model, datamodule=self.dm, ckpt_path=checkpoint_path)

    def test(self, checkpoint_path=None):
        """
        Test a diffusion model

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file of the DiffusionModel to test from. Defaults to None.

        Returns:
            None
        """
        self.trainer.test(self.model, datamodule=self.dm, ckpt_path=checkpoint_path)
