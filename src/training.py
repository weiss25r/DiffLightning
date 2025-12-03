import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataset import Cifar10DataModule
from src.model import DiffusionModel, DiffusionLoggingCallback
from src.backbone import Backbone
from noise_scheduler import NoiseScheduler

class DiffusionModelTrainer():
    def __init__(self, batch_size = 128, lr=2e-4, epochs=100, T=1000):
        self.dm = Cifar10DataModule(
            batch_size=batch_size, 
            val_split=0.1  
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='ddpm-cifar10-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        
        image_callback = DiffusionLoggingCallback()

        mlf_logger = MLFlowLogger(
            experiment_name="DDPM_CIFAR10",
            tracking_uri="file:./mlruns" 
        )

        backbone_params = {
            'in_channels': 3, 
            'base_channels': 64,
            'multipliers': (1, 2, 2, 2)
        }
        
        self.model = DiffusionModel(
            backbone_class=Backbone,
            backbone_params=backbone_params,
            noise_scheduler_class=NoiseScheduler,
            T=T,
            lr=lr
        )

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto", 
            devices=1,
            logger=mlf_logger,
            callbacks=[checkpoint_callback, image_callback],
            precision="16-mixed" if torch.cuda.is_available() else "32", 
            log_every_n_steps=10
        )
    
    def train(self):
        self.trainer.train(model=self.model, datamodule=self.dm)

    def test(self, checkpoint_path=None):
        trainer.test(self.model, datamodule=self.dm, ckpt_path="checkpoints/ddpm-cifar10-epoch=00-val_loss=0.05.ckpt", weights_only=False)

if __name__ == "__main__":
    trainer = DiffusionModelTrainer()
