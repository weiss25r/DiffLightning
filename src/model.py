import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
from lightning import LightningModule, Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

class DiffusionModel(LightningModule):
    def __init__(self, backbone_class, backbone_params, noise_scheduler_class, T=1000, lr=2e-4):
        super().__init__() 
        self.save_hyperparameters() 
        
        self.backbone = backbone_class(**backbone_params)
        self.noise_scheduler = noise_scheduler_class(T=T)
        
        self.metrics = nn.ModuleDict({
            "fid": FrechetInceptionDistance(feature=64, normalize=True) 
        }) 

    def forward(self, x, t):
        return self.backbone(x, t)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch 
        
        current_batch_size = images.shape[0]
        t = torch.randint(0, self.hparams.T, (current_batch_size,), device=self.device).long()

        with torch.no_grad():
            x_t, noise = self.noise_scheduler(images, t)

        noise_pred = self.backbone(x_t, t)
        
        loss = F.mse_loss(noise_pred, noise)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        current_batch_size = images.shape[0]
        t = torch.randint(0, self.hparams.T, (current_batch_size,), device=self.device).long()

        with torch.no_grad():
            x_t, noise = self.noise_scheduler(images, t)
            noise_pred = self.backbone(x_t, t)
            val_loss = F.mse_loss(noise_pred, noise)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        images_uint8 = ((images + 1) / 2).clamp(0, 1)
        self.metrics["fid"].update(images_uint8, real=True)

    def on_validation_epoch_end(self):
        
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalcolo FID all'epoca {self.current_epoch}...")
            
            fake_images = self.generate(n_samples=32, device=self.device)
            
            self.fid.update(fake_images, real=False)
            
            fid_score = self.metrics["fid"].compute()
            self.log('val_fid', fid_score, prog_bar=True)
            
            self.fid.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
    
    def test_step(self, batch, batch_idx):
        images, _ = batch
        current_batch_size = images.shape[0]
        t = torch.randint(0, self.hparams.T, (current_batch_size,), device=self.device).long()

        with torch.no_grad():
            x_t, noise = self.noise_scheduler(images, t)
            noise_pred = self.backbone(x_t, t)
            test_loss = F.mse_loss(noise_pred, noise)

        self.log('test_loss', test_loss, prog_bar=True)

        real_imgs_norm = (images + 1) / 2
        real_imgs_norm = real_imgs_norm.clamp(0, 1)
        
        self.metrics["fid"].update(real_imgs_norm, real=True)

    def on_test_epoch_end(self):
        print("\nCalcolo FID...")
        
        num_samples_total = 2 
        batch_size_gen = 1
        
        num_batches = num_samples_total // batch_size_gen
        
        for i in range(num_batches):
            fake_imgs = self.generate(n_samples=batch_size_gen, device=self.device)
            
            self.metrics["fid"].update(fake_imgs, real=False)
            
            if (i+1) % 5 == 0:
                print(f"Generato batch {i+1}/{num_batches}")

        fid_score = self.metrics["fid"].compute()
        print(f"Final Test FID: {fid_score}")
        
        self.log('test_fid', fid_score)
        
        self.metrics["fid"].reset()

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint, strict=True)
   
    @torch.no_grad()
    def generate(self, n_samples=16, device=None):
        if device is None:
            device = self.device
            
        self.backbone.eval()
        x = torch.randn((n_samples, 3, 32, 32)).to(device)

        T = self.hparams.T 

        for i in reversed(range(T)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            
            predicted_noise = self(x, t)

            beta_t = self.noise_scheduler.betas[i]
            alpha_t = self.noise_scheduler.alphas[i]
            alpha_hat_t = self.noise_scheduler.alphas_cumprod[i]
            
            noise_factor = beta_t / torch.sqrt(1 - alpha_hat_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - noise_factor * predicted_noise)

            if i > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * z
            else:
                x = mean

        self.backbone.train()
        
        x = (x.clamp(-1, 1) + 1) / 2
        return x

class DiffusionLoggingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            samples = pl_module.generate(n_samples=16, device=pl_module.device)
            
            grid = make_grid(samples, nrow=4)
            
            logger = trainer.logger
            
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_image'):
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                
                logger.experiment.log_image(logger.run_id, ndarr, f"generated_epoch_{trainer.current_epoch}.png")
            
            elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_image'):
                logger.experiment.add_image('generated_images', grid, trainer.current_epoch)