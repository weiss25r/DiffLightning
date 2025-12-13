import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
from lightning import LightningModule, Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

class DiffusionModel(LightningModule):
    def __init__(self, backbone_class, backbone_params, noise_scheduler_class, T=1000, lr=2e-4, compute_fid_every_n_epochs=5):
        super().__init__() 
        self.save_hyperparameters() 
        
        self.backbone = backbone_class(**backbone_params)
        self.noise_scheduler = noise_scheduler_class(T=T)
        
        self.metrics = nn.ModuleDict({
            "val_fid": FrechetInceptionDistance(feature=64, normalize=True),
            "test_fid": FrechetInceptionDistance(feature=64, normalize=True)
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
        if images_uint8.shape[1] == 1:
            images_uint8 = images_uint8.repeat(1, 3, 1, 1)

        self.metrics["val_fid"].update(images_uint8, real=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        if (self.current_epoch + 1) % self.hparams.compute_fid_every_n_epochs == 0:
            print(f"\nFID computation for epoch {self.current_epoch}...")
            
            fake_images = self.generate(n_samples=32, device=self.device)
            
            wandb_logger = self.logger
            grid = make_grid(tensor=fake_images, normalize=True)
            wandb_logger.log_image(key=f"samples_epoch_{self.current_epoch}", images=[grid], caption=[f"samples_epoch_{self.current_epoch}"])

            if fake_images.shape[1] == 1:
                fake_images = fake_images.repeat(1, 3, 1, 1)

            self.metrics["val_fid"].update(fake_images, real=False)
            fid_score = self.metrics["val_fid"].compute()
            self.log('val_fid', fid_score, prog_bar=True)
            
            self.metrics["val_fid"].reset()

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

        if real_imgs_norm.shape[1] == 1:
            real_imgs_norm = real_imgs_norm.repeat(1, 3, 1, 1)
        
        self.metrics["test_fid"].update(real_imgs_norm, real=True)

    def on_test_epoch_end(self):
        print("\nFID computation...")
        
        num_samples_total = 128 
        batch_size_gen = 32
        
        num_batches = num_samples_total // batch_size_gen
        
        for i in range(num_batches):
            fake_imgs = self.generate(n_samples=batch_size_gen, device=self.device)

            wandb_logger = self.logger

            grid = make_grid(tensor=fake_imgs, normalize=True)
            wandb_logger.log_image(key="samples_test", images=[grid], caption=["samples_test"])
            
            if fake_imgs.shape[1] == 1:
                fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

            self.metrics["test_fid"].update(fake_imgs, real=False)

        fid_score = self.metrics["test_fid"].compute()
        print(f"Final Test FID: {fid_score}")
        
        self.log('test_fid', fid_score)
        
        self.metrics["test_fid"].reset()

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.load_state_dict(checkpoint, strict=True)
   
    @torch.no_grad()
    def generate(self, n_samples=16, device=None):
        if device is None:
            device = self.device
        self.to(device)
        self.backbone.eval()
        x = torch.randn((n_samples, self.backbone.in_channels, 32, 32)).to(device)

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