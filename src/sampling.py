import torch
from torchvision.utils import save_image
from .model import DiffusionModel

class DiffusionModelSampler():
    def __init__(self, checkpoint_path):
        self.model = DiffusionModel.load_from_checkpoint(checkpoint_path)

    def sample(self, n_samples=16, device=None, save_path=None):
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        
        self.model.to(device)
        sample = self.model.generate(n_samples, device)
        if save_path is not None:
            save_image(sample, save_path)

