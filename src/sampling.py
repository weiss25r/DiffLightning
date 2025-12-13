import torch
from torchvision.utils import save_image
from .model import DiffusionModel

class DiffusionModelSampler():
    def __init__(self, checkpoint_path):
        """
        Initialize a DiffusionModelSampler object to sample from a trained diffusion model.

        Args:
            checkpoint_path (str): Path to the checkpoint file of the DiffusionModel.

        Returns:
            None
        """
        self.model = DiffusionModel.load_from_checkpoint(checkpoint_path, weights_only=False)

    def sample(self, n_samples=16, device=None, save_path=None):
        """
        Sample from a trained diffusion model.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 16.
            device (str, optional): Device to use for sampling. If none, it is auto-selected from the available devices.
            save_path (str, optional): Path to save the generated samples. If none, samples are not saved.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, C, H, W)
        """
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(device)
        sample = self.model.generate(n_samples, device)
        if save_path is not None:
            save_image(sample, save_path)

        return sample
