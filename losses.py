import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        kernel = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel = (kernel.T @ kernel).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.eps = eps

    def _laplacian(self, image):
        filtered = F.conv2d(F.pad(image, (2, 2, 2, 2), mode="replicate"), self.kernel, groups=3)
        down = filtered[:, :, ::2, ::2]
        up = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        filtered = F.conv2d(F.pad(up, (2, 2, 2, 2), mode="replicate"), self.kernel, groups=3)
        return image - filtered

    def forward(self, prediction, target):
        difference = self._laplacian(prediction) - self._laplacian(target)
        return torch.sqrt(difference.square() + self.eps**2).mean()


def restoration_losses(model, prediction, target, weights, edge_loss):
    values = {
        "l1": F.l1_loss(prediction, target),
        "ssim": 1.0 - ssim(prediction, target, data_range=1.0, size_average=True),
        "edge": edge_loss(prediction, target),
    }
    mvgl = prediction.new_zeros(())
    for module in model.modules():
        if hasattr(module, "compute_dino_kl_loss"):
            mvgl = mvgl + module.compute_dino_kl_loss()
    values["mvgl"] = mvgl
    total = sum(float(weights[name]) * value for name, value in values.items())
    return total, values

