import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow.act_norm import ActNorm
from models.glow.coupling2 import Coupling
from models.glow.inv_conv import InvConv


class Glow(nn.Module):

    def __init__(self, cond1, cond2, num_channels, num_levels, num_steps, mode='sketch'):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))
        if mode in ['fix_inpainting', 'random_inpainting']:
            self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                               cond_channels1=4*cond1,
                               cond_channels2 = 4*cond2,
                               mid_channels=num_channels,
                               num_levels=num_levels,
                               num_steps=num_steps)
        else:
            self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                               cond_channels1=4*cond1,
                               cond_channels2 = 4*cond2,
                               mid_channels=num_channels,
                               num_levels=num_levels,
                               num_steps=num_steps)
        self.mode = mode

    def forward(self, x, x_cond1, x_cond2,  reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x = squeeze(x)
        x_cond1 = squeeze(x_cond1)
        x_cond2 = squeeze(x_cond2)
        x, sldj = self.flows(x, x_cond1, x_cond2, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        # y = x
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, cond_channels1, cond_channels2, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels1=cond_channels1,
                                              cond_channels2=cond_channels2,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              cond_channels1=4 * cond_channels1,
                              cond_channels2=4 * cond_channels2,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, x_cond1, x_cond2, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x_cond1, x_cond2, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x_cond1 = squeeze(x_cond1)
            x_cond2 = squeeze(x_cond2)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x_cond1, x_cond2, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x_cond1 = squeeze(x_cond1, reverse=True)
            x_cond2 = squeeze(x_cond2, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x_cond1, x_cond2, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels1, cond_channels2, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, cond_channels1, cond_channels2, mid_channels)

    def forward(self, x, x_cond1, x_cond2, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x_cond1, x_cond2, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, x_cond1, x_cond2, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x


if __name__ == "__main__":
    model = Glow(cond1=3, cond2=10, num_channels=128, num_levels=6, num_steps=6)
    img = torch.randn(4,3,32,32)
    img = img.sigmoid()
    cond1 = torch.randn(4,3,32,32)
    cond2 = torch.randn(4,10)
    cond2 = cond2.view(cond2.size(0), cond2.size(1),1,1)
    cond2 = cond2.repeat(1,1, img.size(2), img.size(3))
    z, ldj = model(img, cond1, cond2)
    rc, _ = model(z, cond1, cond2, reverse=True)
    print(img-rc.sigmoid())
