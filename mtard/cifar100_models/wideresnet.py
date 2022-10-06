from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class _Swish(torch.autograd.Function):
  """Custom implementation of swish."""

  @staticmethod
  def forward(ctx, i):
    result = i * torch.sigmoid(i)
    ctx.save_for_backward(i)
    return result

  @staticmethod
  def backward(ctx, grad_output):
    i = ctx.saved_variables[0]
    sigmoid_i = torch.sigmoid(i)
    return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
  """Module using custom implementation."""

  def forward(self, input_tensor):
    return _Swish.apply(input_tensor)


class _Block(nn.Module):
  """WideResNet Block."""

  def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
    super().__init__()
    self.batchnorm_0 = nn.BatchNorm2d(in_planes)
    self.relu_0 = activation_fn()
    # We manually pad to obtain the same effect as `SAME` (necessary when
    # `stride` is different than 1).
    self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=0, bias=False)
    self.batchnorm_1 = nn.BatchNorm2d(out_planes)
    self.relu_1 = activation_fn()
    self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                            padding=1, bias=False)
    self.has_shortcut = in_planes != out_planes
    if self.has_shortcut:
      self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                stride=stride, padding=0, bias=False)
    else:
      self.shortcut = None
    self._stride = stride

  def forward(self, x):
    if self.has_shortcut:
      x = self.relu_0(self.batchnorm_0(x))
    else:
      out = self.relu_0(self.batchnorm_0(x))
    v = x if self.has_shortcut else out
    if self._stride == 1:
      v = F.pad(v, (1, 1, 1, 1))
    elif self._stride == 2:
      v = F.pad(v, (0, 1, 0, 1))
    else:
      raise ValueError('Unsupported `stride`.')
    out = self.conv_0(v)
    out = self.relu_1(self.batchnorm_1(out))
    out = self.conv_1(out)
    out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
    return out


class _BlockGroup(nn.Module):
  """WideResNet block group."""

  def __init__(self, num_blocks, in_planes, out_planes, stride,
               activation_fn=nn.ReLU):
    super().__init__()
    block = []
    for i in range(num_blocks):
      block.append(
          _Block(i == 0 and in_planes or out_planes,
                 out_planes,
                 i == 0 and stride or 1,
                 activation_fn=activation_fn))
    self.block = nn.Sequential(*block)

  def forward(self, x):
    return self.block(x)


class WideResNet(nn.Module):
  """WideResNet."""

  def __init__(self,
               num_classes: int = 10,
               depth: int = 28,
               width: int = 10,
               activation_fn: nn.Module = nn.ReLU,
               mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
               std: Union[Tuple[float, ...], float] = CIFAR10_STD,
               padding: int = 0,
               num_input_channels: int = 3):
    super().__init__()
    self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
    self.std = torch.tensor(std).view(num_input_channels, 1, 1)
    self.mean_cuda = None
    self.std_cuda = None
    self.padding = padding
    num_channels = [16, 16 * width, 32 * width, 64 * width]
    assert (depth - 4) % 6 == 0
    num_blocks = (depth - 4) // 6
    self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
    self.layer = nn.Sequential(
        _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                    activation_fn=activation_fn))
    self.batchnorm = nn.BatchNorm2d(num_channels[3])
    self.relu = activation_fn()
    self.logits = nn.Linear(num_channels[3], num_classes)
    self.num_channels = num_channels[3]

  def forward(self, x):
    if self.padding > 0:
      x = F.pad(x, (self.padding,) * 4)
    if x.is_cuda:
      if self.mean_cuda is None:
        self.mean_cuda = self.mean.cuda()
        self.std_cuda = self.std.cuda()
      if 'float16' in str(x.dtype):
        self.mean_cuda = self.mean_cuda.half()
        self.std_cuda = self.std_cuda.half()
      out = (x - self.mean_cuda) / self.std_cuda
    else:
      out = (x - self.mean) / self.std
    out = self.init_conv(out)
    out = self.layer(out)
    out = self.relu(self.batchnorm(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, self.num_channels)
    return self.logits(out)

def WideResNet_70_16():
    model = WideResNet(
        num_classes=100, depth=70, width=16,
        activation_fn=Swish, mean=CIFAR10_MEAN,
        std=CIFAR10_STD)
    return model

def WideResNet_22_6():
    model = WideResNet(
        num_classes=100, depth=22, width=6,
        activation_fn=Swish, mean=CIFAR10_MEAN,
        std=CIFAR10_STD)
    return model

def WideResNet_34_20():
    model = WideResNet(
        num_classes=100, depth=34, width=20,
        activation_fn=Swish, mean=CIFAR10_MEAN,
        std=CIFAR10_STD)
    return model

def WideResNet_70_16_cifar10():
    model = WideResNet(
        num_classes=10, depth=70, width=16,
        activation_fn=Swish, mean=CIFAR10_MEAN,
        std=CIFAR10_STD)
    return model
