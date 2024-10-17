"""Define modules used in models"""
from typing import Tuple, Union
import torch
from torch import Tensor


class Conv(torch.nn.Module):
    """Dropout, convolution, batchnorm, activation"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 padding: int = 1,
                 stride: int = 1,
                 dropout: float = 0.,
                 nonlinearity: bool = True) -> None:
        super().__init__()
        self.nonlinearity = nonlinearity
        self.dropout = torch.nn.Dropout2d(dropout)
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity="relu",
                                      mode="fan_out")
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        if self.nonlinearity:
            self.relu = torch.nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        out = self.dropout(inp)
        out = self.conv(out)
        out = self.batchnorm(out)
        if self.nonlinearity:
            out = self.relu(out)
        return out


class Collapse(torch.nn.Module):
    """Fully connected on frequencies and average on time"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout2d(dropout)
        self.conv = Conv(in_channels,
                         out_channels, (kernel_size[0], 1),
                         stride=1,
                         padding=0)
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size[1])

    def forward(self, inp: Tensor) -> Tensor:
        out = self.dropout(inp)
        out = self.conv(out)
        out = out.squeeze(2)
        out = self.avg(out)
        return out


class FC(torch.nn.Module):
    """Fully connected layer"""
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.dense = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     1,
                                     stride=1,
                                     padding=0)
        torch.nn.init.kaiming_normal_(self.dense.weight,
                                      nonlinearity="relu",
                                      mode="fan_out")
        self.relu = torch.nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        out = self.dropout(inp)
        out = self.dense(out)
        out = self.relu(out)
        return out


class Output(torch.nn.Module):
    """Final binary classification layer"""
    def __init__(self,
                 in_channels: int,
                 n_class: int,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.dense = torch.nn.Conv1d(in_channels, n_class, kernel_size=1)
        torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, inp: Tensor) -> Tensor:
        out = self.dropout(inp)
        out = self.dense(out)
        return out


class Filter(torch.nn.Module):
    """Frequency dependent convolutions"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 freq_dim: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (2, 2),
                 dropout: float = 0.,
                 nonlinearity: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.kernel_size = kernel_size
        self.unfolded_num = (freq_dim + 2 * padding[0] - kernel_size[0] +
                             stride[0]) // stride[0]
        self.dropout = torch.nn.Dropout2d(dropout)
        self.unfold = torch.nn.Unfold((kernel_size[0], 1),
                                      padding=(padding[0], 0),
                                      stride=(stride[0], 1))
        self.conv = torch.nn.Conv2d(in_channels * self.unfolded_num,
                                    out_channels * self.unfolded_num,
                                    kernel_size=kernel_size,
                                    stride=(1, stride[1]),
                                    padding=(0, padding[1]),
                                    groups=self.unfolded_num)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity="relu",
                                      mode="fan_out")
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        bsize = inp.shape[0]
        tsize = inp.shape[3]
        out = self.unfold(inp)
        out = out.reshape(bsize, self.in_channels, self.kernel_size[0],
                          self.unfolded_num, tsize)
        out = out.permute(0, 3, 1, 2, 4)
        out = out.reshape(bsize, self.in_channels * self.unfolded_num,
                          self.kernel_size[0], tsize)
        out = self.dropout(out)
        out = self.conv(out)
        out = out.reshape(bsize, self.unfolded_num, self.out_channels, -1)
        out = out.permute(0, 2, 1, 3)
        out = self.batchnorm(out)
        if self.nonlinearity:
            out = self.relu(out)
        return out
