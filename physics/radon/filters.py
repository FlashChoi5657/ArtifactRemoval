import torch
from torch import nn
import torch.nn.functional as F

from .utils import PI, fftfreq

'''source: https://github.com/matteo-ronchetti/torch-radon'''

# class AbstractFilter(nn.Module):
#     def __init__(self):
#         super(AbstractFilter, self).__init__()

#     def forward(self, x):
#         input_size = x.shape[2]
#         projection_size_padded = \
#             max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
#         pad_width = projection_size_padded - input_size
#         padded_tensor = F.pad(x, (0,0,0,pad_width))
#         f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
#         fourier_filter = self.create_filter(f)
#         fourier_filter = fourier_filter.unsqueeze(-2)
#         projection = torch.view_as_real(torch.fft.fft(padded_tensor.transpose(2,3), dim=1)).transpose(2,3) * fourier_filter
#         return torch.fft.irfft(torch.view_as_complex(projection.transpose(2,3)), n=projection.transpose(2,3).shape[1], dim=1).transpose(2,3)[:,:,:input_size,:]

#     def _get_fourier_filter(self, size):
#         n = torch.cat([
#             torch.arange(1, size / 2 + 1, 2),
#             torch.arange(size / 2 - 1, 0, -2)
#         ])

#         f = torch.zeros(size)
#         f[0] = 0.25
#         f[1::2] = -1 / (PI * n) ** 2

#         fourier_filter = torch.view_as_real(torch.fft.fft(f, dim=1))
#         fourier_filter[:, 1] = fourier_filter[:, 0]

#         return 2*fourier_filter

#     def create_filter(self, f):
#         raise NotImplementedError
    
class AbstractFilter(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0, 0, 0, pad_width))
        f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(f)
        fourier_filter = fourier_filter.unsqueeze(-2)

        projection = torch.view_as_real(torch.fft.fft(padded_tensor.transpose(2, 3))).transpose(2, 3) * fourier_filter
        result = torch.view_as_real(torch.fft.ifft(torch.view_as_complex(projection).transpose(2, 3)))[..., 0]
        result = result.transpose(2, 3)[:, :, :input_size, :]

        return result

    def _get_fourier_filter(self, size):
        n = torch.cat([torch.arange(1, size / 2 + 1, 2), torch.arange(size / 2 - 1, 0, -2)])

        f = torch.zeros(size, dtype=self.dtype, device=self.device)
        f[0] = 0.25
        f[1::2] = -1 / (PI * n) ** 2

        fourier_filter = torch.view_as_real(torch.fft.fft(f, dim=-1))
        fourier_filter[:, 1] = fourier_filter[:, 0]

        return 2 * fourier_filter

    def create_filter(self, f):
        raise NotImplementedError
    
class RampFilter(AbstractFilter):
    def __init__(self):
        super(RampFilter, self).__init__()

    def create_filter(self, f):
        return f

class HannFilter(AbstractFilter):
    def __init__(self):
        super(HannFilter, self).__init__()

    def create_filter(self, f):
        n = torch.arange(0, f.shape[0])
        hann = 0.5 - 0.5*(2.0*PI*n/(f.shape[0]-1)).cos()
        return f*hann.roll(hann.shape[0]//2,0).unsqueeze(-1)

class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(2*fftfreq(filter_size).abs().view(-1, 1))

    def forward(self, x):
        fourier_filter = self.filter.unsqueeze(-1).repeat(1,1,2).to(x.device)
        projection = torch.view_as_real(torch.fft.fft(x.transpose(2,3), dim=1)).transpose(2,3) * fourier_filter
        return torch.fft.irfft(torch.view_as_complex(projection.transpose(2,3)), n=projection.transpose(2,3).shape[1], dim=1).transpose(2,3)

        # projection = torch.fft.rfft(x.transpose(2, 3), 1).transpose(2, 3) * fourier_filter
        # return torch.fft.irfft(projection.transpose(2, 3), 1).transpose(2, 3)