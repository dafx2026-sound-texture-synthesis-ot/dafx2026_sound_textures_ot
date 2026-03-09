import numpy as np
import torch
import torch.nn as nn
import torchaudio
import itertools
import torch.nn.functional as func
import torchaudio.functional as F
import torchaudio.transforms as T

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RISpecGaussianOT(nn.Module):
    def __init__(self,
                 ref_waveform,
                 device = None,
                 out_chan = [128] * 8,
                 dilations = [(1,1)] * 8,
                 strides = [(1,1)] * 8,
                 filter_sizes = [(101,2), (53,3), 
                                 (11,5), (3,3), 
                                 (7, 7), (11,11), 
                                 (19, 19), (27,27)],
                 n_fft = 512):
        '''
        ref_waveform: reference waveform
        out_chan: number of output channels from each shallow CNN
        device: cuda or not
        filter_sizes: list of filter sizes
        '''
        super(RISpecGaussianOT, self).__init__()
        self.ref_waveform = ref_waveform
        #convolutions for shallow CNN
        self.convs = nn.ModuleList([
                                      nn.Conv2d(in_channels=2, 
                                                out_channels=out_chan[i], 
                                                kernel_size=filter_sizes[i],
                                                stride=strides[i], 
                                                dilation=dilations[i], 
                                                groups=1, 
                                                padding='valid', 
                                                bias=False).to(device)
                                      for i in range(len(filter_sizes))
                                  ])
        for conv in self.convs:
            conv.weight.data.uniform_(-0.05, 0.05)
        self.relu = nn.ReLU()
        self.device = device
        self.transform = torchaudio.transforms.Spectrogram(n_fft = n_fft,
                                                           power = None).to(device)

    #add scalogram option here?
    def get_RI_spec(self, waveform):
        '''
        gets RI spectogram of waveform
        '''
        waveform_stft = self.transform(waveform)
        waveform_stft_real = torch.view_as_real(waveform_stft)
        waveform_stft_real = waveform_stft_real/torch.max(torch.abs(waveform_stft_real) + 1e-8)
        RI_spec_unpermuted = 2 * func.sigmoid(30 * waveform_stft_real) - 1
        RI_spec = torch.permute(RI_spec_unpermuted, (0,3,1,2))
        return RI_spec

    def apply_random_CNN(self, RI_spec, detach = False):
        '''
        applies random CNNs to RI spec for feature extraction
        '''
        output = []
        for conv in self.convs:
             if detach:
                 out = self.relu(conv(RI_spec)).detach()
             else:
                 out = self.relu(conv(RI_spec))
            #  #squeeze ensures everything is 3d
             output.append(out.squeeze())

        return output

    def get_stats_for_wasserstein(self, waveform, detach = False):
        """
        Wrapper to get components for OT calc for all random CNNs.
        """
        stats = []
        RI_spec = self.get_RI_spec(waveform)
        CNN_outputs = self.apply_random_CNN(RI_spec)
        for out in CNN_outputs:
            stats.append(self.get_layer_desc(out, detach))
        return stats

    def calc_moments(self, tensor):
        """calculates mean for each channel and calculates cov matrix too"""
        tensor = tensor.squeeze()                          # [C, T] or [B, C, T]
        mu = torch.mean(tensor, dim=[1, 2], keepdim=True)  # [B, C, 1, 1] → mean per channel
        cov = torch.matmul(tensor - mu,
                          torch.transpose(tensor - mu, 2, 1)) / mu.shape[-1]
        return mu.squeeze(), cov

    def get_cov_sqrt_and_diag(self, cov, calc_root_cov = False):
        '''
        Calculates eigenfunction decomposition of a covariance matrix
        and returns the root of the covariance matrix and trace of the matrix.
        '''
        root_cov = None
        tr_cov = None
        if calc_root_cov:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigval_max = torch.clamp(eigvals,  min=1e-12)
            tr_cov = torch.sum(eigval_max, dim = 1)
            root_cov = eigvecs * torch.sqrt(eigval_max).unsqueeze(1)
            root_cov = root_cov @ eigvecs.transpose(1, 2)
        else:
            #this is slow, switch to ein sum
            #tr_cov = torch.sum(torch.clamp(torch.linalg.eigvalsh(cov), min=1e-12), dim = 1)
            tr_cov =  torch.einsum("bii->b", cov)
        return root_cov, tr_cov

    def get_layer_desc(self, tensor, calc_root_cov = False, detach = False):
        """ Gets required components for OT calc for each layer. """
        mu, cov = self.calc_moments(tensor)
        root_cov, tr_cov = self.get_cov_sqrt_and_diag(cov, calc_root_cov)
        if detach:
          return [mu.detach(), cov.detach(), root_cov.detach(), tr_cov.detach()]
        else:
            return [mu, cov, root_cov, tr_cov]

    def gaussian_wasserstein_l2_dist(self, ref_stat, syn_stat):
        """
        Wasserstein Distance between two gaussian distributions. In this case,
        we use the mean/cov of the reference image that's created durng the
        the init call.
        """
        #mean component
        diff_squared = (ref_stat[0]- syn_stat[0])**2
        cov_prod = torch.matmul(torch.matmul(ref_stat[2], syn_stat[1]),ref_stat[2])
        eigvals_prod = torch.linalg.eigvalsh(cov_prod)
        #OT loss can be slightly negative here...
        var_overlap = torch.sum(torch.sqrt(torch.clamp(eigvals_prod, min=1e-12)), dim = 1)
        wasserstein_loss =  diff_squared + ref_stat[3] + syn_stat[3]-2*var_overlap
        return torch.mean(wasserstein_loss)

    def forward(self, waveform):
        """
        forward call
        """
        losses  = []
        syn_stats = []
        RI_spec = self.get_RI_spec(waveform)
        CNN_outputs = self.apply_random_CNN(RI_spec)
        for out in CNN_outputs:
            syn_stats.append(self.get_layer_desc(out))
        for i in range(len(syn_stats)):
            loss = self.gaussian_wasserstein_l2_dist(self.ref_stats[i], syn_stats[i])
            losses.append(loss)

        return torch.sum(torch.stack(losses))
