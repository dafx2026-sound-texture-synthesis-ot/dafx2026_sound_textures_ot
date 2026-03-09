import numpy as np
import torch
import torch.nn as nn
import torchaudio
import itertools
import torch.nn.functional as func
import torchaudio.functional as F
import torchaudio.transforms as T
from Synthesis import *
from LBFGS_wrapper import *

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if name == '__main__': 
  #arg parse here
  in_path ='/content/drive/MyDrive/fire.wav'
  out_path = '/content/drive/MyDrive/syn_fire.wav'
  ITERS = 20
  LBFGS_HISTORY_SIZE = 25
  VERBOSE = True
  
  waveform, sample_rate = torchaudio.load(IN_PATH)
  waveform = waveform.to(device)
  waveform_max = torch.max(waveform)
  waveform_min = torch.min(waveform)

  noise = torch.randn_like(waveform)
  win_len = 512
  noise[:,:win_len // 2] = noise[:,:win_len // 2] * torch.hann_window(win_len, device = device)[:win_len // 2]
  noise[:,-win_len // 2:] = noise[:,-win_len // 2:] * torch.hann_window(win_len, device = device)[-win_len // 2:]

  syn_audio = torch.autograd.Variable(noise.clone().detach(),
                                      requires_grad=True).to(device)
  #class for making texture
  textureNet = RISpecGaussianOT(waveform,
                                device = device)
  textureNet.ref_stats = []
  textureNet.RI_spec = textureNet.get_RI_spec(textureNet.ref_waveform)
  textureNet.ref_activations = textureNet.apply_random_CNN(textureNet.RI_spec, detach = True)
  for out in textureNet.ref_activations:
        textureNet.ref_stats.append(textureNet.get_layer_desc(out, detach = True, calc_root_cov  = True))
		
  #optimizer
  optimizer = LBFGSWithCounter([syn_audio],
                              history_size = LBFGS_HISTORY_SIZE,
                              line_search_fn='strong_wolfe'
	)
  #training loop
  for k in range(ITERS):
      def closure(count = True):
          optimizer.zero_grad()
          #loss = textureNet(syn_audio + 1e-3 * torch.randn_like(syn_audio))
          loss = textureNet(syn_audio)
          if count:
              optimizer.num_iterations += 1
              loss.backward()
          return loss
      optimizer.step(closure)
      if ((k+1)%2 == 0) & VERBOSE:
          print('Epoch {} Complete!'.format(k+1))
          print("Total function iterations:", optimizer.num_iterations)
          print('Loss is:', closure(count = False).detach().item())

	#scaling at the end
	syn_audio = syn_audio * torch.std(waveform)/torch.std(syn_audio)
	torchaudio.save(OUT_PATH, syn_audio.detach().cpu(), sample_rate)