import torch
import numpy as np

class DatasetAugmentation():
    def __init__(self):
        pass
    
    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)

        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate

        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)

        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        
        return x, y

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)
        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=0, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        

        x2, y2 = torch.from_numpy(x2), torch.from_numpy(y2)
        xy2 = torch.cat([x2,y2],dim=0)
        xy2_f = torch.fft.rfft(xy2,dim=0)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=0)
        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y
    