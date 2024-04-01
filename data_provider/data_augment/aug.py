import torch 
import numpy as np

def downsample(self, data, interval):
    # interval can be options like 3D, 1W, 2W
    # Assuming data is a dataframe with index = time, columns = stocks
    resampled = data.resample(interval).first()

    # Get the sequence length & dimensions 
    sequence_length = self.sequence_length
    dimensions = resampled.shape[1]

    # Convert to numpy array for faster processing
    data_array = resampled.values

    # Calculate the sequence length
    windows = data.shape[0] - sequence_length + 1

    # Initialize an empty array to store the mapped sequences
    mapped_array = np.empty((windows, sequence_length, dimensions))

    # Map the sequences using a sliding window approach
    for i in range(windows):
        mapped_array[i] = data_array[i:i+sequence_length]

    return mapped_array


class DatasetAugmentation():
    
    def __init__(self, sequence_length, aug_choice = 'downsample', frAug_choice = 'freq_mix', dropout_rate = 0.2):
        self.sequence_length = sequence_length
        self.aug_choice = aug_choice
        self.frAug_choice = frAug_choice
        self.dropout_rate = dropout_rate
    
    

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

    def dataAug(self, data):
        if self.aug_choice == 'downsample':
            outputs = downsample()
            pass

        if self.aug_choice == 'frAug':
            pass