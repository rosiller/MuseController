import torch
import numpy as np
from os import path
from pylsl import StreamInlet, resolve_byprop
from muselsl import constants as mlsl_cnsts



class TransformSubset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole dataset.
        indices (sequence): Indices in the whole set selected for subset.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __getitem__(self, idx):
        # if self.dataset.transform != self.transform:
        #     self.dataset.transform = self.transform
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class GazeObject(torch.utils.data.Dataset):

    def __init__(self, datapath, num_classes=6, transform=None):
        self.datapath = datapath
        self.num_classes = num_classes
        X = np.load(path.join(datapath, f'trials.npy'))
        y = np.load(path.join(datapath, f'labels_{num_classes}class.npy'))
        self.samples = torch.Tensor(X).to(dtype=torch.float)
        self.labels = torch.Tensor(y).to(dtype=torch.long)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
        label = self.labels[idx]

        return sample.unsqueeze(0), label
        
    def ignore_trials(self,target=2):
        trial_list = torch.where(self.labels!=target)[0]
        self.samples = self.samples[trial_list]
        self.labels = self.labels[trial_list]

def get_stream_and_inlet(sensor='EEG'):
    streams = resolve_byprop('type', sensor, timeout=mlsl_cnsts.LSL_SCAN_TIMEOUT)

    if sensor=='EEG':
        max_chunklen = mlsl_cnsts.LSL_EEG_CHUNK
    elif sensor =='ACC':
        max_chunklen = mlsl_cnsts.LSL_ACC_CHUNK
    elif sensor =='GYRO':
        max_chunklen = mlsl_cnsts.LSL_GYRO_CHUNK

    inlet = StreamInlet(streams[0], max_chunklen=max_chunklen)

    return inlet