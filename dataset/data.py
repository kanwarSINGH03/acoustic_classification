from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


class MineData(Dataset):
    """This class is used to store the data for the mines."""
    def __init__(self, mat_file, transform=None):
        self.data = loadmat(mat_file)
        
        # Load and shape data
        self.samples = self.data["x"].T 
        self.labels = torch.tensor(self.data["y"].flatten())  # shape: (3309,)
        
        self.transform = transform
        self.number_datapoints = self.samples.shape[0]

    def __len__(self):
        return self.number_datapoints

    def __getitem__(self, idx):
        waveform = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)

        if self.transform:
            sample = self.transform(waveform) 
            sample = sample.squeeze(0).flatten()
        else:
            sample = waveform.flatten()

        return sample, self.labels[idx]


# data = MineData("./data/mine_impact_data_2019.mat")


