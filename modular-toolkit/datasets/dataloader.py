import torch
from torch.utils.data import Dataset, DataLoader


#The generated or loaded data is passed into the Dataset for the dataloader to output data in batches
class MathOPDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.x = torch.tensor([x[:4]for x in data])
        self.y = torch.tensor([[x[-1]]for x in data])
        self.tgt = torch.tensor([x[4:]for x in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.tgt[idx]



class MathOPImageDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor([[a, b] for a, b, _ in data], dtype=torch.long)
        self.y = torch.tensor([c for _, _, c in data], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]




def data_loader(X, batch_size):
    data_loader = DataLoader(X, batch_size, shuffle=True)
    return data_loader


