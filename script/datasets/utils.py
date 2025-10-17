from torch.utils.data import Dataset

class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        return [0]
