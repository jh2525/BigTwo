from torch.utils.data import Dataset, DataLoader

class PPODataset(Dataset):
    def __init__(self, tensors, device):
        self.tensors = tensors
        for i in self.tensors:
            self.tensors[i] = self.tensors[i].to(device)
        
        
    def __getitem__(self, idx):
        return tuple([self.tensors[i][idx]  for i in self.tensors])
    
    def __len__(self):
        return len(self.tensors['advantages'])
    
