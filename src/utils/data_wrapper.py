from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class Wrap_Dataset(Dataset):
    """Wrapper, convert <doc_feature, click_tag, ips_weight> Tensor into Pytorch Dataset"""
    def __init__(self, X, y, use_cuda=True):
        if use_cuda:
            self.X = torch.LongTensor(X).cuda()
            self.y = torch.Tensor(y).cuda()
        else:
            self.X = torch.LongTensor(X).cpu()
            self.y = torch.Tensor(y).cpu()


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y) 