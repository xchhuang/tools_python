import torch
import numpy as np
import platform


class Loader(Dataset):
    def __init__(self):
        self.splitter = '/'
        if platform.system() == 'Windows':
            self.splitter = '\\'

    # def cmp(self, x):
    #     x = x.split(self.splitter)[-1].split('.')[0]
    #     x = int(x)
    #     # print (x)
    #     return x

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.all_data)
