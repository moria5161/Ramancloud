'''
load dataset:

'''
from torch_geometric.loader import DataLoader
import torch
import os
def loadDataset(file_path:str   =   None,
                batch_size:int  =   None,
                shuffle :bool = True):
    print("load"+file_path.split('/')[-1].split('.')[0])
    dataset = torch.load(file_path)
    datasetloader = DataLoader(dataset,batch_size,shuffle)
    return datasetloader
if __name__ == "__main__":
    pass