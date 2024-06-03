import torch
import numpy as np

def R2addl1loss(out,target):
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    diff = out-target
    return torch.mean(diff ** 2) * (SSE / SST)
def mR2(out,target):
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return SSE / SST
def attionl1loss(out,target):
    loss_sum = torch.sum(torch.abs(out-target))
    
    return 10
    
def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)

def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))

def rmse(out, target):
    '''Root Mean Square Eroor(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2))

def state_l2loss(out,target):
    '''    #It is used to get the loss of the excited state vector,
    but we have tested that it does not perform well in QM9spectra's transition dipole prediction.


    from J. Phys. Chem. Lett. 2020, 11, 3828−3834
    <Combining SchNet and SHARC: The SchNarc Machine Learning Approach for Excited-State Dynamics>'''
    diffa=torch.abs(out-target)**2
    diffb=torch.abs(out+target)**2
    diff=torch.min(diffa,diffb)
    return torch.mean(diff)

def R2(out,target):
    '''coefficient of determination,Square of Pearson's coefficient, used to assess regression accuracy'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)

def R2addl2loss(out,target):
    
    return l2loss(out,target)/R2(out,target)

def combine_lose(out_tuple,target_tuple,lamb=10):
    '''Combine loss of energy and force,Training for ab initio dynamics simulation'''
    return l2loss(out_tuple[0],target_tuple[0])+lamb*l2loss(out_tuple[1],target_tuple[1])
if __name__ == "__main__":
    pass