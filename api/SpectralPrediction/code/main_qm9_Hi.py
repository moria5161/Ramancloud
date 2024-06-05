import torch
import torch.nn as nn
import os
import datetime
import numpy as np
from net import Net
from tranier import Trainer
from loaddataset import loadDataset
from loss_function import *

print("pid",os.getpid())
mode_save_name = (str(datetime.datetime.now().year) + "_" 
                + str(datetime.datetime.now().month) + "_" 
                + str(datetime.datetime.now().day) + "_"
                + str(datetime.datetime.now().hour) + "_"
                + str(datetime.datetime.now().minute)+ "_" 
                + str(datetime.datetime.now().second))
print("time:\t",mode_save_name)
print(__file__)
## param
num_features = 128
act='swish'
maxl=3
num_block=3
num_radial=32
attention_head=8
rc=5.0
max_atomic_number=35
scalar_outsize=1
irreps_out= None
summation= False
out_type='scalar'
grad_type='Hi'
device=torch.device("cuda:1")
# ----------------------------------------------
targ = 'Hi'
seed = 25
loss_function = l1loss
train_data_path ="/data/garelee/qm9/my/train_dataset.pt"
val_data_path ="/data/garelee/qm9/my/val_datasets.pt"
test_data_path ="/data/garelee/qm9/my/test_datasets.pt"
optimizer = 'AdamW'
epoch = 240
lr = 1e-3
batch_size= 48
weight_decay = 0
lr_decay_factor=0.5
lr_decay_step_size= 10
model_save_path = "/data/garelee/code/git-v2/model"
model_load_path = ""
mode_parameter = ("num_features\t=\t"   +   str(num_features)  +   "\n" 
                 + "act\t=\t"   +   str(act)    +   "\n" 
                 + "maxl\t=\t"   +   str(maxl)    +   "\n" 
                 + "num_block\t=\t" + str(num_block) + "\n"
                 + "num_radial\t=\t"  + str(num_radial) + "\n"
                 + "attention_head\t=\t" + str(attention_head) + "\n"
                 + "rc\t=\t" + str(rc) + "\n"  
                 + "max_atomic_number\t=\t" + str(max_atomic_number) + "\n"
                 + "scalar_outsize\t=\t" +str(scalar_outsize) + "\n"
                 + "irreps_out\t=\t" + str(irreps_out) + "\n"
                 + "summation\t=\t" + str(summation) + "\n"
                 + "out_type\t=\t" + str(out_type) + "\n"
                 + "grad_type\t=\t" + str(grad_type) + "\n"
                 + "targ\t=\t" + targ + "\n" 
                 + "device\t=\t" + str(device) + "\n"
                 + "random_seed\t=\t" + str(seed) + "\n"
                 + "loss_function\t=\t" + str(loss_function) + "\n"
                 + "train_data_path\t=\t"+os.path.basename(train_data_path).split('.')[0]+ "\n"
                 + "optimizer\t=\t" + str(optimizer) + "\n"
                 + "epoch\t=\t" +   str(epoch)  +   "\n"
                 + "lr\t=\t"    +   str(lr) +   "\n"
                 + "batch_size\t=\t"    +   str(batch_size) +   "\n" 
                 + "weight_decay\t=\t" + str(weight_decay) + "\n"
                 + "lr_decay_factor\t=\t" + str(lr_decay_factor) + "\n"
                 + "lr_decay_step_size\t=\t" + str(lr_decay_step_size) + "\n"
                 + "model_save_path\t=\t" + os.path.join(model_save_path,mode_save_name+".pth") +"\n"
                 + "model_load_path\t=\t" + model_load_path + "\n"
                 )
print(mode_parameter)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# model = torch.load(model_load_path)
model = Net(     num_features=num_features,
                 act=act,
                 maxl=maxl,
                 num_block=num_block,
                 num_radial=num_radial,
                 attention_head=attention_head,
                 rc=rc,
                 max_atomic_number=max_atomic_number,
                 scalar_outsize=scalar_outsize,
                 irreps_out=irreps_out,
                 summation=summation,
                 out_type=out_type,
                 grad_type=grad_type,
                 device=device)
model.train()
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')
train_loader = loadDataset(file_path = train_data_path,
                            batch_size=batch_size,
                            shuffle=True)
val_loader = loadDataset(file_path = val_data_path,
                           batch_size = batch_size,
                           shuffle = False)
test_loader = loadDataset(file_path = test_data_path,
                           batch_size = batch_size,
                           shuffle = False)
trainer = Trainer(model  = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  test_loader=test_loader,
                  loss_function = loss_function,
                  device = device,
                  optimizer = optimizer,
                  lr = lr,
                  rc = rc,
                  weight_decay = weight_decay ,
                  lr_decay_factor=lr_decay_factor,
                  lr_decay_step_size= lr_decay_step_size
                  )
trainer.train(epoch = epoch,
               targ = targ,
               save_path = os.path.join(model_save_path,mode_save_name+".pth")
               )