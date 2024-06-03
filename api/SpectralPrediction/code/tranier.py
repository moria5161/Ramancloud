'''
train_model
'''
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from loss_function import *
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
class Trainer:
    def __init__(self,model = None
                    ,train_loader=None,
                    val_loader=None,
                    test_loader=None,
                    loss_function=None,
                    device=torch.device('cuda:0'),
                    optimizer='Adam_amsgrad',
                    lr=5e-4,
                    rc=5.0,
                    weight_decay=0,
                    lr_decay_factor=0.5,
                    lr_decay_step_size= 25):
        self.opt_type=optimizer
        self.device= device
        self.model= model
        self.train_data=train_loader
        self.val_data=val_loader
        self.test_data=test_loader
        self.device=device
        self.opts={'AdamW':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'AdamW_amsgrad':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adam':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'Adam_amsgrad':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adadelta':torch.optim.Adadelta(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'RMSprop':torch.optim.RMSprop(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'SGD':torch.optim.SGD(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        }
        self.rc = rc
        self.optimizer=self.opts[self.opt_type]
        self.loss_function=loss_function
        self.scheduler = ReduceLROnPlateau(  optimizer=self.optimizer, 
                        mode='min', 
                        factor=lr_decay_factor,
                        patience=lr_decay_step_size, 
                        verbose=False,
                        threshold=0.0001,
                        threshold_mode= 'rel', 
                        cooldown=0,
                        min_lr=0, 
                        eps=1e-08)
    def bondsbatch2atoms_batch(self,atoms_count:torch.Tensor=None,atoms_x:torch.Tensor=None):
        atoms_batch = torch.zeros(size=[atoms_x.shape[0]],dtype=torch.int64)
        strat_x = 0
        for i in range(atoms_count.shape[0]):
            atoms_batch[strat_x : strat_x + atoms_count[i][0]] = i
            strat_x += atoms_count[i][0]
        return atoms_batch

    def train_per_epoch(self,len_batch,targ):
        loss_all = 0
        self.model.train()
        for j,batch in enumerate(self.train_data):
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            atoms_batch = self.bondsbatch2atoms_batch(batch.atoms_count,
                                                       batch.atoms_x)
            atoms_edge_index=radius_graph(x=batch.atoms_pos,r=self.rc,batch=atoms_batch)
            out = self.model(   atoms_x = batch.atoms_x.to(self.device), 
                                atoms_pos = batch.atoms_pos.to(self.device), 
                                atoms_edge_index = atoms_edge_index.to(self.device),   
                                atoms_batch = atoms_batch.to(self.device),
                                bonds_x = batch.x.to(self.device),    
                                bonds_edge_index=batch.edge_index.to(self.device), 
                                bonds_edge_attr=batch.edge_attr.to(self.device),    
                                bonds_batch=batch.batch.to(self.device))
                    
            target = batch[targ].to(self.device)
            # print(out.shape,target.shape)
            loss = self.loss_function(out,target)
            loss_all += loss.detach().cpu().numpy()
            loss.backward()
            self.optimizer.step()
        return loss_all / len_batch
    def val_per_epoch(self,len_batch,targ):
        loss_all = 0
        self.model.eval()
        for j,batch in enumerate(self.val_data):
            torch.cuda.empty_cache()
            atoms_batch = self.bondsbatch2atoms_batch(batch.atoms_count,
                                                      batch.atoms_x)
            atoms_edge_index=radius_graph(x=batch.atoms_pos,r=self.rc,batch=atoms_batch)
            out = self.model(   atoms_x = batch.atoms_x.to(self.device), 
                                atoms_pos = batch.atoms_pos.to(self.device), 
                                atoms_edge_index = atoms_edge_index.to(self.device),   
                                atoms_batch = atoms_batch.to(self.device),
                                bonds_x = batch.x.to(self.device),    
                                bonds_edge_index=batch.edge_index.to(self.device), 
                                bonds_edge_attr=batch.edge_attr.to(self.device),    
                                bonds_batch=batch.batch.to(self.device))
                    
            target = batch[targ].to(self.device)
            loss = l1loss(out,target)
            loss_all += loss.detach().cpu().numpy()
        return loss_all / len_batch
    def test_per_epoch(self,len_batch,targ):
        loss_all = 0
        self.model.eval()
        for j,batch in enumerate(self.test_data):
            torch.cuda.empty_cache()
            atoms_batch = self.bondsbatch2atoms_batch(batch.atoms_count,
                                                      batch.atoms_x)
            atoms_edge_index=radius_graph(x=batch.pos,r=self.rc,batch=atoms_batch)
            out = self.model(   atoms_x = batch.atoms_x.to(self.device), 
                                atoms_pos = batch.pos.to(self.device), 
                                atoms_edge_index = atoms_edge_index.to(self.device),   
                                atoms_batch = atoms_batch.to(self.device),
                                bonds_x = batch.x.to(self.device),    
                                bonds_edge_index = batch.edge_index.to(self.device), 
                                bonds_edge_attr = batch.edge_attr.to(self.device),    
                                bonds_batch = batch.batch.to(self.device))
                    
            target = batch[targ].to(self.device)
            loss = l1loss(out.reshape(target.shape),target)
            loss_all += loss.detach().cpu().numpy()
        return loss_all / len_batch
    def train(self,epoch,
                    targ,
                    save_path =None):
            print("train strat")
            len_train = len(self.train_data)
            len_val = len(self.val_data)
            len_test = len(self.test_data)
            
            for i in tqdm(range(epoch)):
                prev_val_loss = 1000000000
                train_loss = self.train_per_epoch(len_train,
                                            targ)
                
                val_loss = self.val_per_epoch(len_val,
                                        targ)
                test_loss = self.test_per_epoch(len_test,
                                                targ)
                self.scheduler.step(val_loss)
                if(val_loss < prev_val_loss):
                    torch.save(self.model.state_dict(),save_path)
                    prev_val_loss = val_loss
                # print('train_loss:{:.8f}'
                #               .format(train_loss))
                print('train_loss:{:.8f},val_loss:{:.8f},test_loss:{:.8f}'
                              .format(train_loss,val_loss,test_loss))
                    
if __name__ == "__main__":
    pass                
                
                
