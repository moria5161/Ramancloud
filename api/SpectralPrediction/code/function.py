import io
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .mol_from_np_2 import *
from scipy.sparse import dok_matrix
import scipy
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Chem as Chem
import shutil
import ase
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
# import gensim
from torch_geometric.nn import radius_graph
# from gensim.models import KeyedVectors
from tqdm import tqdm
from openbabel import openbabel as ob
import openbabel


def car2xyz(car_filepath:str,xyz_filepath:str):
    with open(car_filepath,'r') as f:
        car_lines = f.readlines()
        f.close()
    xyz_lines = car_lines[0].split()[0].split('.')[0] +"\n" + "\t" + "\n"
    for  i in range(1,len(car_lines)):
        car_line = car_lines[i].split()
        xyz_lines += car_line[0].split('=')[-1] + "\t" + car_line[2] \
            + "\t" + car_line[3]+ "\t" + car_line[4] + "\n"
    with open(xyz_filepath,'w') as f:
        f.write(xyz_lines)
        f.close()

def log2arc(log_filepath,arc_filepath):
    with open(log_filepath,'r') as f:
        log_lines =  f.readlines()
        f.close()
    for i in range(len(log_lines)):
        if i < len(log_lines)-1 and log_lines[i] == "\n" and log_lines[i+1] == " ----------------------------------------------------------------------\n" and log_lines[i+2] != "\n":
            log_lines = log_lines[i+2:]
            break
    idx = 0
    while(True):
        line = log_lines[idx].split()[0].split('\\')
        if(len(line) > 2 and line[0] == "1" and line[1] == "1"):
            break
        idx += 1   
    log_lines = log_lines[idx:] 
    # 获取@尾部
    idx = 0 
    for i in range(0,len(log_lines)):
        if (len(log_lines[len(log_lines) - 1 -i].split())>0 and len(log_lines[len(log_lines) - 1 -i].split()[-1].split('\\'))>0 and 
        log_lines[len(log_lines) - 1 -i].split()[-1].split('\\')[-1] == '@'):
            idx = i
            break
    log_lines = log_lines[:len(log_lines) - idx]
    arc_lines = ""
    for i in range(0,len(log_lines)):
        log_lines[i] = log_lines[i].replace(' ','')
        if(len(log_lines[i].split())>1):
            print('error',log_filepath.split('\\')[-1])
        else:         
            arc_lines += log_lines[i].split()[0] 
    arc_lines = arc_lines.split('Freq\\\\Title\\\\0,1\\')[-1]
    arc_lines = arc_lines.split('\\\\\\')[0]
    with open(arc_filepath,'w') as f:
        f.write(arc_lines)
        f.close()
        
def arc2xyz(arc_path:str,xyz_path:str):
    with open(arc_path,'r') as f:
        line = f.readline()
        f.close()
    s,_ = line.split('\\\\Version=')
    atoms = s.split('\\')
    lines = str(len(atoms)) + "\n" + "\t" + "\n"
    for i in range(len(atoms)):
        atom = atoms[i].split(',')
        lines += atom[0] + "\t" + atom[1] + "\t" +atom[2] +"\t" +atom[3] +"\n"
    with open(xyz_path,'w') as f:
        f.writelines(lines)    
        
def np2xyz(atoms_x:np.array,atoms_pos:np.array,file_path:str=None):
    lines = str(atoms_pos.shape[0]) + "\n" + "\t" + "\n"
    for i in range(atoms_pos.shape[0]):
        lines += Chem.GetPeriodicTable().GetElementSymbol(int(atoms_x[i])) + "\t" + str(float(atoms_pos[i][0])) + "\t" +str(float(atoms_pos[i][1])) +"\t" +str(float(atoms_pos[i][2])) +"\n" 
    if file_path is None:
        f = io.StringIO(lines)
        return f
    else:
        with open(file_path,'w') as f:
            f.writelines(lines)

def obmol2rdmol(obmol:openbabel.openbabel.OBMol,rdmol:Chem.rdchem.Mol):
    bond_type = [   Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC
                ]
    mol = Chem.RWMol(rdmol)
    AC = np.zeros(shape=[obmol.NumAtoms(),obmol.NumAtoms()],dtype=np.int64)
    for i in range(obmol.NumBonds()):
        bond = obmol.GetBondById(i)
        BeginAtomIdx = bond.GetBeginAtomIdx() - 1
        EndAtomIdx = bond.GetEndAtomIdx() - 1 
        if AC[BeginAtomIdx][EndAtomIdx] == 1:
            continue
        AC[BeginAtomIdx][EndAtomIdx]  = 1
        AC[EndAtomIdx][BeginAtomIdx]  = 1
        j = 3
        if bond.IsAromatic() == False:
            j = bond.GetBondOrder() - 1   
        mol.AddBond(BeginAtomIdx,EndAtomIdx,bond_type[j])
    return mol.GetMol(),AC

# 根据mol_path生成rdmol和邻接矩阵  
def xyz2rdmol_AC(mol_path:str):
    rdmol = Chem.MolFromXYZFile(mol_path)
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("xyz")
    obmol = ob.OBMol()
    gard_2 = obconversion.ReadFile(obmol, mol_path)
    if  gard_2 == False:
            print(mol_path.split('/')[-1].split('.')[-1],"\tno")
            return KeyError
    AC = np.zeros(shape=[rdmol.GetNumAtoms(),rdmol.GetNumAtoms()],dtype=np.int64)
    rdmol ,AC = obmol2rdmol(obmol,rdmol)
    return rdmol,AC

def pdb2rdmol_AC(pdb_file:str):
    obconversion = ob.OBConversion()
    obconversion.SetInFormat("pdb")
    obmol = ob.OBMol()
    gard_2 = obconversion.ReadFile(obmol, pdb_file)
    if  gard_2 == False:
            print(pdb_file.split('/')[-1].split('.')[-1],"\tno")
            return KeyError
    lines  =  str(obmol.NumAtoms()) + "\n" + "\t" + "\n"
    for i in range(obmol.NumAtoms()):
        atom = obmol.GetAtomById(i)
        lines += Chem.GetPeriodicTable().GetElementSymbol(int(atom.GetAtomicNum())) + "\t" + str(float(atom.GetX())) + "\t" +str(float(atom.GetY())) +"\t" +str(float(atom.GetZ())) +"\n" 
    with open("./"+os.path.basename(pdb_file).split('.')[0]+".xyz",'w') as f:
        f.writelines(lines)
    rdmol = Chem.MolFromXYZFile("./"+os.path.basename(pdb_file).split('.')[0]+".xyz")
    os.remove("./"+os.path.basename(pdb_file).split('.')[0]+".xyz")
    AC = np.zeros(shape=[rdmol.GetNumAtoms(),rdmol.GetNumAtoms()],dtype=np.int64)
    rdmol ,AC = obmol2rdmol(obmol,rdmol)
    return rdmol,AC

# 根据rdmol生成mol的atoms_x和atoms_pos
def rdmol2x_pos(rdmol):
    atoms_x = np.zeros(shape = [rdmol.GetNumAtoms()],dtype=np.int64)
    atoms_pos = np.zeros(shape = [rdmol.GetNumAtoms(),3],dtype=np.float32)
    confer = rdmol.GetConformer()
    for i in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(i)
        atoms_x[i] = atom.GetAtomicNum()
        atoms_pos[i] = np.array([confer.GetAtomPosition(i).x,confer.GetAtomPosition(i).y,confer.GetAtomPosition(i).z],dtype=np.float32)
    return atoms_x,atoms_pos

def generate_atoms_edge_index(atoms_adjacent_matrix: np.array = None,
                              mol:Chem.rdchem.Mol = None):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 按照化学键遍历
    # if np.where(atoms_adjacent_matrix==0)[0].shape[0] != atoms_adjacent_matrix.shape[0]:
    #     return KeyError
    edge_index = np.zeros(shape=[1,4],dtype=np.int32)
    for i in range(mol.GetNumBonds()):
        # BeginAtomIdx
        atom_mid_1 = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        # EndAtomIdx
        atom_mid_2 = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atom_head = np.where(atoms_adjacent_matrix[atom_mid_1]==1)[0]
        atom_tail = np.where(atoms_adjacent_matrix[atom_mid_2]==1)[0]
        for j in range(atom_head.shape[0]):
            if atom_head[j] == atom_mid_2:
                continue
            for k in range(atom_tail.shape[0]):
                if atom_tail[k] == atom_mid_1:
                    continue  
                edge_index = np.r_[edge_index,np.array([[atom_head[j],atom_mid_1,atom_mid_2,atom_tail[k]]],dtype=np.int16)]
                edge_index = np.r_[edge_index,np.array([[atom_tail[k],atom_mid_2,atom_mid_1,atom_head[j]]],dtype=np.int16)]
    
    return edge_index[1:]


def atoms2bonds_edge_index(atoms_edge_index: np.array = None,
                            mol: Chem.rdchem.Mol = None):
    bond_type = [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC'
    ]
    bonds_edge_index = np.zeros(shape=[atoms_edge_index.shape[0],3],dtype=np.int32)
    bonds_x = np.zeros(shape=[mol.GetNumBonds(),4],dtype=np.float32)
    atom_distance = Chem.Get3DDistanceMatrix(mol)
    for i in range(mol.GetNumBonds()):
        bonds_x[i,0] = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        bonds_x[i,1] = mol.GetBondWithIdx(i).GetEndAtomIdx()
        bonds_x[i,2] = bond_type.index(mol.GetBondWithIdx(i).GetBondType().name)
        bonds_x[i,0] = atom_distance[mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                                    mol.GetBondWithIdx(i).GetEndAtomIdx()]
    for i in range(atoms_edge_index.shape[0]):
        for j in range(atoms_edge_index.shape[1]-1):
            bonds_edge_index[i,j] = mol.GetBondBetweenAtoms(int(atoms_edge_index[i,j]),
                                                            int(atoms_edge_index[i,j+1])).GetIdx()
    return bonds_edge_index,bonds_x

def generate_angel_edge_bondes(atoms_pos: np.array = None,
                             atoms_edge_index: np.array = None):
    angle_edge_bonds = np.zeros(shape=[atoms_edge_index.shape[0],4],dtype=np.float32)
    for i in range(atoms_edge_index.shape[0]):
        bond_1 = atoms_pos[atoms_edge_index[i,1]] - atoms_pos[atoms_edge_index[i,0]]
        bond_2 = atoms_pos[atoms_edge_index[i,2]] - atoms_pos[atoms_edge_index[i,1]]
        bond_3 = atoms_pos[atoms_edge_index[i,3]] - atoms_pos[atoms_edge_index[i,2]]
        # 1-2的夹角余弦
        angle_edge_bonds[i,0] = np.arccos(np.dot(bond_1,bond_2) / np.linalg.norm(bond_1) /np.linalg.norm(bond_2))/pi
        # 2-3的夹角余弦
        angle_edge_bonds[i,1] = np.arccos(np.dot(bond_2,bond_3) / np.linalg.norm(bond_2) /np.linalg.norm(bond_3))/pi
        # 1-3的夹角余弦,特别的需要为其添加<+,-,0>三种符号实现唯一区分
        angle_edge_bonds[i,2] = np.arccos(np.dot(bond_1,bond_3) / np.linalg.norm(bond_1) /np.linalg.norm(bond_3))/pi
        temp = np.dot(np.cross(bond_1,bond_2),bond_3)
        if temp == 0:
           angle_edge_bonds[i,3] =  1
        else:
           angle_edge_bonds[i,3] =  int(temp /abs(temp) + 1) 

    return angle_edge_bonds   

def Is_bonds_edge_index(bonds_edge_index,bonds_x):
    if int(bonds_x.shape[0]) - 1 == int(np.max(bonds_edge_index[:,2])):
        return True
    else :
        return False

# 生成特征矩阵
def generate_data(atomicNumList, xyz_coordinates,rdmol,matrix_mol):
    atoms_edge_index = generate_atoms_edge_index(matrix_mol,rdmol)
    bonds_edge_index,bonds_x = atoms2bonds_edge_index(atoms_edge_index,rdmol)
    if Is_bonds_edge_index(bonds_edge_index,bonds_x) == False:
        # print("Is error")
        return KeyError
    try:
        angle_edge_bonds = generate_angel_edge_bondes(xyz_coordinates,atoms_edge_index)
    except:
        # print("angle_edge_bonds")
        return KeyError
    bonds_edge_attr = np.c_[atoms_edge_index[:,1:3],bonds_x[bonds_edge_index[:,1]][:,2:]]
    bonds_edge_attr = np.c_[bonds_edge_attr,angle_edge_bonds]

    atomicNumList = torch.tensor(atomicNumList,dtype=torch.long)
    xyz_coordinates = torch.tensor(xyz_coordinates,dtype=torch.float32)
    bonds_x = torch.tensor(bonds_x,dtype=torch.float32)
    bonds_edge_index = torch.tensor(bonds_edge_index,dtype=torch.long)
    bonds_edge_attr = torch.tensor(bonds_edge_attr,dtype=torch.float32)
    # group = torch.tensor(group,dtype=torch.float32)
    return atomicNumList,xyz_coordinates,bonds_x,bonds_edge_index,bonds_edge_attr

# 根据Hessian 生成Hi，Hij
def HessiantoHiHij(Hessian: torch.tensor=None,atoms_edge_index:torch.tensor=None):
    atoms_count = Hessian.shape[0] // 3
    Hi = torch.zeros(size=[atoms_count,3,3],dtype = torch.float32)
    Hij = torch.zeros(size = [atoms_edge_index.shape[1],3,3])
    for i in range(atoms_count):
        Hi[i] = Hessian[i*3:i*3+3,i*3:i*3+3]
    for i in range(atoms_edge_index.shape[1]):
        Hij[i] = Hessian[atoms_edge_index[1,i]*3:atoms_edge_index[1,i]*3+3,atoms_edge_index[0,i]*3:atoms_edge_index[0,i]*3+3]
    return Hi, Hij

# 根据arc生成dipole,dedipole,polar,depolar,hessian
def str2dedipole(s:str,atoms_count:int):
    dedipole = torch.tensor(np.array(s.split(','),dtype=np.float32),dtype=torch.float32).reshape(atoms_count,3,3)
    return dedipole
def str2polar(s:str,atoms_count:int):
    polar = torch.zeros(size=[1,3,3],dtype=torch.float32)
    polar_list = np.array(s.split(','))
    for i in range(3):
        for j in  range(i+1):
            polar[0,i,j] = float(polar_list[(i*i+i)//2 + j])
            polar[0,j,i] = float(polar_list[(i*i+i)//2 + j])
    return polar
def str2depolar(s:str,atoms_count:int):
    depolar = torch.tensor(np.array(s.split(','),dtype=np.float32),dtype=torch.float32).reshape(atoms_count,3,6)
    return depolar
def str2hessian(s:str,atoms_count:int):
    hessian = torch.zeros(size=[int(3*atoms_count),int(3*atoms_count)])
    s = s.replace('\\\\',',')
    hessian_list = np.array(s.split(','),dtype=np.float32)
    for i in range(int(3*atoms_count)):
        for j in range(i+1):
            hessian[i,j] = float(hessian_list[(i*i+i)//2 + j])
            hessian[j,i] = float(hessian_list[(i*i+i)//2 + j])
    return hessian 
def arc2tensor(arc_filepath:str):
    with open(arc_filepath,'r') as f:
        line = f.readline()
        f.close()

    sxyz,line = line.split('\\\\Version=')
    atoms_count = len(sxyz.split('\\'))
    energy,line = line.split('\\RMSD=')
    energy = float(energy.split('\\HF=')[-1])
    line = line.split('\\Dipole=')[-1]
    dipole,line = line.split('\\DipoleDeriv=')
    dipole = torch.tensor(np.array(dipole.split(','),dtype=np.float32),dtype=torch.float32).reshape(1,-1)
    dedipole,line = line.split('\\Polar=')
    dedipole = str2dedipole(dedipole,atoms_count)
    polar,line = line.split('\\PolarDeriv=')
    polar = str2polar(polar,atoms_count)
    depolar,line = line.split('\\HyperPolar=')
    hessian = line.split('\\NImag=')[-1][3:]  
    # hessian = hessian.split('\\\\')[-1]
    depolar = str2depolar(depolar,atoms_count)
    hessian = str2hessian(hessian,atoms_count)
    return  energy,dipole,dedipole,polar,depolar,hessian
# 根据.b 和 .out文件生成振动模式频率和活性
def shiftactfromfile(b_filepath:str,out_filepath:str,max_step:int):
    with open(out_filepath ,'r') as f:
        lines_out = f.readlines()
        f.close()
    with open(b_filepath ,'r') as f:
        lines_8 = f.readlines()
        f.close() 
    out_index = lines_out.index(" FREQUENCIES\n") + 2
    shift = torch.zeros(size=[1,max_step],dtype=torch.float32)
    act = torch.zeros(size=[1,max_step],dtype=torch.float32)
    for j in range(2,len(lines_8),2):
        act[0,j//2 -1] = float(lines_8[j].split()[0])
        shift[0,j//2 -1] = float(lines_out[j//2 -1 + out_index].split()[2])
    return shift,act  

# 根据目标分子xyz生成对应的pt
def file2pt(xyz_pdb_path:str,arc_path:str=None):
    if os.path.basename(xyz_pdb_path).split('.')[1] == 'xyz':
        rdmol, AC = xyz2rdmol_AC(xyz_pdb_path)
    if os.path.basename(xyz_pdb_path).split('.')[1] == 'pdb':
        rdmol, AC = pdb2rdmol_AC(xyz_pdb_path)
    # 生成具有完备信息的rdkit_mol和邻接矩阵
    
    # 生成bonds_x,bonds_edge_index,bonds_edge_attr
    atoms_x,atoms_pos = rdmol2x_pos(rdmol)
    atoms_x,atoms_pos,bonds_x,bonds_edge_index,bonds_edge_attr  = generate_data(atoms_x,atoms_pos,rdmol,AC)
    atoms_edge_index = radius_graph(x=atoms_pos,r=5.0,batch=torch.zeros_like(atoms_x,dtype=torch.long)) 
    atoms_count = torch.tensor([[atoms_x.shape[0],atoms_edge_index.shape[1],rdmol.GetNumAtoms()]],dtype = torch.long)
    if arc_path != None:
        energy,dipole,dedipole,polar,depolar,hessian = arc2tensor(arc_path)
    # 打包生成data
        Hi,Hij = HessiantoHiHij(hessian,atoms_edge_index)
        data = Data(        x = bonds_x,    edge_attr = bonds_edge_attr,
                        edge_index = bonds_edge_index.T, atoms_pos = atoms_pos, 
                        atoms_x = atoms_x, atoms_count = atoms_count,
                        energy =torch.tensor([[energy]]), dipole = dipole,
                        dedipole = dedipole,polar = polar,
                        depolar = depolar, Hi = Hi,
                        Hij = Hij,
                        name = os.path.basename(xyz_pdb_path).split('.')[0],
                        
                        )
    else:
        data = Data(       x = bonds_x,    edge_attr = bonds_edge_attr,
                        edge_index = bonds_edge_index.T, atoms_pos = atoms_pos, 
                        atoms_x = atoms_x, atoms_count = atoms_count,
                        name = os.path.basename(xyz_pdb_path).split('.')[0],
                        
                        )
    return data


def rdmol2pt(rdmol:Chem.rdchem.Mol,name:str):
    
    atoms_x,atoms_pos = rdmol2x_pos(rdmol)
    AC = Chem.GetAdjacencyMatrix(rdmol)
    atoms_x,atoms_pos,bonds_x,bonds_edge_index,bonds_edge_attr  = generate_data(atoms_x,atoms_pos,rdmol,AC)
    atoms_edge_index = radius_graph(x=atoms_pos,r=5.0,batch=torch.zeros_like(atoms_x,dtype=torch.long)) 
    atoms_count = torch.tensor([[atoms_x.shape[0],atoms_edge_index.shape[1],rdmol.GetNumAtoms()]],dtype = torch.long)
    data = Data(       x = bonds_x,    edge_attr = bonds_edge_attr,
                        edge_index = bonds_edge_index.T, atoms_pos = atoms_pos, 
                        atoms_x = atoms_x, atoms_count = atoms_count,
                        name = name
                        
                        )
    return data

def np2data(atoms_x: np.array, atoms_pos: np.array, name):
    cache_path = f"/media/ramancloud/cache/{hash(name)}.xyz"
    np2xyz(atoms_x, atoms_pos, file_path=cache_path)
    rdmol, _ = xyz2rdmol_AC(cache_path)
    data = rdmol2pt(rdmol, name)
    os.remove(cache_path)
    return data

if __name__ == '__main__':

    # 示例数据
    z = np.array([8, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1])
    pos = np.array(
        [[ 9.7233e-02,  1.3689e+00,  1.3115e-01],
        [ 6.7777e-02,  3.2910e-03,  3.8910e-02],
        [ 7.6066e-02, -6.6728e-01, -1.1832e+00],
        [ 4.4743e-02, -2.0582e+00, -1.2085e+00],
        [ 5.2200e-03, -2.7850e+00, -2.4071e-02],
        [-2.7410e-03, -2.1055e+00,  1.1924e+00],
        [ 2.8234e-02, -7.1819e-01,  1.2309e+00],
        [ 1.2295e-01,  1.7542e+00, -7.5192e-01],
        [ 1.0686e-01, -1.0483e-01, -2.1109e+00],
        [ 5.1425e-02, -2.5716e+00, -2.1622e+00],
        [-1.9064e-02, -3.8668e+00, -4.6867e-02],
        [-3.3378e-02, -2.6606e+00,  2.1220e+00],
        [ 2.2412e-02, -1.7973e-01,  2.1697e+00]])

    data = np2data(z, pos, 'test')
    print('test')