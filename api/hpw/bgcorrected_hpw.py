import torch
import streamlit as st
import time
import numpy as np
from torch.autograd import Variable
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from api.hpw.model.Simple_FCN import F_CN


def normalization(data):
    data = data/np.max(data)
    return data

def distance_Reciprocal(a,b):
    d_all=a**2+b**2
    d=d_all**0.5
    d=1/d
    return d

def PEER(data,c):
    m = (c-1)//2
    n=len(data)
    new_data= [0 for x in range(n)]
    for i in range(0,n):
        w_sum=0
        if i <= m - 1:#左端不完整的几个点
            lx = 0
            ly = i + m
            p_sum=0
            w_sum=w_sum+data[0]*(m-i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range(lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        elif i >= n - m:#右端不完整的几个点
            lx = i - m
            ly = n-1
            p_sum=0
            w_sum=w_sum+data[n-1]*(m-n+i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        else:
            lx = i - m
            ly = i + m
            p_sum=0
            happens = [0 for x in range(ly-lx+1)]
            a=0
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
    return new_data


def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    return result

def WhittakerSmooth(x,w,lambda_,differences=2):
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def test(model, device, spectrum, spectrum_raw, spectrum_max):
    model.eval()
    spectrum = Variable(spectrum).to(device)
    output = model(spectrum)
    output = np.array(output.cpu().detach().numpy()[0, 0, :])
    spectrum = np.array(spectrum.cpu().detach().numpy()[0, 0, :])
    output1=np.arange(0,len(spectrum),1)
    for t in range(len(output)):
        if output[t]>0.5:
            output[t]=0
            output1[t]=1
        else: 
            output[t]=1
            output1[t]=0
    output[0]=1
    output[len(output)-1]=1
    output1[0]=0
    output1[len(output)-1]=0
    itermax=1000
    a4=np.array(output)
    for m in range(1,itermax+1):
        c6=WhittakerSmooth(spectrum_raw,a4,1000000000,2)
        d=spectrum_raw-c6
        d0=d[d<0]
        a0=a4[d<0]
        for v in range(len(d0)):
            if d0[v]<-0.001:
                a0[v]=a0[v]+0.1
            else:
                a0[v]=a0[v]
        a4[d<0]=a0
        a4[0]=max(a4)
        a4[-1]=max(a4)
    for m in range(1,itermax+1):
        c6=WhittakerSmooth(spectrum_raw,a4,100000000,2)
        d=spectrum_raw-c6
        d0=d[d<0]
        a0=a4[d<0]
        for v in range(len(d0)):
            if d0[v]<-0.001:
                a0[v]=a0[v]+0.1
            else:
                a0[v]=a0[v]
        a4[d<0]=a0
        a4[0]=max(a4)
        a4[-1]=max(a4)

    baseline_corrected = WhittakerSmooth(spectrum_raw,a4,100000000,2)
    spectrum_raw = spectrum_raw * spectrum_max
    baseline_corrected = baseline_corrected * spectrum_max
    output_corrected = spectrum_raw - baseline_corrected
    return output_corrected


@st.cache_resource
def build_net():
    net = F_CN()
    net.load_state_dict(torch.load("/media/ramancloud_beta/api/hpw/model_ckpt.pth", 
                                   map_location=torch.device('cpu')))
    return net



def reference(keys, spectrum):
    DEVICE = torch.device('cpu')
    begin_time = time.perf_counter()
    f = interp1d(keys, spectrum, kind='linear')
    keys_new = np.linspace(min(keys), max(keys), 2500)
    spectrum = f(keys_new)
    spectrum_raw = spectrum
    spectrum_max = max(spectrum_raw)
    spectrum_raw = normalization(spectrum_raw)
    spectrum = PEER(spectrum_raw , 5)
    for p in range(1):
        spectrum = PEER(spectrum, 5)
    spectrum = normalization(spectrum)
    spectrum = torch.tensor(spectrum)
    spectrum = spectrum.reshape(1, spectrum.shape[0])
    spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])  # data_type: batch_size x embedding_size x text_len
    spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
    spectrum = spectrum.permute(1, 0, 2)
    net = build_net()

    spectrum_processed = test(net, DEVICE, spectrum, spectrum_raw, spectrum_max)
    f_inv = interp1d(keys_new, spectrum_processed, kind='linear')
    spectrum_processed = f_inv(keys)
    end_time = time.perf_counter()
    print(f"time consumed: {end_time - begin_time:0.4f} seconds")
    return spectrum_processed
    