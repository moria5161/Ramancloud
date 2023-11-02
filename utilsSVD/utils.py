import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import ALRMA
import matlab


def Alrma_denoise(x, mode='aAlrma', C=400, ref=None):
    
    
    Alrma = ALRMA.initialize()
    mat_in = matlab.double(x.tolist())

    if mode =='cAlrma':
        ref = matlab.double(ref.tolist())

    img_region = np.arange(1, 3).astype(float).tolist()

    if mode == 'aAlrma':
        mat_res = Alrma.ALRMA(mat_in, C, matlab.double(img_region), 5, 10, 0.001, 0.01)
    elif mode == 'cAlrma':
        mat_res = Alrma.CLRMA(mat_in, ref, C, matlab.double(img_region), 5, 10, 0.001, 0.01)

    res = np.array(mat_res)
    Alrma.terminate()
    return res


if __name__ == "__main__":
    
    
    data = loadmat('/home/room/streamlit/ramancloud_beta/samples/target_2s.mat')
    ref = loadmat('/home/room/streamlit/ramancloud_beta/samples/ref1.mat')
    target = data['cube']
    ref = ref['cube']

    res = Alrma_denoise(target, mode='aAlrma')
    res2 = Alrma_denoise(target, mode='cAlrma', ref=ref)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(target.reshape(1337, -1, 400)[284])
    plt.subplot(1, 3, 2)
    plt.imshow(res.reshape(1337, -1, 400)[284])
    plt.subplot(1, 3, 3)
    plt.imshow(res2.reshape(1337, -1, 400)[284])
    plt.savefig('test2.png')
