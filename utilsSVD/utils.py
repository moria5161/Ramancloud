import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import LRMA
import matlab


def lrma_denoise(x, mode='alrma', C=400, ref=None):
    
    
    lrma = LRMA.initialize()
    mat_in = matlab.double(x.tolist())

    if ref and mode =='clrma':
        mat_ref = matlab.double(ref.tolist())

    if mode == 'alrma':
        mat_res = lrma.ALRMA(mat_in, C,)
    elif mode == 'clrma':
        mat_res = lrma.CLRMA(mat_in, ref, C, )

    res = np.array(mat_res)
    return res




if __name__ == "__main__":
    
    
    data = loadmat('target_2s.mat')
    target = data['cube']

    res = lrma_denoise(target, mode='alrma')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(target.reshape(1337, -1, 400)[284])
    plt.subplot(1, 2, 2)
    plt.imshow(res.reshape(1337, -1, 400)[284])
    plt.savefig('test.png')
