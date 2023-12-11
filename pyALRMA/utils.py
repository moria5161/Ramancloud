import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import ALRMA
import matlab


def ALRMADenoise(x, spec_region, mode='ALRMA', img_columns=400, window_size=5, count=10, ref=None):
    
    Alrma = ALRMA.initialize()
    mat_in = matlab.double(x.tolist())

    if mode =='CLRMA':
        ref = matlab.double(ref.tolist())

    if type(spec_region) != list:
        spec_region = list(spec_region)

    img_rows = x.shape[-1] // img_columns
    if img_rows * img_columns != x.shape[-1]:
        raise ValueError('Image shape error! Please check the image columns.')
    if (img_rows - window_size + 1) * (img_columns - window_size + 1) < count:
        raise ValueError('Please reduce the count or window size.')
    if mode == 'ALRMA':
        mat_res = Alrma.ALRMA(mat_in, img_columns, matlab.double(spec_region), window_size, count, 0.001, 0.01)
    elif mode == 'CLRMA':
        mat_res = Alrma.CLRMA(mat_in, ref, img_columns, matlab.double(spec_region), window_size, count, 0.001, 0.01)

    res = np.array(mat_res)
    Alrma.terminate()
    return res


if __name__ == "__main__":
    
    data = loadmat('/home/room/streamlit/ramancloud_beta/samples/mapping/target_2s.mat')
    ref = loadmat('/home/room/streamlit/ramancloud_beta/samples/mapping/ref1.mat')
    target = data['cube']
    ref = ref['cube']

    res = ALRMADenoise(target, mode='ALRMA')
    res2 = ALRMADenoise(target, mode='CLRMA', ref=ref)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(target.reshape(1337, -1, 400)[284])
    plt.subplot(1, 3, 2)
    plt.imshow(res.reshape(1337, -1, 400)[284])
    plt.subplot(1, 3, 3)
    plt.imshow(res2.reshape(1337, -1, 400)[284])
    plt.savefig('test2.png')
