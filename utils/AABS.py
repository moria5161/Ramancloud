import os 
import numpy as np
import pandas as pd
import time 
import uuid

main_path = '/home/room/streamlit/ramancloud_public/utils/algorithms/main'
def auto_adaptive(df:pd.Series):
    now = time.time()
    now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(now))
    cache_filename = f'cache_{now}_{uuid.uuid4()}.txt'
    cache = df.to_numpy()
    np.savetxt(cache_filename, np.c_[np.arange(len(cache)), cache])
    command = f'{main_path} {cache_filename} pre_{cache_filename}'
    os.system(command)
    res = np.loadtxt(f'pre_{cache_filename}')[:, -1]
    # delete cache
    os.remove(f'{cache_filename}')
    os.remove(f'pre_{cache_filename}')
    return res[:-1]