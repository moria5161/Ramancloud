import numpy as np

def read_horiba(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        waves_line = f.readline()
        waves = np.array(waves_line.split(), dtype=float)
        data_block = np.loadtxt(f)

    x_coords = data_block[:, 0]
    y_coords = data_block[:, 1]
    spectra_matrix = data_block[:, 2:]

    if waves[0] > waves[-1]:
        waves = waves[::-1]
        spectra_matrix = spectra_matrix[:, ::-1]

    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    map_size = (len(unique_x), len(unique_y))

    return {
        'waves': waves,
        'x': unique_x,
        'y': unique_y,
        'size': map_size,
        'spectra': spectra_matrix
    }

def write_horiba(processed_data, save_path):
    waves = processed_data['waves']
    unique_x = processed_data['x']
    unique_y = processed_data['y']
    spectra = processed_data['spectra']
    
    num_x, num_y = len(unique_x), len(unique_y)
    
    if spectra.shape[0] != num_x * num_y:
        raise ValueError("光谱矩阵的行数与坐标点总数不匹配。")
    if spectra.shape[1] != len(waves):
        raise ValueError("光谱矩阵的列数与光谱轴的点数不匹配。")
        
    x_coords_full = np.tile(unique_x, num_y)
    y_coords_full = np.repeat(unique_y, num_x)

    with open(save_path, 'w', encoding='utf-8') as f:
        waves_str = '\t'.join([f'{w:.2f}' for w in waves])
        f.write(f"\t\t{waves_str}\n")
        
        for i in range(spectra.shape[0]):
            x_coord = x_coords_full[i]
            y_coord = y_coords_full[i]
            spec_line = '\t'.join([f'{val:.6f}' for val in spectra[i, :]])
            f.write(f"{x_coord}\t{y_coord}\t{spec_line}\n")
            
    return save_path