import numpy as np
import os 

def patching_numpy_array(ARR, n, PRINT=False):
    if PRINT:
        print('ORIGINAL ARRAY SHAPE:', ARR.shape)
    arr_shape = ARR.shape

    axis_1_steps = arr_shape[1]//n
    axis_2_steps = arr_shape[2]//n
    
    data = np.split(ARR[:, :n*axis_1_steps, :], axis_1_steps, axis=1)
    if axis_1_steps < arr_shape[1]/n:
        data.append(ARR[:, -n:, :])
    
    data = np.stack(data)
    
    lis = []
    for sub_arr in data:
        lis.append(np.stack(np.split(sub_arr[:, :, :n*axis_2_steps], axis_2_steps, axis = 2)))
        if axis_2_steps < arr_shape[2]/n:
            lis.append(np.expand_dims(sub_arr[:, :, -n:], 0))
    final = np.concatenate(lis, axis = 0)
    if PRINT:
        print('OUTPUT ARRAY SHAPE:', final.shape)
    return final

os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/normed_bin_data')
for FILE in os.listdir():
    ARR = np.load(FILE)
    ARR = patching_numpy_array(ARR, 128)
    for idx, a in enumerate(ARR):
        np.save(f'./{idx}_{FILE}', a)
    os.remove(f'./{FILE}')
    print('FILE:', FILE, 'PROCESSED')