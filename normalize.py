import time as TM
import numpy as np
import os
import sys

sys.path.append('../Functions')

from DATA_ANALYSIS_FUNCTIONS import normalizing_tiles

# Adjusting the files to be float32 and change -9999.0 to np.nan

# os.chdir('/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps')
# files_list = [i for i in os.listdir() if i.endswith('.npy')]

# for FILE in files_list:
#     np_array = np.load(FILE).astype('float32')
#     np_array[np_array == -9999.0] = np.nan
#     np.save(f'/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps/{FILE}', np_array)

# normalize all of the files 
normalizing_tiles(tiles_dir='/mnt/locutus/remotesensing/r62/fire_danger/binary_data_maps',
                  axis=0,
                  meta_path = '/mnt/locutus/remotesensing/r62/fire_danger/meta/',
                  new_bins_path = '/mnt/locutus/remotesensing/r62/fire_danger/normed_bin_data/')

