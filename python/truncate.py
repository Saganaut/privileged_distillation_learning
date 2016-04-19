import scipy.io as sio
import numpy as np
import os

data_path = '/Volumes/Samsung T3/cse8803_mip/stats_298_grain_grain/'
new_path_root = '/Volumes/Samsung T3/cse8803_mip/grain_grain/truncated/'
files = os.listdir(data_path)
for fi in files:
  full = os.path.join(data_path,fi)
  new_path = os.path.join(new_path_root, fi)
  print fi
  a = sio.loadmat(full)['stats']
  print a.shape
  truncated_data = a[0,100:199,100:199,100:199,0]
  np.save(new_path, truncated_data)


