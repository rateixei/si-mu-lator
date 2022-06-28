import numpy as np
import matplotlib as mpl
import numpy as np
import h5py
import sys
sys.path.append('../algorithms/')

import datatools
import trainingvariables
from glob import glob
from sklearn.utils import shuffle

files_loc = "/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/20220628/"
fdir = "atlas_mm_vmm_bkgr_1_TEST"
nevs=10000

all_files = glob(files_loc+fdir+'/*.h5')

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(all_files, max_files=500, sort_by='z')

vars_of_interest = np.zeros(dmat.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1
X = dmat[:,:,vars_of_interest]
target = Y_mu

X_test, Y_test = shuffle(X[:nevs], target[:nevs])

np.save(f"{fdir}_X_test_{nevs}.npy", X_test)
np.save(f"{fdir}_Y_test_{nevs}.npy", Y_test)