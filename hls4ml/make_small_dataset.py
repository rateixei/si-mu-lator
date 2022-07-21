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
do_det_matrix=False
det_card="../cards/atlas_mm_vmm.yml"

all_files = glob(files_loc+fdir+'/*.h5')

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(all_files, max_files=500, sort_by='z')

if do_det_matrix:
    X_prep = datatools.detector_matrix(dmat, sig_keys, det_card)
else:
    X_prep = datatools.training_prep(dmat, sig_keys)

vars_of_interest = np.zeros(X_prep.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1
X = X_prep[:,:,vars_of_interest]

target = Y_mu

X_test, Y_test = shuffle(X, target)

out_name_tag = f"test_{nevs}_padMat_noSig.npy"
if do_det_matrix:
    det_card_name = det_card.split('/')[-1].replace('.yml', '')
    out_name_tag = f"test_{nevs}_detMat_{det_card_name}.npy"

np.save(f"X_{out_name_tag}", X_test[:nevs])
np.save(f"Y_{out_name_tag}", Y_test[:nevs])
