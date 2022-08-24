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

files_loc = "/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/stgc/"
# fdir = "atlas_mm_vmm_bkgr_1_TEST"
fdir = "atlas_nsw_pad_z0_stgc20Max1_bkgr_1_CovAngle_TRAIN"
nevs=100000
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

mult_fact_X = max(data['ev_mu_x'])
mult_fact_a = max(data['ev_mu_theta'])
print(f"#&#&#&#&#&#&# X mult fact = {mult_fact_X}, Angle mult fact = {mult_fact_a} #&#&#&#&#&#&#")
data_ev_mu_x = (data['ev_mu_x'])/mult_fact_X
data_ev_mu_a = (data['ev_mu_theta'])/mult_fact_a

X_test, Y_clas_test, Y_xreg_test, Y_areg_test = shuffle(X, Y_mu, data_ev_mu_x, data_ev_mu_a)

Y_test = np.zeros( (Y_clas_test.shape[0], 3 ) )
Y_test[:,0] = Y_clas_test
Y_test[:,1] = Y_xreg_test
Y_test[:,2] = Y_areg_test

out_name_tag = f"test_{nevs}_padMat_noSig_{fdir}.npy"
if do_det_matrix:
    det_card_name = det_card.split('/')[-1].replace('.yml', '')
    out_name_tag = f"test_{nevs}_detMat_{det_card_name}.npy"

np.save(f"X_{out_name_tag}", X_test[:nevs])
np.save(f"Y_{out_name_tag}", Y_test[:nevs])
