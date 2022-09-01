import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py
from sklearn import metrics

import sys
sys.path.append('../')
import datatools
from glob import glob
import trainingvariables
import json
from scipy.stats import skew, iqr

def sigmoid(x):
    return 1/(1 + np.exp(-x))

SIM="/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/"
DATA_LOC=f"{SIM}/stgc/atlas_nsw_pad_z0_stgc20Max1_bkgr_1_CovAngle_TRAIN/*.h5"
detmat="/sdf/home/r/rafaeltl/home/Muon/21062022/si-mu-lator/cards/atlas_nsw_pad_z0_stgc20Max1.yml"
linearized = True

files=glob(DATA_LOC)

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(files, max_files=500, sort_by='z')
this_cut=(Y_mu==1)


X_pad = datatools.training_prep(dmat, sig_keys)
X_det = datatools.detector_matrix(dmat, sig_keys, detcard=detmat)

assert X_pad.shape[2] == X_det.shape[2]

vars_of_interest = np.zeros(X_pad.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1

mdicts = []

modloc = "models/MyTCN*"
allmodels = glob(modloc)

for mloc in allmodels:
    
    # if 'QKeras' in mloc:
    #     import qkeras as keras
    # else:
    from tensorflow import keras

    # if 'MyTCN_64,3,1:10,3,1_' not in mloc: 
    #     continue
    if mloc+'/history.npy' not in glob(mloc+'/*'): 
        print("Training hasn't finished...")
        continue

    if mloc+'/saved_model.pb' not in glob(mloc+'/*') and mloc+'/saved_model.pbtxt' not in glob(mloc+'/*'):
        print("Couldn't find saved model")
        continue
        
    if mloc+'/perf.json' in glob(mloc+'/*'): 
        print("Performance already saved...")
        continue
    
    print(mloc)
    mdict = {}
    mdict['name'] = mloc
    mdict['clayers'] = ''
    mdict['dlayers'] = ''
    mdict['CBNorm'] = 'CBNormTrue' in mloc
    mdict['DBNorm'] = 'DBNormTrue' in mloc
    mdict['IBNorm'] = 'IBNormTrue' in mloc
    mdict['LL'] = ''
    mdict['ptype'] = ''
    mdict['penX'] = 'penXTrue' in mloc
    mdict['penA'] = 'penATrue' in mloc
    mdict['bkgPen'] = 'bkgPenTrue' in mloc
    mdict['regBias'] = 'regBiasTrue' in mloc
    mdict['qkeras'] = 'QKeras' in mloc
    mdict['qbits'] = 0
    mdict['qibits'] = 0
    mdict['pool'] = 0
    mdict['l1reg'] = 0
    mdict['detmat'] = 0
    if '_AvgPool' in mloc: mdict['pool'] = 1
    if '_Flatten' in mloc: mdict['pool'] = 2
    
    confs = mloc.split('_')
    for cc in confs:
        if 'CL' in cc: mdict['clayers'] = cc
        elif 'DL' in cc: mdict['dlayers'] = cc
        elif 'll' in cc: mdict['LL'] = cc
        elif 'ptype' in cc: mdict['ptype'] = cc
        elif 'QKeras.b' in cc: mdict['qbits'] = cc
        elif 'QKeras.i' in cc: mdict['qibits'] = cc
        elif 'L1R' in cc: mdict['l1red'] = cc

    if 'DetMat' in mloc:
        X_prep = X_det
        mdict['detmat'] = 1
    else:
        X_prep = X_pad
             
    model_loc = '../models/'
    
    model = keras.models.load_model(mloc,compile=False)
    
    try:
        preds = model.predict(X_prep[:,:,vars_of_interest], batch_size=1024)
    except AttributeError:
        print("Couldn't predict")
        continue

    
    mult_fact = max(data['ev_mu_x'])
    mult_facta = max(data['ev_mu_theta'])
    
    yhat = sigmoid(preds[:,0]) if linearized else preds[:,0]
    x_reg = preds[:,1]*mult_fact
    a_reg = preds[:,2]*mult_facta
    
    mdict['npars'] = model.count_params()
    mdict['mod_mae_x'] = metrics.mean_absolute_error( data['ev_mu_x'][this_cut], x_reg[this_cut] )
    mdict['mod_mae_a'] = metrics.mean_absolute_error( data['ev_mu_theta'][this_cut], a_reg[this_cut] )

    mdict['mod_mse_x'] = metrics.mean_squared_error( data['ev_mu_x'][this_cut], x_reg[this_cut] )
    mdict['mod_mse_a'] = metrics.mean_squared_error( data['ev_mu_theta'][this_cut], a_reg[this_cut] )

    mdict['mod_auc'] = metrics.roc_auc_score( Y_mu, yhat )

    mdict['mod_dx_median'] = np.median( data['ev_mu_x'][this_cut] - x_reg[this_cut] )
    mdict['mod_da_median'] = np.median( data['ev_mu_theta'][this_cut] - a_reg[this_cut] )
    
    mdict['mod_dx_mean'] = ( data['ev_mu_x'][this_cut] - x_reg[this_cut] ).mean()
    mdict['mod_da_mean'] = ( data['ev_mu_theta'][this_cut] - a_reg[this_cut] ).mean()
    
    mdict['mod_dx_std'] = ( data['ev_mu_x'][this_cut] - x_reg[this_cut] ).std()
    mdict['mod_da_std'] = ( data['ev_mu_theta'][this_cut] - a_reg[this_cut] ).std()
    
    mdict['mod_dx_iqr'] = iqr( data['ev_mu_x'][this_cut] - x_reg[this_cut], rng=(16, 84) )
    mdict['mod_da_iqr'] = iqr( data['ev_mu_theta'][this_cut] - a_reg[this_cut], rng=(16, 84) )

    mdict['skew_dx'] = skew( data['ev_mu_x'][this_cut] - x_reg[this_cut] )
    mdict['skew_da'] = skew( data['ev_mu_theta'][this_cut] - a_reg[this_cut] )
    
    mdict['dx_mmm'] = ( mdict['mod_dx_mean'] - mdict['mod_dx_median'] ) / mdict['mod_dx_std']
    mdict['da_mmm'] = ( mdict['mod_da_mean'] - mdict['mod_da_median'] ) / mdict['mod_da_std']
    
    dx_hist = np.histogram( ( data['ev_mu_x'][this_cut] - x_reg[this_cut] ), range=(-1,1), bins=100 )
    dx_bins = 0.5*( dx_hist[1][:-1] + dx_hist[1][1:] )
    da_hist = np.histogram( ( data['ev_mu_theta'][this_cut] - a_reg[this_cut] ), range=(-0.02,0.02), bins=100 )
    da_bins = 0.5*( da_hist[1][:-1] + da_hist[1][1:] )
    
    mdict['dx_mode'] = dx_bins[ np.argmax( dx_hist[0] ) ]
    mdict['da_mode'] = da_bins[ np.argmax( da_hist[0] ) ]
    
    
    with open(mloc+'/perf.json', 'w') as fp:
        json.dump(mdict, fp, indent="\t") 
        print("Saved", mloc+'/perf.json')
