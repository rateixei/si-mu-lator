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
linearized = True

files=glob(DATA_LOC)

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(files, max_files=500, sort_by='z')
this_cut=(Y_mu==1)
X_pad = datatools.training_prep(dmat, sig_keys)

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
        # continue

    if mloc+'/saved_model.pb' not in glob(mloc+'/*') and mloc+'/saved_model.pbtxt' not in glob(mloc+'/*'):
        print("Couldn't find saved model")
        continue
        
    if mloc+'/perf.json' in glob(mloc+'/*'): 
        print("Performance already saved...")
        continue
    
    print(mloc)
    mdict = {}
    mdict['name'] = mloc

    X_prep = X_pad
            
    vars_of_interest = np.zeros(X_prep.shape[2], dtype=bool)
    training_vars = trainingvariables.tvars
    for tv in training_vars:
        vars_of_interest[sig_keys.index(tv)] = 1
    
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
    mdict['qkeras'] = 0

    confs = mloc.split('_')
    
    base_ind = 0
    if 'QKeras' in mloc:
        base_ind = 3
        mdict['qkeras'] = 1
        mdict['qbits'] = confs[1]
        mdict['qibits'] = confs[2]

    mdict['clayers'] = confs[base_ind+1]
    mdict['dlayers'] = confs[base_ind+2]
    mdict['CBNorm'] = 1.0 if 'True' in confs[base_ind+3] else 0.0
    mdict['DBNorm'] = 1.0 if 'True' in confs[base_ind+4] else 0.0
    mdict['LL'] = float(confs[base_ind+5].replace('ll', ''))
    mdict['ptype'] = float(confs[base_ind+6].replace('ptype', ''))
    mdict['penX'] = 1.0 if 'True' in confs[base_ind+7] else 0.0
    mdict['penA'] = 1.0 if 'True' in confs[base_ind+8] else 0.0
    mdict['bkgPen'] = 1.0 if 'True' in confs[base_ind+9] else 0.0
    mdict['regBias'] = 1.0 if 'True' in confs[base_ind+10] else 0.0
    mdict['res'] = 1.0 if 'ResNet' in mloc else 0.0
    mdict['pool'] = 0
    if '_AvgPool' in mloc: mdict['pool'] = 1
    if '_Flatten' in mloc: mdict['pool'] = 2
    
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
