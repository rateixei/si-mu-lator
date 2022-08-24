import numpy as np
import argparse
import sys
import glob 
import datetime

import datatools
import mlmodels

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import trainingvariables

parser = argparse.ArgumentParser(description='Train neural network')

parser.add_argument('-f', '--files', dest='files', type=str, required=True,
                    help='Files')
parser.add_argument('-m', '--model', dest='mod_type', required=True,
                    choices=['deepsets', 'attn_deepsets', 'lstm', 'gru', 'mlp', 'tcn'],
                    help='Type of network')
parser.add_argument('-t', '--task', dest='task_type', default='classification',
                    choices=['classification', 'regression'],
                    help='Type of training task')
parser.add_argument('--do-angle', dest='do_angle', default=False, action='store_true',
                    help='Regress angle as well as X')
parser.add_argument('-l', '--label', dest='label', default='none')
parser.add_argument('--batchnorm', dest='bnorm', default=False, action='store_true',
                    help='use batch norm layer')
parser.add_argument('--masking', dest='masking', default=False, action='store_true',
                    help='use masking layer')
parser.add_argument('--detector', dest='detcard', default='none', type=str,
                    help='Pass detector card to build detector matrix')
parser.add_argument('--lambda', dest='ll', default=0, type=int,
                    help='Combined loss constant')

args = parser.parse_args()

file_names = glob.glob(args.files)

if len(file_names) < 1:
    print("Couldn't find any files from expression", args.files)
    sys.exit()

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(file_names, max_files=500, sort_by='z')

print('Data matrix shape:', dmat.shape)
print('Data matrix keys:', sig_keys)
print('Y shape:', Y.shape)
print('Y_mu shape:', Y_mu.shape)
print('Y_hit shape:', Y_hit.shape)
print('N signal:', (Y_mu == 1).sum())
print('N background:', (Y_mu == 0).sum())

if 'none' not in args.detcard:
    print('Building detector-based data matrix')
    X_prep = datatools.detector_matrix(dmat, sig_keys, args.detcard)
else:
    X_prep = datatools.training_prep(dmat, sig_keys)
    

vars_of_interest = np.zeros(X_prep.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1
X = X_prep[:,:,vars_of_interest]

my_model = 0
n_reg = 0
mod_name = args.mod_type

if 'none' not in args.label:
    mod_name += '_' + args.label

mod_name += f'_BatchNorm{args.bnorm}'
mod_name += f'_Masking{args.masking}'
mod_name += '_' + args.task_type
if 'regression' in args.task_type:
    mod_name += str(args.ll)
if args.do_angle:
    mod_name += '_Angle'
    
now = datetime.datetime.now()
date_time = now.strftime("_%d%m%Y_%H.%M.%S")
mod_name += date_time

if 'none' not in args.detcard:
    mod_name += '_detMatrix'
else:
    mod_name += '_normalMatrix'

reg_outs = 1 if args.task_type=='regression' else 0
reg_outs += 1 if args.do_angle else 0

if args.mod_type == 'deepsets':
    my_model = mlmodels.model_deep_set_muon(input_shape=(X.shape[1],X.shape[2]), 
        phi_layers=[50,50,50], 
        F_layers=[20,10], 
        batchnorm=args.bnorm, mask_v=-99, 
        add_selfattention=False,
        do_reg_out=n_reg,
        masking=args.masking)
elif args.mod_type == 'attn_deepsets':
    my_model = mlmodels.model_deep_set_muon(input_shape=(X.shape[1],X.shape[2]), 
        phi_layers=[50,50,50], 
        F_layers=[20,10], 
        batchnorm=args.bnorm, mask_v=-99, 
        add_selfattention=True,
        do_reg_out=n_reg,
        masking=args.masking)
elif args.mod_type == 'lstm' or args.mod_type == 'gru':
    my_model = mlmodels.model_recurrent_muon(input_shape=(X.shape[1],X.shape[2]),
        rec_layer=args.mod_type,
        rec_layers=[10], 
        F_layers=[10,5], 
        batchnorm=args.bnorm, mask_v=-99, 
        do_reg_out=0,
        masking=args.masking)
elif args.mod_type == 'mlp':
    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    my_model = mlmodels.model_mlp_muon(input_shape=X.shape[1],
        dense_layers=[X.shape[1],100,50,10], 
        batchnorm=args.bnorm, 
        do_reg_out=0)
elif args.mod_type == 'tcn':
    my_model = mlmodels.model_tcn_muon(input_shape=(X.shape[1],X.shape[2]),
        convs_1ds=[ (10,3,1), (10,3,1) ],
        F_layers=[20,20], 
        batchnorm_dense=False, 
        batchnorm_conv=True,
        do_reg_out=reg_outs,
        ll=args.ll)
else:
    print("!!!")
    sys.exit()

if args.task_type=='regression':
    print("~~ This is a combined classification+regression task ~~")
    mult_fact_X = max(data['ev_mu_x'])
    mult_fact_a = max(data['ev_mu_theta'])
    print(f"#&#&#&#&#&#&# X mult fact = {mult_fact_X}, Angle mult fact = {mult_fact_a} #&#&#&#&#&#&#")
    data_ev_mu_x = (data['ev_mu_x'])/mult_fact_X
    data_ev_mu_a = (data['ev_mu_theta'])/mult_fact_a
    X_train, Y_clas_train, Y_xreg_train, Y_areg_train = shuffle(X, Y_mu, data_ev_mu_x, data_ev_mu_a)
    Y_train = np.zeros( (Y_clas_train.shape[0], reg_outs+1 ) ) 
    Y_train[:,0] = Y_clas_train
    Y_train[:,1] = Y_xreg_train
    if args.do_angle:
        Y_train[:,2] = Y_areg_train
elif args.task_type=='classification':
    X_train, Y_train = shuffle(X, Y_mu)

my_model.save(f'models/{mod_name}')
model_json = my_model.to_json()
with open(f'models/{mod_name}/arch.json', 'w') as json_file:
    json_file.write(model_json)

history = my_model.fit( X_train, Y_train,
                        callbacks = [
                                EarlyStopping(monitor='val_loss', patience=1000, verbose=1),
                                ModelCheckpoint(f'models/{mod_name}/weights.h5', monitor='val_loss', verbose=True, save_best_only=True) ],
                        epochs=3000,
                        validation_split = 0.2,
                        batch_size=2**14,
                        verbose=1
                       )
    
my_model.load_weights(f'models/{mod_name}/weights.h5')
my_model.save(f'models/{mod_name}')

np.save(f'models/{mod_name}/history.npy', history.history)
