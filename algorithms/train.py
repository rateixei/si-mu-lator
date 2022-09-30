import numpy as np
import argparse
import sys
import glob 
import datetime

import datatools
import mlmodels

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import trainingvariables

parser = argparse.ArgumentParser(description='Train neural network')

parser.add_argument('-f', '--files', dest='files', type=str, required=True,
                    help='Files')
parser.add_argument('--detmat', dest='detmat', default='none', type=str)

parser.add_argument('--test', dest='test', default=False, action='store_true',
                    help='Testing mode')

## batch norm
parser.add_argument('--conv-batchnorm', dest='convbnorm', default=False, action='store_true',
                    help='use batch norm layer')
parser.add_argument('--dense-batchnorm', dest='densebnorm', default=False, action='store_true',
                    help='use batch norm layer')
parser.add_argument('--input-batchnorm', dest='inputbnorm', default=False, action='store_true',
                    help='use batch norm layer')
parser.add_argument('--batchnorm-constraint', dest='bnormconst', default=False, action='store_true',
                    help='Constrain batchnorm weights')

## loss penalty
parser.add_argument('--no-pen', dest='penalty', default=True, action='store_false',
                    help='Remove loss penalty')
parser.add_argument('--pen-type', dest='pentype', default=0, type=int, choices=[0, 1, 2],
                    help='Penalty type: 0, 1 or 2')

## layers
parser.add_argument('--clayers', dest='clayers', default="64,3,1,0:64,3,1,0", type=str,
                    help='Convolution layers')
parser.add_argument('--dlayers', dest='dlayers', default="20:10", type=str,
                    help='Dense layers')
parser.add_argument('--avgpool', dest='avgpool', default=False, action='store_true',
                    help='Use average pooling')
parser.add_argument('--flatten', dest='flatten', default=False, action='store_true',
                    help='Flatten instead of pooling')

## weight adjustment
parser.add_argument('--no-bias', dest='regbias', default=True, action='store_false',
                    help='Remove bias in output regression layers')
parser.add_argument('--l1reg', dest='l1reg', default=0, type=float)
parser.add_argument('--lrate', dest='learning_rate', default=0.001, type=float)


## qkeras mode
parser.add_argument('--do-q', dest='qkeras', default=False, action='store_true',
                    help='Use qkeras')
parser.add_argument('--q-bits', dest='qbits', default=0, type=int, help='qKeras bits')
parser.add_argument('--q-ibits', dest='qibits', default=0, type=int, help='qKeras bits')

## double training
parser.add_argument('--bigger-model', dest='bmod', default='none', type=str,
                    help='use bigger model as reference')

args = parser.parse_args()

mod_name = 'MyTCN_'
mod_name += 'CL'+args.clayers + '_'
mod_name += 'DL'+args.dlayers + '_'
mod_name += f'CBNorm{args.convbnorm}_DBNorm{args.densebnorm}_IBNorm{args.inputbnorm}_'
mod_name += f'pen{args.penalty}_'
mod_name += f'ptype{args.pentype}_'
mod_name += f'regBias{args.regbias}_'
mod_name += f'lrate{args.learning_rate}'
if args.avgpool:
    mod_name += '_AvgPool'
elif args.flatten:
    mod_name += '_Flatten'
else:
    mod_name += '_MaxPool'
if args.l1reg > 0:
    mod_name += f'_L1R{args.l1reg:.4f}'
if 'none' not in args.detmat:
    mod_name += '_DetMat'
    
mod_name = mod_name.replace(',', '.')
mod_name = mod_name.replace(':', '..')

file_names = glob.glob(args.files)

if len(file_names) < 1:
    print("Couldn't find any files from expression", args.files)
    sys.exit()

mfiles = 1000
if args.test: mfiles = 2
data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(file_names, max_files=mfiles, sort_by='z')

print('Data matrix shape:', dmat.shape)
print('Data matrix keys:', sig_keys)
print('Y shape:', Y.shape)
print('Y_mu shape:', Y_mu.shape)
print('Y_hit shape:', Y_hit.shape)
print('N signal:', (Y_mu == 1).sum())
print('N background:', (Y_mu == 0).sum())

if 'none' in args.detmat:
    X_prep = datatools.training_prep(dmat, sig_keys)    
else:
    X_prep = datatools.detector_matrix(dmat, sig_keys, detcard=args.detmat)    

vars_of_interest = np.zeros(X_prep.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1
X = X_prep[:,:,vars_of_interest]

conv_layers = [ ( int(cl.split(',')[0]), int(cl.split(',')[1]), int(cl.split(',')[2]), int(cl.split(',')[3]) ) for cl in args.clayers.split(':')]
print(conv_layers)

if 'none' in args.dlayers:
    dense_layers = []
else:
    dense_layers = [ int(dl) for dl in args.dlayers.split(':') ]

pooling = 'max'
if args.avgpool:
    pooling = 'average'
if args.flatten:
    pooling = 'flat'
    
n_outs = 4

mod_name += f'_{n_outs}Outputs'
mod_name += '_LONG'

my_model_class = mlmodels.MyTCNModel(
        input_shape=(X.shape[1],X.shape[2]),
        convs_1ds=conv_layers,
        F_layers=dense_layers, 
        batchnorm_dense=args.densebnorm,
        batchnorm_conv=args.convbnorm, 
        batchnorm_inputs=args.inputbnorm,
        batchnorm_constr=args.bnormconst,
        n_outs=n_outs,
        l1_reg=args.l1reg, l2_reg=0,
        use_pen=args.penalty, pen_type=args.pentype,
        do_bias=args.regbias,
        pooling=pooling,
        learning_rate=args.learning_rate)

my_model = my_model_class.model()

print("~~ This is a combined classification+regression task ~~")
mult_fact_X = max(data['ev_mu_x'])
mult_fact_a = max(data['ev_mu_theta'])
mult_fact_Y = max(data['ev_mu_y'])
print(f"#&#&#&#&#&#&# X mult fact = {mult_fact_X}, Angle mult fact = {mult_fact_a} #&#&#&#&#&#&#")
data_ev_mu_x = (data['ev_mu_x'])/mult_fact_X
data_ev_mu_a = (data['ev_mu_theta'])/mult_fact_a
data_ev_mu_y = (data['ev_mu_y'])/mult_fact_Y

X_train, Y_xreg_train, Y_areg_train, Y_yreg_train = shuffle(X, data_ev_mu_x, data_ev_mu_a, data_ev_mu_y)

Y_train = np.zeros( (X_train.shape[0], 2 ) ) 
Y_train[:,0] = Y_xreg_train
Y_train[:,1] = Y_areg_train

if 'none' not in args.bmod:
    from tensorflow import keras
    
    mod_name += '_BIG'
    big_ref_model = keras.models.load_model(args.bmod,compile=False)
    big_ref_preds = big_ref_model.predict(X_train, batch_size=1024)
    Y_train[:,0] = big_ref_preds[:,0]
    Y_train[:,1] = big_ref_preds[:,2]
    
# if n_outs > 2:
#     Y_train[:,2] = Y_yreg_train

if args.qkeras:
    
    mod_name = f'QKeras.b{args.qbits}_QKeras.i{args.qibits}_' + mod_name
    
    q_dict = {
        "QActivation": {
            "relu": "quantized_relu({args.qbits},0)"
        },
        "QBatchNormalization": {
            "kernel_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)",
            "bias_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)"
        },
        "QConv1D": {
            "kernel_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)",
            "bias_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)"
        },
        "QDense": {
            "kernel_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)",
            "bias_quantizer": f"quantized_bits({args.qbits},{args.qibits},1)"
        }
    }
    
    from qkeras.utils import model_quantize
    
    q_model = model_quantize(my_model, q_dict, 4)
    q_model.summary()
    
    combloss = my_model_class.MyLoss()

    from tensorflow.keras.optimizers import Adam
    
    opt = Adam(learning_rate=0.01)
    q_model.compile(loss=combloss, optimizer=opt)
    
    q_model.save(f'models/{mod_name}')
    model_json = q_model.to_json()
    with open(f'models/{mod_name}/arch.json', 'w') as json_file:
        json_file.write(model_json)
        
    history = q_model.fit( X_train, Y_train,
                            callbacks = [
                                    EarlyStopping(monitor='val_loss', patience=5000, verbose=1), # lr_callback,
                                    ModelCheckpoint(f'models/{mod_name}/weights.h5', monitor='val_loss', verbose=True, save_best_only=True) ],
                            epochs=5000,
                            validation_split = 0.25,
                            batch_size=2**14,
                            verbose=2
                           )
    
    q_model.load_weights(f'models/{mod_name}/weights.h5')
    q_model.save(f'models/{mod_name}')

    np.save(f'models/{mod_name}/history.npy', history.history)
    
else:
    my_model.save(f'models/{mod_name}')
    model_json = my_model.to_json()
    with open(f'models/{mod_name}/arch.json', 'w') as json_file:
        json_file.write(model_json)

    history = my_model.fit( X_train, Y_train,
                            callbacks = [
                                    EarlyStopping(monitor='val_loss', patience=1000, verbose=1), # lr_callback,
                                    ModelCheckpoint(f'models/{mod_name}/weights.h5', monitor='val_loss', verbose=True, save_best_only=True) ],
                            epochs=5000,
                            validation_split = 0.25,
                            batch_size=2**14,
                            #batch_size=1000000,
                            verbose=2
                           )

    my_model.load_weights(f'models/{mod_name}/weights.h5')
    my_model.save(f'models/{mod_name}')

    np.save(f'models/{mod_name}/history.npy', history.history)
