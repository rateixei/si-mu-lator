import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json, load_model

from sklearn.metrics import accuracy_score, roc_auc_score

import argparse

parser = argparse.ArgumentParser(description='Do HLS Things')

parser.add_argument('-d', '--data', dest='data', type=str, 
                    default='../X_test_10000.npy,../Y_test_10000.npy',
                    help='data to evaluate')

parser.add_argument('--name', dest='mod_name', type=str, required=True, help='Model name')

parser.add_argument('-a', '--nn-architecture', dest='nn_arch', type=str, required=True,
                    help='neural network architecture to test')

parser.add_argument('-w', '--nn-weights', dest='nn_weights', type=str, required=True,
                    help='neural network weights to test')

parser.add_argument('-o', '--output-dir', dest='out_dir', type=str, default='./',
                    help='Output location')

parser.add_argument('-p', '--precision', dest='prec', type=str, default='16,6',
                    help='HLS precision')

parser.add_argument('-r', '--reuse', dest='reuse', type=int, default=1,
                    help='HLS precision')

parser.add_argument('--strategy', dest='strat', default='Latency', 
                    help='Strategy')
parser.add_argument('--vivado', dest='viv', action='store_true', default=False,
                    help='Do vivado things')
parser.add_argument('--lut', dest='new_table', action='store_true', default=False, 
                    help='Change LUT precision')
parser.add_argument('--static', dest='static', type=int, default=1, 
                    help='From static to non-static')


args = parser.parse_args()

mod_name = args.mod_name
mod_arch = args.nn_arch
mod_weig = args.nn_weights

out_loc_name = '/'.join( [args.out_dir, mod_name] )

try:
    os.mkdir( out_loc_name )
except FileExistsError:
    print('Directory already exists', out_loc_name)

try:
    os.mkdir( out_loc_name + '/plots' )
except FileExistsError:
    pass

print('--------- Loading network')
arch_json = open(mod_arch, 'r').read()
model = model_from_json(arch_json)
model.load_weights(mod_weig)

if model is None:
    print("Model is none, exiting")
    sys.exit()

print('--------- Loading data')
x_test = None
y_test = None

if ',' in args.data:
    x_test = np.load(args.data.split(',')[0])
    y_test = np.load(args.data.split(',')[1])
else:
    print("Specify as \"X_test_file_loc,Y_test_file_loc\"")
    sys.exit()

print('--------- Keras testing')
y_test_keras = model.predict(x_test, batch_size=2**10)
print(y_test_keras)
print(y_test)
print()

keras_auc = roc_auc_score(y_test, y_test_keras)

print(f"Keras-Accuracy: {keras_auc}")

import hls4ml

config = hls4ml.utils.config_from_keras_model(model, granularity='name', 
                                              default_precision=f'ap_fixed<{args.prec}>', 
                                              default_reuse_factor=args.reuse)


strat = args.strat
if args.reuse < 2:
    strat = 'Latency'

print('STATIC: ', args.static)
if args.static < 1:
    for layer in config['LayerName'].keys():
        print(layer)
        if 'top' in args.nn and 'layer1' in layer:
            print("Setting static")
            config['LayerName'][layer]['static'] = 'false' 

if "Resource" in  strat:
    config['Model']['Strategy'] = 'Resource'

if args.new_table:
    t_size = max( int(2**(1+float(args.prec.split(',')[1]))  ), 1024)

    for layer in config['LayerName'].keys():
        if 'softmax' in layer or 'sigmoid' in layer or 'relu' in layer or 'tanh' in layer:
            config['LayerName'][layer]['table_t'] = 'ap_fixed<18,1>'
            config['LayerName'][layer]['table_size'] = f'{t_size}'

print("-----------------------------------")
print("Configuration")
# plotting.print_dict(config)
print(config)
print("-----------------------------------")

print("\n-----------------------------------")
print('Starting Convert')
hls_model_name = '_'.join( ['model', args.prec.replace(',', '.'), 'reuse', str(args.reuse), strat ] )

if args.static < 1:
    hls_model_name += '_NonStatic'
else:
    hls_model_name += '_Static'

hls_model_name += '_NewImage'

if args.new_table:
    hls_model_name += '_NewBigTable'

hls_model_name += '_NewPart'

proj_loc = out_loc_name + f'/{hls_model_name}/myproject_prj/'
hls_model = hls4ml.converters.convert_from_keras_model(model, part='xcu250-figd2104-2-e',
                                                       hls_config=config,
                                                       output_dir=proj_loc)
print("-----------------------------------")

print("\n-----------------------------------")
print('Starting Compile')
hls_model.compile()
print('Done Compile')
print("-----------------------------------")


print("\n-----------------------------------")
print('Starting Predict')
print(x_test.shape)
x_test_cont = np.ascontiguousarray(x_test)
y_test_hls = hls_model.predict(x_test_cont)
hls_auc = roc_auc_score(y_test, y_test_hls)
print(f"HLS-Accuracy: {hls_auc}")
print('Done HLS predict')
print("-----------------------------------")

if args.viv:
    
    try:
        os.mkdir( out_loc_name + '/reports' )
    except FileExistsError:
        pass

    print("\n-----------------------------------")
    print('Loading Vivado 2019.2')
    import os
    os.environ['PATH'] = '/gpfs/slac/atlas/fs1/d/rafaeltl/public/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
    os.environ['LM_LICENSE_FILE'] = '2100@rdlic1:2100@rdlic2:2100@rdlic3'
    hls_model.build(csim=False, vsynth=True)
    from contextlib import redirect_stdout
    with open(out_loc_name+f'/reports/{hls_model_name}.txt', 'w') as f:
        with redirect_stdout(f):
            hls4ml.report.read_vivado_report(proj_loc)
    with open(out_loc_name+f'/reports/{hls_model_name}.txt', 'a') as f:
        f.write( f"KERAS_AUC {keras_auc}")
        f.write( "\n" )
        f.write( f"HLS_AUC {hls_auc}" )
        f.write( "\n" )
    print('Done Vivado 2019.2', out_loc_name+f'/reports/{hls_model_name}.txt')
    print("-----------------------------------")
