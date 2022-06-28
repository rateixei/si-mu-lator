import numpy as np
import argparse
import sys
import glob 
import datatools

parser = argparse.ArgumentParser(description='Data preparation for algorithms')

parser.add_argument('-f', '--files', dest='files', type=str, required=True,
                    help='Files')
parser.add_argument('-s', '--split-train-test', dest='split', type=float, default=0.0,
                    help='Split train and test datasets')

args = parser.parse_args()

file_names = glob.glob(args.files)

if len(file_names) < 1:
    print("Couldn't find any files from expression", args.files)
    sys.exit()

data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(file_names, max_files=500, sort_by='z')

print('Data keys/entries:', data.keys(), len(data[list(data.keys())[0]]))
print('Data matrix shape:', dmat.shape)
print('Data matrix keys:', sig_keys)
print('Y shape:', Y.shape)
print('Y_mu shape:', Y_mu.shape)
print('Y_hit shape:', Y_hit.shape)
