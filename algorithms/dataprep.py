import numpy as np
import argparse
import sys
import glob 
import datatools

parser = argparse.ArgumentParser(description='Data preparation for algorithms')

parser.add_argument('-f', '--files', dest='files', type=str, required=True,
                    help='Files')
parser.add_argument('-s', '--split-train-test', dest='split', type=float, default=0.0
                    help='Split train and test datasets')

args = parser.parse_args()

file_names = glob.glob(args.files)

if len(file_names) < 1:
    print("Couldn't find any files from expression", args.files)
    sys.exit()

data, dmat, Y, Y_mu, Y_hit, sig_keys = dataprep.make_data_matrix(all_files, max_files=500, sort_by='z')

print('Data shape:', data.shape)
print('Data matrix shape:', dmat.shape)
print('Data matrix keys:', sig_keys)
print('Y shape:', Y.shape)
print('Y_mu shape:', Y_mu.shape)
print('Y_hit shape:', Y_hit.shape)
