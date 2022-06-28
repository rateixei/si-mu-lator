#!/bin/bash

SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnn_hls_keras_rnn_staticswitch_Apr25.sif
MOD=gru_BatchNormTrue_MaskingFalse_28062022_06.21.52

singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ../../algorithms/models/${MOD}/arch.json -w ../../algorithms/models/${MOD}/weights.h5