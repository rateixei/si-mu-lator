#!/bin/bash

SIM="/Data/ML/si-mu-lator/simulation_data"
CARD="atlas_nsw_pad_z0_normalRate"
DATA_LOC="${SIM}/${CARD}_bkgr_1/*.h5"
#SING_IMG=/Data/images/slac-ml@20211101.0.sif
SING_IMG=/Data/images/hls4ml_sandbox
DET="../cards/${CARD}.yml"


singularity exec -B /Data ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification" --detector ${DET}

# TCN with padded matrix for classification 
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification"

# TCN with detector matrix for classification
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification" --detector ${DET}

# TCN with padded matrix for classification and X regression
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "regression" --lambda 1  # --detector ${DET}

# LSTM with detector matrix for classification
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "lstm" --task "classification" --detector ${DET}
