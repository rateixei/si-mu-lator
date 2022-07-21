#!/bin/bash

SIM="/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/"
DATA_LOC="${SIM}/20220628/atlas_mm_vmm_bkgr_1_TRAIN/*.h5"
SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif
DET="../cards/atlas_mm_vmm.yml"

# TCN with padded matrix for classification 
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification"

# TCN with detector matrix for classification
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification" --detector ${DET}

# TCN with padded matrix for classification and X regression
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "regression" --lambda 1  # --detector ${DET}

# LSTM with detector matrix for classification
# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "lstm" --task "classification" --detector ${DET}
