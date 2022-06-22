#!/bin/bash

SIM="/sdf/home/r/rafaeltl/home/Muon/21062022/si-mu-lator/"
DATA_LOC="${SIM}/batch_slac/out_files/atlas_mm_vmm_bkgr_1/*.h5"
SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif

singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" -m "lstm"
