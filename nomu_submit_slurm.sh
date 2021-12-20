#!/bin/bash

OUT=$1
DET=$2
NEV=$3
BKG=$4
RND=$5

SINGULARITY_IMAGE_PATH=/sdf/sw/ml/slac-ml/20200227.0/slac-jupyterlab@20200227.0.sif

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} python /gpfs/slac/atlas/fs1/u/rafaeltl/Muon/toy_sim/si-mu-lator/si_mu_late.py -o ${OUT} -d ${DET} -n ${NEV} -b ${BKG} -r ${RND}