#!/bin/bash

if [ $(hostname -d) == "slac.stanford.edu" ]; then
    SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif

    singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python make_small_dataset.py
fi;

if [ $(hostname -d) == "dyndns.cern.ch" ]; then
    SING_IMG=/Data/images/slac-ml@20211101.0.sif

    singularity exec -B /Data ${SING_IMG} python make_small_dataset.py
fi;
