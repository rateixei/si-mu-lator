#!/bin/bash

SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif

singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python Evaluate-Performance-Reg.py

# singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python Evaluate-Performance-Class.py

