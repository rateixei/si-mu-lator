#!/bin/bash

DATA_LOC=""
SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif

singularity exec -B /sdf,/gpfs,/scratch ${SINGULARITY_IMAGE_PATH} python dataprep.py -f ${DATA_LOC} -s 0