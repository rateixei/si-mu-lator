export SINGULARITY_IMAGE_PATH=/sdf/sw/ml/slac-ml/20200227.0/slac-jupyterlab@20200227.0.sif

singularity shell --nv -B /sdf,/gpfs,/scratch ${SINGULARITY_IMAGE_PATH}
