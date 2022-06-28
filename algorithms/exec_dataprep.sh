#!/bin/bash

DATA_LOC="/sdf/home/r/rafaeltl/home/Muon/21062022/si-mu-lator/batch_slac/out_files/atlas_mm_vmm_bkgr_1/WithMuon.atlas_mm_vmm.nevs_1000.bkgr_1.mux.-2.0.2.0_Rnd116396.h5"
SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif

singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python dataprep.py -f "${DATA_LOC}" -s 0
