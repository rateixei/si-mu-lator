#!/bin/bash

OUTLOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/toy_sim/si-mu-lator/out_files/
DETC=atlas_mm
DET=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/toy_sim/si-mu-lator/cards/${DETC}.yml
NEV=1000
BKG=10000000
# BKG=0

for rnd in $(seq 70 89);
do
    echo $rnd
    jname=nomuon_${DETC}_evs${NEV}_bkg${BKG}_rnd${rnd}
    OUTF=${OUTLOC}/${jname}.h5
    job="./nomu_submit_slurm.sh ${OUTF} ${DET} ${NEV} ${BKG} ${rnd}"
    sbatch --partition=usatlas --job-name=${jname} --output=${jname}_o.txt --error=${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
done