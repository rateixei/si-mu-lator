#!/bin/bash
SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnn_hls_keras_rnn_staticswitch_Apr25.sif

tot_width=( 8 10 12 14 16 18 20 22 24 26 28 30 )
int_width=( 2 4 6 8 10 12 )
r_factor=( 1 5 10 50 100 )

# tot_width=( 20 )
# int_width=( 6 )
# r_factor=( 1 )

MOD_LOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/algorithms/models/
MOD=gru_BatchNormTrue_MaskingFalse_28062022_06.21.52

DATALOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/hls4ml/
DATA="${DATALOC}/X_test_10000.npy,${DATALOC}/Y_test_10000.npy"

OUTD=/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/hls/

for tw in "${tot_width[@]}"
do
    for iw in "${int_width[@]}"
    do
        for rf in "${r_factor[@]}"
        do
            jname=${MOD}_${tw}_${iw}_${rf}
            
            sbatch --partition=usatlas \
                --job-name=${jname} --output=out/${jname}_o.txt \
                --error=err/${jname}_e.txt --ntasks=1 \
                --cpus-per-task=4 --mem-per-cpu=10g \
                --time=1:00:00 \
                << EOF
#!/bin/sh
singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} -p "${tw},${iw}" -r ${rf} --lut --vivado
EOF
            
        done
    done
done
