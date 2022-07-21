#!/bin/bash
IMG_NAME_TCN=muon_hls4ml_05072022.sif
IMG_NAME_RNN=rnn_hls_keras_rnn_staticswitch_Apr25.sif

SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/${IMG_NAME_RNN}

fra_width=( 2 4 6 8 10 12 14 16 18 20 )
int_width=( 2 4 6 8 10 12 14 )
r_factor=( 1 5 10 50 100 )
stat=1 #default==1
strat="Resource" #--strategy

# tot_width=( 20 )
# int_width=( 6 )
# r_factor=( 1 )

MOD_LOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/algorithms/models/

# MOD=tcn_BatchNormFalse_MaskingFalse_06072022_05.50.29_normalMatrix
# MOD=tcn_BatchNormFalse_MaskingFalse_08072022_02.28.45_normalMatrix #variance scaling
MOD=lstm_BatchNormFalse_MaskingFalse_05072022_08.14.31_detMatrix
# MOD=lstm_BatchNormFalse_MaskingFalse_12072022_01.55.48_normalMatrix
# MOD=lstm_BatchNormFalse_MaskingFalse_13072022_03.25.21_normalMatrix

DATALOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/hls4ml/
# DATA="${DATALOC}/X_test_10000_padMat.npy,${DATALOC}/Y_test_10000_padMat.npy"
# DATA="${DATALOC}/X_test_10000_padMat_noSig.npy,${DATALOC}/Y_test_10000_padMat_noSig.npy"
DATA="${DATALOC}/X_test_10000_detMat_atlas_mm_vmm.npy,${DATALOC}/Y_test_10000_detMat_atlas_mm_vmm.npy"

OUTD=/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/hls/

for fw in "${fra_width[@]}"
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
singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw}  -r ${rf} --vivado --static ${stat} --strategy ${strat}  --lut
EOF
            
        done
    done
done
