#!/bin/bash
# IMG_NAME_TCN=muon_hls4ml_05072022.sif
IMG_NAME_TCN=muon_hls4ml_qkeras.sif
IMG_NAME_RNN=rnn_hls_keras_rnn_staticswitch_Apr25.sif

SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/${IMG_NAME_TCN}

fra_width=( 2 4 6 8 10 12 14 16 18 20 )
int_width=( 0 1 2 3 4 6 )
r_factor=( 1 2 5 10 50 )

stat=1 #default==1
strat=( "Resource" "Latency"  )
# strat=( "Latency" )
# strat=( "Resource" )

# fra_width=( 12 )
# int_width=( 6 )
# r_factor=( 1 )

MOD_LOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/algorithms/models/

# MOD="MyTCN_CL5.3.1.0..5.3.3.0_DLnone_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL20.4.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL7.4.4.0..5.3.1.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL4.3.1.0..4.3.3.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
MOD="MyTCN_CL7.3.1.0..5.3.1.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"

echo "HERE"

DATALOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/hls4ml/
DATA="${DATALOC}/X_test_50000_detMat_atlas_nsw_pad_z0.npy,${DATALOC}/Y_test_50000_detMat_atlas_nsw_pad_z0.npy"

OUTD=/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/hls/

for fw in "${fra_width[@]}"
do
    for iw in "${int_width[@]}"
    do
        for rf in "${r_factor[@]}"
        do
            for st in "${strat[@]}"
            do

                jname=${MOD}_${tw}_${iw}_${rf}_${st}
            
                # singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw}  -r ${rf} --vivado --static ${stat} --strategy ${strat}  --lut

                # break 91382098

                sbatch --partition=usatlas \
                    --job-name=${jname} --output=out/${jname}_o.txt \
                    --error=err/${jname}_e.txt --ntasks=1 \
                    --cpus-per-task=4 --mem-per-cpu=10g \
                    --time=1:00:00 \
                    << EOF
#!/bin/sh
singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw}  -r ${rf} --vivado --static ${stat} --strategy ${st}  --lut
EOF
            
            done
        done
    done
done
