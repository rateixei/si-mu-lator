#!/bin/bash
# IMG_NAME_TCN=muon_hls4ml_05072022.sif
IMG_NAME_TCN=muon_hls4ml_qkeras.sif
IMG_NAME_RNN=rnn_hls_keras_rnn_staticswitch_Apr25.sif

SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/${IMG_NAME_TCN}

fra_width=( 2 4 6 8 10 12 14 16 18 20 )
int_width=( 0 1 2 3 4 6 )
r_factor=( 1 2 5 10 )

stat=1 #default==1
strat=( "Resource" "Latency"  )
# strat=( "Latency" )
# strat=( "Resource" )

# fra_width=( 12 )
# int_width=( 6 )
# r_factor=( 1 )

MOD_LOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/algorithms/models_sparse/

# MOD=tcn_BatchNormFalse_MaskingFalse_06072022_05.50.29_normalMatrix
# MOD=tcn_BatchNormFalse_MaskingFalse_08072022_02.28.45_normalMatrix #variance scaling
# MOD=lstm_BatchNormFalse_MaskingFalse_05072022_08.14.31_detMatrix
# MOD=lstm_BatchNormFalse_MaskingFalse_12072022_01.55.48_normalMatrix
# MOD=lstm_BatchNormFalse_MaskingFalse_13072022_03.25.21_normalMatrix
# MOD="MyTCN_20,1,1:20,3,1_25_CBNormTrue_DBNormFalse_ll1_ptype0_penXFalse_penAFalse_bkgPenTrue_regBiasFalse"
# MOD="MyTCN_5,1,1:10,3,1_5_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue"
# MOD="MyTCN_20,1,1:20,3,1_25_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenTrue_regBiasFalse"
# MOD="MyTCN_5,1,1:10,3,1_5_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue"
# MOD="MyTCN_5,1,1:10,3,1_5_CBNormFalse_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue"
# MOD="MyTCN_5,1,1:10,3,1_5_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue"
# MOD="MyTCN_20,1,1:20,3,1_10_CBNormTrue_DBNormFalse_ll1_ptype2_penXTrue_penATrue_bkgPenTrue_regBiasTrue"
# MOD="MyTCN_10,3,1:10,3,1_30_CBNormTrue_DBNormFalse_ll1_ptype2_penXTrue_penATrue_bkgPenTrue_regBiasTrue_AvgPool"
# MOD="MyTCN_20,3,1:5,3,1_20_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenTrue_regBiasFalse_Flatten"
# MOD="MyTCN_10,3,1:10,3,1_30_CBNormTrue_DBNormFalse_ll1_ptype2_penXTrue_penATrue_bkgPenTrue_regBiasTrue_AvgPool"
# MOD="MyTCN_20,3,1:20,3,1_40_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenTrue_regBiasTrue_Flatten"
# MOD="QKeras_10_4_MyTCN_30,3,1_30_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_AvgPool_ResNet"
# MOD="MyTCN_30,3,1:40,3,1_100:50_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_5,3,1:5,3,1_50:50_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_10,5,1_none_CBNormTrue_DBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_CL7.1.1.0..10.3.1.0_DL10..50_CBNormTrue_DBNormFalse_IBNormFalse_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_CL10.5.1.0_DLnone_CBNormTrue_DBNormFalse_IBNormTrue_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_CL5.3.1.0..5.3.1.0_DLnone_CBNormTrue_DBNormFalse_IBNormTrue_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten"
# MOD="MyTCN_CL5.3.1.0..5.3.1.0_DLnone_CBNormTrue_DBNormFalse_IBNormTrue_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten_0.5"
#MOD="MyTCN_CL5.3.1.0..5.3.1.0_DLnone_CBNormTrue_DBNormFalse_IBNormTrue_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten_HAND1_6"
MOD="MyTCN_CL5.3.1.0..5.3.1.0_DLnone_CBNormTrue_DBNormFalse_IBNormTrue_ll1_ptype0_penXTrue_penATrue_bkgPenFalse_regBiasTrue_Flatten_HAND0_8"

echo "HERE"

DATALOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/hls4ml/
#DATA="${DATALOC}/X_test_10000_padMat.npy,${DATALOC}/Y_test_10000_padMat.npy"
DATA="${DATALOC}/X_test_100000_padMat_noSig_atlas_nsw_pad_z0_stgc20Max1_bkgr_1_CovAngle_TRAIN.npy,${DATALOC}/Y_test_100000_padMat_noSig_atlas_nsw_pad_z0_stgc20Max1_bkgr_1_CovAngle_TRAIN.npy"
# DATA="${DATALOC}/X_test_10000_padMat_noSig.npy,${DATALOC}/Y_test_10000_padMat_noSig.npy"
# DATA="${DATALOC}/X_test_10000_detMat_atlas_mm_vmm.npy,${DATALOC}/Y_test_10000_detMat_atlas_mm_vmm.npy"

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
