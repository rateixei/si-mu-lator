#!/bin/bash
#SING_IMG=/Data/images/rnn_hls_keras_rnn_staticswitch_Apr25_viv_RR #use for LSTM and GRU only. can be used also to build using image's vivado
SING_IMG=/Data/images/hls4ml_sandbox
SING_VIVADO_ONLY_IMG=/Data/images/hls4ml.0.6.0_vivado.sif # remember it has hls4ml=0.5.1 do not use it, only for vivado
#SING_IMG=/Data/images/muon_hls4ml_05072022.sif
#SING_IMG=/Data/images/rnn_hls_keras_rnn_staticswitch_Apr25_RR_hls4ml.0.6.0
#tot_width=( 12 14 16 18 )
#int_width=( 2 4 6 8  )
#r_factor=( 1 )

fra_width=( 8 )
int_width=( 12 )
r_factor=( 10 )

MOD_LOC=/afs/cern.ch/user/r/rrojas/public/ML/si-mu-lator/algorithms/models
# MOD=gru_BatchNormTrue_MaskingFalse_28062022_06.21.52
# MOD=lstm_BatchNormTrue_MaskingFalse_28062022_10.27.00
# MOD=lstm_BatchNormFalse_MaskingFalse_29062022_00.09.58
#MOD=gru_BatchNormFalse_MaskingFalse_11072022_16.56.28
#MOD=tcn_BatchNormFalse_MaskingFalse_classification_21072022_16.11.06_detMatrix
MOD=tcn_BatchNormFalse_MaskingFalse_classification_21072022_17.14.41_detMatrix

DATALOC=/afs/cern.ch/user/r/rrojas/public/ML/si-mu-lator/hls4ml/
DATA="${DATALOC}/X_test_10000.npy,${DATALOC}/Y_test_10000.npy"

OUTD=/Data/Muon_hls/
STRAT="Latency"
STATIC=1
export SINGULARITYENV_CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu
export SINGULARITYENV_C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu
export SINGULARITYENV_LM_LICENSE_FILE=2112@licenxilinx
function usage()
{
    echo "usage: exec_do_jls_things.sh [-s,v,a]"
    echo "    -s synchronize between umassminipc02 and umasscastor1n1"
    echo "    -v run vivado_hls csim, synth and vivado cosim, vsynth, export"
    echo "    -a use vivado accelerator, create project and produce the bitfile"
    echo "    -h this help"
    exit 0
}
while getopts vash opt
do
    case "${opt}" in
        v) VIVADO=1;;
        a) ACCELERATOR=1;;
        s) SYNC=1;;
        h|*) usage;;
    esac
done
    

for fw in "${fra_width[@]}"
do
    for iw in "${int_width[@]}"
    do
        for rf in "${r_factor[@]}"
        do
            jname=${MOD}_${tw}_${iw}_${rf}
            

#!/bin/sh
if [[ ! -z $SYNC ]];
then
    if [ $(hostname) == "umasscastor1n1" ]; then
        echo "INFO: here will first sync the directory from umassminipc02 to local computer (umasscastor1n1):"
        echo "     ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static"
        rsync -av --progress umassminipc02:${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static ${OUTD}${MOD}
    elif [ $(hostname) == "umassminipc02" ]; then
        echo "INFO: here will first sync the directory from umasscastor1n1 to local computer (umassminipc02):"
        echo "     ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static"
        rsync -av --progress umasscastor1n1:${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static ${OUTD}${MOD}
    else
        echo "WARNING: the hostname does not match any of the recorded pc names"
    fi
fi;

if [[ ! -z $VIVADO ]];
then
    echo "INFO: here will do vivado things only using:"
    echo "     $SING_IMG"
    echo "INFO: on the directory:"
    echo "     ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static"

    export SINGULARITYENV_PREPEND_PATH=/opt/Xilinx/Vivado/Vivado/2019.2/bin/
    singularity exec -C -e  -H /home/rrojas/hls4ml_home -B /Data,/afs   ${SING_VIVADO_ONLY_IMG} /bin/bash -c "cd ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static/myproject_prj;"' vivado_hls -f build_prj.tcl "csim=1 synth=1 vsynth=1 cosim=1 export=1"'
    #"cd ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static/myproject_prj;"
    #' vivado_hls -f build_prj.tcl "csim=1 synth=0 cosim=0 export=0"'
    
fi;

if [[ ! -z $ACCELERATOR ]];
then
    echo "INFO: here will do vivado accelerator only using:"
    echo "     $SING_IMG"
    echo "INFO: on the directory:"
    echo "     ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static"
    
    export SINGULARITYENV_PREPEND_PATH=/opt/Xilinx/Vivado/Vivado/2019.2/bin/
    singularity exec -C -e  -H /home/rrojas/hls4ml_home -B /Data,/afs   ${SING_VIVADO_ONLY_IMG} /bin/bash -c "cd ${OUTD}${MOD}/model_$(( ${fw} + ${iw})).${iw}_reuse_${rf}_${STRAT}_Static/myproject_prj;"' vivado -mode batch -source  project.tcl design.tcl'


fi;

if [[ (-z $ACCELERATOR) && (-z $VIVADO) && (-z $SYNC)]]
then
    #if no argument do the initial plan
    singularity exec -C -e  -H /home/rrojas/hls4ml_home -B /Data,/afs,/tools   ${SING_IMG}  python -d /afs/cern.ch/user/r/rrojas/public/ML/si-mu-lator/hls4ml/batch/do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw} -r ${rf} --strategy ${STRAT} --static ${STATIC} --vivado #-t #-b #-t
    
fi;
            
        done
    done
done
