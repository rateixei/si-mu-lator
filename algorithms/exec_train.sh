#!/bin/bash

SIM="/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/20220912/SIG_atlas_nsw_pad_z0_xya/"
DATA_LOC="${SIM}/TRAIN/*.h5"
SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/muon_qkeras.sif

VALS_CBNORM=( "--conv-batchnorm  --input-batchnorm" " --conv-batchnorm " " ")
VALS_DBNORM=(  " " )
VALS_BIAS=( "" )
VALS_PEN=(" ")
VALS_PTYPE=( 0 )
VALS_BKGPEN=(" ")

VALS_CLAYERS=( "7,1,1,0:5,3,1,0" "7,1,1,0:5,4,1,0" "7,1,1,0:5,5,1,0" "7,1,1,0:10,3,1,0" "7,1,1,0:10,4,1,0" "7,1,1,0:10,5,1,0" "7,1,1,0:20,3,1,0" "7,1,1,0:20,4,1,0" "7,1,1,0:20,5,1,0" "5,3,1,0" "5,4,1,0" "5,5,1,0" "10,3,1,0" "10,4,1,0" "10,5,1,0"  "15,3,1,0" "15,4,1,0" "15,5,1,0" "20,3,1,0" "20,4,1,0" "20,5,1,0" "5,3,1,0:5,3,1,0" "5,4,1,0:5,4,1,0" "5,5,1,0:5,5,1,0" "7,1,1,0:25,3,1,0" "7,1,1,0:25,4,1,0" "7,1,1,0:25,5,1,0" "7,1,1,0:30,3,1,0" "7,1,1,0:30,4,1,0" "7,1,1,0:30,5,1,0" "25,3,1,0" "25,4,1,0" "25,5,1,0" "30,3,1,0" "30,4,1,0" "30,5,1,0" "7,4,4,0:5,3,1,0" "7,4,4,0:5,4,1,0" "7,4,4,0:5,5,1,0" "7,4,4,0:10,3,1,0" "7,4,4,0:10,4,1,0" "7,4,4,0:10,5,1,0" "7,4,4,0:20,3,1,0" "7,4,4,0:20,4,1,0" "7,4,4,0:20,5,1,0" "7,4,4,0:25,3,1,0" "7,4,4,0:25,4,1,0" "7,4,4,0:25,5,1,0" "7,4,4,0:30,3,1,0" "7,4,4,0:30,4,1,0" "7,4,4,0:30,5,1,0" "7,1,1,0:7,6,6,0" "7,1,1,0:7,5,5,0" "7,1,1,0:7,4,4,0" "7,1,1,0:7,3,3,0" "7,1,1,0:7,2,2,0" "5,3,1,0:6,3,1,0" "7,3,1,0:5,3,1,0" "5,3,1,0:5,3,1,0" "5,4,1,0:5,4,1,0" "4,3,1,0:4,3,1,0" "5,3,1,0:5,3,3,0" "5,4,1,0:5,4,3,0" "4,3,1,0:4,3,3,0" )

VALS_DLAYERS=(  "none" "10" "20" "50" )

POOL=( "--flatten"  )

# QK="--do-q"
QK=" "

# QKBS=( 4 6 8 10 12 )
# QIKBS=( 0 2 4 6 )
QKBS=( 0 )
QIKBS=( 0 )

L1REG=" "
# L1REG="--l1reg 0.001"

LRATE=( "0.0005" "0.001" "0.005" )

ijob=0

# DCARD=" "
DCARD=" --detmat ${SIM}/atlas_nsw_pad_z0.yml "

for cbnorm in "${VALS_CBNORM[@]}"
do
    for dbnorm in "${VALS_DBNORM[@]}"
    do
        for rbias in "${VALS_BIAS[@]}"
        do
                for pen in "${VALS_PEN[@]}"
                do
                    for ptype in "${VALS_PTYPE[@]}"
                    do
                        for bkgpen in "${VALS_BKGPEN[@]}"
                        do
                            for clayers in "${VALS_CLAYERS[@]}"
                            do
                                for dlayers in "${VALS_DLAYERS[@]}"
                                do
                                    for pool in "${POOL[@]}"
                                    do
                                            for QKB in "${QKBS[@]}"
                                            do
                                                for QIKB in "${QIKBS[@]}"
                                                do
                                                    echo ${cbnorm} ${dbnorm} ${lambda} ${pen} ${ptype} ${bkgpen} ${clayers} ${dlayers} ${rbias} ${pool} ${QKB} ${QIKB}
                                                    # jname="TCN_${cbnorm}_${dbnorm}_${lambda}_${pen}_${ptype}_${bkgpen}_${clayers}_${dlayers}_${rbias}"
                                                    jname="TCNjob${ijob}"
                                                    ((ijob=ijob+1))
                                                    COMM="python train.py -f \"${DATA_LOC}\" ${cbnorm} ${dbnorm} --lambda ${lambda} ${pen} --pen-type ${ptype} ${bkgpen} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} ${RES} --q-bits ${QKB} --q-ibits ${QIKB} ${L1REG} ${DCARD}"
                                                    echo ${COMM}

                                                    singularity exec --nv -B /sdf,/gpfs,/scratch ${SING_IMG} python train.py -f "${DATA_LOC}" ${cbnorm} ${dbnorm} ${pen} --pen-type ${ptype} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} --q-bits ${QKB} --q-ibits ${QIKB} ${L1REG} ${DCARD}

                                                    break 80931809

                                                    sbatch --partition=atlas \
                                                        --job-name=${jname} --output=out/${jname}_o.txt \
                                                         --error=err/${jname}_e.txt --ntasks=1 \
                                                        --gpus-per-task=1 \
                                                        --cpus-per-task=4 --mem-per-cpu=4g \
                                                        --time=5:00:00 \
                                                    << EOF
#!/bin/sh
singularity exec --nv -B /sdf,/gpfs,/scratch ${SING_IMG} ${COMM}
EOF

                                                # break 1000000
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
