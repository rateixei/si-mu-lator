#!/bin/bash

SIM="/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/"
DATA_LOC="${SIM}/stgc/atlas_nsw_pad_z0_stgc20Max1_bkgr_1_CovAngle_TRAIN/*.h5"
# SING_IMG=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif
SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/muon_qkeras.sif

VALS_CBNORM=( "--conv-batchnorm" )
VALS_DBNORM=( "" )
# VALS_BIAS=( "" "--no-bias" )
VALS_BIAS=( "" )
VALS_LAMBDA=( 1 )
# VALS_PEN=("" "--no-penx --no-pena")
VALS_PEN=(" ")
# VALS_PTYPE=(0 1 2)
# VALS_PTYPE=(0 2)
VALS_PTYPE=( 0 )
# VALS_BKGPEN=("" "--bkg-pen")
VALS_BKGPEN=(" ")
# VALS_CLAYERS=( "64,3,1:64,3,1" "10,3,1:10,3,1" "64,3,1:10,3,1" "10,3,1:10,3,1:10,3,1" "1,1,1:10,3,1:10,3,1" )
#VALS_CLAYERS=( "5,1,1:5,3,1" "5,1,1:10,3,1" "10,1,1:10,3,1" "10,1,1:20,3,1" "20,1,1:20,3,1" "20,1,1:10,3,1" "20,1,1:5,3,1" "5,3,1:5,3,1" "10,3,1:10,3,1" "20,3,1:20,3,1" "20,3,1:10,3,1" "20,3,1:5,3,1" "5,3,1:20,3,1" "5,3,1:5,3,1:5,3,1" ) 
# VALS_CLAYERS=( "10,1,1" "7,3,1" "7,5,1" "7,10,1" "7,1,1:7,5,1" "7,3,1:7,5,1" "7,5,1:7,10,1" "7,1,1:7,3,1:7,5,1" "7,3,1:7,3,1:7,3,1" ) 
# VALS_CLAYERS=( "20,5,1" "5,5,1:5,5,1" "10,1,1:10,5,1"  )

# VALS_CLAYERS=( "7,3,1" "7,5,1" "10,1,1" "10,3,1" "10,5,1" "20,1,1" "20,3,1" "20,5,1" "7,1,1:7,3,1" "10,1,1:7,3,1" "20,3,1:10,3,1" )
# VALS_CLAYERS=( "5,3,1:5,3,1" "20,3,1" "30,3,1" "50,3,1" )
# VALS_CLAYERS=( "5,3,1:10,3,1" "5,3,1:5,3,1"  )
# VALS_CLAYERS=( "5,1,1::10,4,1" "10,4,1" "5,4,1:10,4,1" "5,4,1:5,4,1" )
# VALS_CLAYERS=( "10,3,1" "10,5,1" "15,3,1" "15,4,1" "15,5,1" "10,3,1:10,3,1" "10,3,1:20,3,1" "20,3,1:20,3,1" "10,4,1:10,4,1" "10,4,1:20,4,1" "20,4,1:20,4,1" )
# VALS_CLAYERS=( "20,4,1" "20,5,1" "25,3,1" "25,4,1" "25,5,1" "30,3,1" "30,4,1" "30,5,1")
VALS_CLAYERS=( "10,4,1:20,4,1" "7,3,1:20,3,1" "7,4,1:20,4,1" "7,5,1:20,5,1" "7,1,1:20,4,1" )
# VALS_DLAYERS=( "10" "20" "30" "40")
# VALS_DLAYERS=( "20" "20:10" "30" "30:20" "40" "40:20")
# VALS_DLAYERS=( "100:50" "100" "50:50" "50:25" "20:10" "none" )
VALS_DLAYERS=(  "none" )
# POOL=( "--avgpool" "--flatten" )
# POOL=( "--avgpool" )
POOL=( "--flatten"  )
# QK="--do-q"
QK=" "
# RESS=( " " "--residual" )
# RESS=(  "--residual" )
RESS=(  " " )

# QKBS=( 4 6 8 10 12 )
# QIKBS=( 0 2 4 6 )

QKBS=( 0 )
QIKBS=( 0 )


ijob=0

for cbnorm in "${VALS_CBNORM[@]}"
do
    for dbnorm in "${VALS_DBNORM[@]}"
    do
        for rbias in "${VALS_BIAS[@]}"
        do
            for lambda in "${VALS_LAMBDA[@]}"
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
                                        for RES in "${RESS[@]}"
                                        do
                                            for QKB in "${QKBS[@]}"
                                            do
                                                for QIKB in "${QIKBS[@]}"
                                                do
                                                    echo ${cbnorm} ${dbnorm} ${lambda} ${pen} ${ptype} ${bkgpen} ${clayers} ${dlayers} ${rbias} ${pool} ${QKB} ${QIKB}
                                                    # jname="TCN_${cbnorm}_${dbnorm}_${lambda}_${pen}_${ptype}_${bkgpen}_${clayers}_${dlayers}_${rbias}"
                                                    jname="TCNjob${ijob}"
                                                    ((ijob=ijob+1))
                                                    COMM="python tcn_train.py -f \"${DATA_LOC}\" ${cbnorm} ${dbnorm} --lambda ${lambda} ${pen} --pen-type ${ptype} ${bkgpen} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} ${RES} --q-bits ${QKB} --q-ibits ${QIKB} "
                                                    echo ${COMM}

                                                    # singularity exec --nv -B /sdf,/gpfs,/scratch ${SING_IMG} python tcn_train.py -f "${DATA_LOC}" ${cbnorm} ${dbnorm} --lambda ${lambda} ${pen} --pen-type ${ptype} ${bkgpen} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} ${RES} --q-bits ${QKB} --q-ibits ${QIKB}

                                                    # break 80931809

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
