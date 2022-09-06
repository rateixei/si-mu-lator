# bash

INSTALLATION_PATH="/srv/conda/envs/notebook/lib/python3.7/site-packages"

function usage()
{
    echo "patch_hls4ml.sh"
    echo "patch script to adapt few hls4ml files for ML project needs"
    echo "usage: patch_hls4ml.sh [-i,d]"
    echo "    -i Installation directory of hls4ml (default: ${INSTALLATION_PATH})"
    echo "    -d dryrun, do not excecute, just print"
    echo "    -h this help"
    exit 0
}
while getopts i:dh opt
do
    case "${opt}" in
        i) INSTALLATION_PATH=${OPTARG};;
        d) DRYRUN=1;;
        h|*) usage;;
    esac
done


N_PATCHES=3
PATCH_DIR="hls4ml"
P_FILES=( "build_prj.tcl"
	 "axi_stream_design.tcl"
	 "nnet_activation_stream.h" )
	 
P_PATHS=( "hls4ml/templates/vivado"
	  "hls4ml/templates/vivado_accelerator/zcu102/tcl_scripts"
	  "hls4ml/templates/vivado/nnet_utils" )

for (( i=0; i<N_PATCHES; i++)); do
    F_tb_patch="${INSTALLATION_PATH}/${P_PATHS[i]}/${P_FILES[i]}"
    patch="${PATCH_DIR}/${P_FILES[i]}.patch"
    echo "INFO#: going to patch $F_tb_patch with ${patch}"
    cmd="patch $F_tb_patch ${patch}"
    if [ -z $DRYRUN ]; then 
	$cmd
    else
	echo dryrun: $cmd
    fi
done
    
