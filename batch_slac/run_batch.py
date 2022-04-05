import os
import sys
from datetime import datetime

njobs=100
iseed=datetime.now().microsecond

here_batch      = os.getcwd() + '/'
here            = here_batch.replace('/batch_slac', '')
detcard_name    = "atlas_mm_lm1_OneHit_BC"
det_card        = here+"/cards/"+detcard_name+".yml"
out_loc         = here_batch+"/out_files/"
nevs            = 1000
## Bkg rate is 25 kHz/cm2
## ROI area is 4.5*426.7 mm2 = 0.45*42.67 cm2 = 19.2 cm2
## Bkg rate in ROI is 475 kHz 
bkg_rate        = 50*47.5*1e3
generate_muon   = True
# muon_x_range    = [-100, -90]
# muon_a_range    = [3.141/2, 3.142/2]
muon_a_range    = []
muon_x_range    = []
# muon_a_range    = []
exec_file =  '''#!/bin/bash

SINGULARITY_IMAGE_PATH=/sdf/sw/ml/slac-ml/20200227.0/slac-jupyterlab@20200227.0.sif

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} python _HERE_/si_mu_late.py _OPTIONS_ -r $1

'''

exec_file = exec_file.replace("_HERE_", here)
base_name = f"{detcard_name}.nevs_{nevs}.bkgr_{bkg_rate}"

if generate_muon:
    base_name = 'WithMuon.' + base_name
    if len(muon_x_range) > 0: base_name += f'.mux.{muon_x_range[0]}.{muon_x_range[1]}'
    if len(muon_a_range) > 0: base_name += f'.mua.{muon_a_range[0]}.{muon_a_range[1]}'
else:
    base_name = 'NoMuon.' + base_name

options  = f"-o {out_loc}/{base_name}.h5 "
options += f"-d {det_card} "
options += f"-n {nevs} "
options += f"-b {bkg_rate} "

if generate_muon:
    options += "-m "
    if len(muon_x_range) > 0: options += f"-x {muon_x_range[0]} {muon_x_range[1]} "
    if len(muon_a_range) > 0: options += f"-a {muon_a_range[0]} {muon_a_range[1]} "

exec_file = exec_file.replace("_OPTIONS_", options)
exec_file_name = here+base_name+".sh"

runf = open(exec_file_name, 'w')
runf.write(exec_file)
runf.close()
os.system(f"chmod a+rwx {exec_file_name}")


for ir in range(iseed, iseed+njobs):

    jname = base_name + f'.Rnd{ir}'
    torun = f"{exec_file_name} {ir} "

    batch_exec  = f"sbatch --partition=usatlas --job-name={jname} "
    batch_exec += f"--output={here_batch}/logs/{jname}_o.txt "
    batch_exec += f"--error={here_batch}/logs/{jname}_e.txt "
    batch_exec += f"--ntasks=1 --cpus-per-task=8 --mem-per-cpu=3g "
    batch_exec += f"--time=2:00:00 {torun}"

    print(batch_exec)
    os.system(batch_exec)
