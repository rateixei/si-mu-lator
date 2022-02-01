import os
import sys

njobs=20
iseed=8181

here            = os.getcwd() + '/'
detcard_name    = "atlas_mm"
det_card        = here+"/cards/"+detcard_name+".yml"
out_loc         = here+"/out_files/"
nevs            = 1000
bkg_rate        = 10000000
generate_muon   = True
muon_x_width    = [-0.015, 0.015]
exec_file =  '''#!/bin/bash

SINGULARITY_IMAGE_PATH=/sdf/sw/ml/slac-ml/20200227.0/slac-jupyterlab@20200227.0.sif

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} python _HERE_/si_mu_late.py _OPTIONS_ -r $1

'''

exec_file = exec_file.replace("_HERE", here)
base_name = f"{detcard_name}.nevs_{nevs}.bkgr_{bkg_rate}"

if generate_muon:
    base_name = 'WithMuon.' + base_name
    base_name += f'.mux.{muon_x_width[0]}.{muon_x_width[1]}'
else:
    base_name = 'NoMuon.' + base_name

options  = f"-o {out_loc}/{base_name}.h5 "
options += f"-d {det_card} "
options += f"-n {nevs} "
options += f"-b {bkg_rate} "

if generate_muon:
    options += "-m "
    options += f"-x {muon_x_width[0]} {muon_x_width[1]} "

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
    batch_exec += f"--output={here}/logs/{jname}_o.txt "
    batch_exec += f"--error={here}/logs/{jname}_e.txt "
    batch_exec += f"--ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g "
    batch_exec += f"--time=2:00:00 {torun}"

    print(batch_exec)
    os.system(batch_exec)
