import os
import sys
from datetime import datetime

njobs=200
#njobs=1
iseed=datetime.now().microsecond

## setup
#here_batch      = os.getcwd() + '/'
script_dir       = os.path.abspath( os.path.dirname( __file__ ) )
root_dir         = script_dir.replace('/batch_slac', '')
#detcard_name    = "atlas_mm_road"
#detcard_name    = "atlas_nsw_vmm"
#detcard_name    = "atlas_nsw_pad_z0"
#detcard_name    = "atlas_nsw_pad_z0_stgc20"
#detcard_name    = "atlas_nsw_pad_z0_stgc20Max1"
detcard_name    = "atlas_nsw_pad_z0_normalRate"
det_card        = root_dir+"/cards/"+detcard_name+".yml"

## events and noise
nevs            = 2000
bkg_rate        = 1
override_total_n_noise = -1 ## only accepted if not generating muon
#det_noise_dict = " --minhitsdet stgc 5 "
det_noise_dict = ""

## muon
generate_muon   = False
muon_x_range    = [-20.0,20.0]
muon_a_range    = [-3*1e-3, 3*1e-3]
do_cov_angle    = False
#muon_a_range    = []

import subprocess
result = subprocess.run(['hostname', '-d'], stdout=subprocess.PIPE)
domain = result.stdout.decode('utf-8').split('\n')[0]
if 'cern' in domain:
#    out_loc         = script_dir +f"/out_files/{detcard_name}_bkgr_{bkg_rate}"
    out_loc         =f"/Data/ML/si-mu-lator/simulation_data/{detcard_name}_bkgr_{bkg_rate}"
else:
    out_loc         = script_dir+f"/out_files/{detcard_name}_bkgr_{bkg_rate}"

if not generate_muon and override_total_n_noise > 0:
    out_loc += f'_override_{override_total_n_noise}'
else:
    out_loc += ''

if not os.path.exists(out_loc):
  os.makedirs(out_loc)
  print("The new directory is created!")

if 'slac' in domain:
    
    exec_file =  '''#!/bin/bash

SINGULARITY_IMAGE_PATH=/sdf/sw/ml/slac-ml/20200227.0/slac-jupyterlab@20200227.0.sif

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} python _ROOTDIR_/si_mu_late.py _OPTIONS_ -r $1 _det_noise_
'''
elif 'cern' in domain:
    exec_file =  '''#!/bin/bash                                                                                                                                      

SINGULARITY_IMAGE_PATH=/Data/images/slac-jupyterlab@20200227.0.sif                                                                                                        

singularity exec  -B /Data ${SINGULARITY_IMAGE_PATH} python _ROOTDIR_/si_mu_late.py _OPTIONS_ -r $1 _det_noise_
 '''
else:
        exec_file =  '''#!/bin/bash                                                                                                                                     

SINGULARITY_IMAGE_PATH=/your/sing/image                                                                                                              

singularity exec --nv  ${SINGULARITY_IMAGE_PATH} python _ROOTDIR_/si_mu_late.py _OPTIONS_ -r $1 _det_noise_       
'''

    
exec_file = exec_file.replace("_ROOTDIR_", root_dir)
exec_file = exec_file.replace("_det_noise_", det_noise_dict)
base_name = f"{detcard_name}.nevs_{nevs}.bkgr_{bkg_rate}"

if generate_muon:
    base_name = 'WithMuon.' + base_name
    if len(muon_x_range) > 0: 
        base_name += f'.mux.{muon_x_range[0]}.{muon_x_range[1]}'
    if do_cov_angle:
        base_name += '.CoverageAngle'
    else:
        if len(muon_a_range) > 0: 
            base_name += f'.mua.{muon_a_range[0]}.{muon_a_range[1]}'
else:
    base_name = 'NoMuon.' + base_name
    if override_total_n_noise > 0:
        base_name += f'_OverrideTotalNoise_{override_total_n_noise}' 

options  = f"-o {out_loc}/{base_name}.h5 "
options += f"-d {det_card} "
options += f"-n {nevs} "
options += f"-b {bkg_rate} "

if generate_muon:
    options += "-m "
    if len(muon_x_range) > 0: 
        options += f"-x {muon_x_range[0]} {muon_x_range[1]} "
    if do_cov_angle:
        options += " --coverage-angle "
    else:
        if len(muon_a_range) > 0: 
            options += f"-a {muon_a_range[0]} {muon_a_range[1]} "
else:
    if override_total_n_noise > 0:
        options += f" --override-n-noise-hits-per-event {override_total_n_noise} --minhits {override_total_n_noise} "

exec_file = exec_file.replace("_OPTIONS_", options)
exec_file_name = root_dir+base_name+".sh"

runf = open(exec_file_name, 'w')
runf.write(exec_file)
runf.close()
os.system(f"chmod a+rwx {exec_file_name}")

for ir in range(iseed, iseed+njobs):

    jname = base_name + f'.Rnd{ir}'
    torun = f"{exec_file_name} {ir} "

    batch_exec  = f"sbatch --partition=usatlas --job-name={jname} "
    batch_exec += f"--output={script_dir}/logs/{jname}_o.txt "
    batch_exec += f"--error={script_dir}/logs/{jname}_e.txt "
    batch_exec += f"--ntasks=1 --cpus-per-task=12 --mem-per-cpu=1g "
    batch_exec += f"--time=10:00:00 {torun}"

    print(batch_exec)
    # break
    if 'slac' in domain:
        os.system(batch_exec)
    elif 'cern' in domain:
        print ("#INFO: using cern resources, no sbatch, so running each job sequentially")
        os.system(torun)
    else:
        print ("#WARNING: Not a recognized environment, doing nothing")
    # break


