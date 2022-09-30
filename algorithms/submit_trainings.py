import os
import sys
from copy import deepcopy

SIM      = "/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/20220912/SIG_atlas_nsw_pad_z0_xya/"
#SIM      = "/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/20220912/NoNoise_SIG_atlas_nsw_pad_z0_xya/"
DATA_LOC = f"{SIM}/TRAIN/*.h5"
SING_IMG = "/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/muon_qkeras.sif"

DCARD    = f" --detmat {SIM}/atlas_nsw_pad_z0.yml "
TEST     = False
LOCAL    = False

configs = [

    {'name': 'cbnorm',
     'prefix': '',
     'vals': [ 
#              "--conv-batchnorm  --input-batchnorm",
              # "--conv-batchnorm",
              " " 
              ]
    },
    
    {'name': 'dbnorm',
     'prefix': '',
     'vals': []
    },
    
    {'name': 'bias',
     'prefix': '',
     'vals': []
    },
    
    {'name': 'penalty',
     'prefix': '',
     'vals': []
    },
    
    {'name': 'penalty_type',
     'prefix': '--pen-type',
     'vals': [0]
    },
    
    {'name': 'penalty_type',
     'prefix': '--pen-type',
     'vals': [0]
    },
    
    {'name': 'conv_layers',
     'prefix': '--clayers',
     'vals': [#"7,1,1,0:5,3,1,0",
              #"7,1,1,0:5,4,1,0",
              #"7,1,1,0:5,5,1,0",
               #"7,1,1,0:10,3,1,0",
               #"7,1,1,0:10,4,1,0",
               #"7,1,1,0:10,5,1,0",
               #"7,1,1,0:20,3,1,0",
               #"7,1,1,0:20,4,1,0",
               #"7,1,1,0:20,5,1,0",
                "5,3,1,0",
                "5,4,1,0",
                "5,5,1,0",
               # "10,3,1,0",
               # "10,4,1,0",
               # "10,5,1,0",
               # "15,3,1,0",
               # "15,4,1,0",
               # "15,5,1,0",
               # "20,3,1,0",
               # "20,4,1,0",
               # "20,5,1,0",
               # "5,3,1,0:5,3,1,0",
               # "5,4,1,0:5,4,1,0",
               # "5,5,1,0:5,5,1,0",
               #"7,1,1,0:25,3,1,0",
               #"7,1,1,0:25,4,1,0",
               #"7,1,1,0:25,5,1,0",
               #"7,1,1,0:30,3,1,0",
               #"7,1,1,0:30,4,1,0",
               #"7,1,1,0:30,5,1,0",
               # "25,3,1,0",
               # "25,4,1,0",
               # "25,5,1,0",
               # "30,3,1,0",
               # "30,4,1,0",
               # "30,5,1,0",
                "7,4,4,0:5,3,1,0",
                "7,4,1,0:5,4,1,0",
                "7,4,4,0:5,5,1,0",
               # "7,4,4,0:10,3,1,0",
               # "7,4,4,0:10,4,1,0",
               # "7,4,4,0:10,5,1,0",
               # "7,4,4,0:20,3,1,0",
               # "7,4,4,0:20,4,1,0",
               # "7,4,4,0:20,5,1,0",
               # "7,4,4,0:25,3,1,0",
               # "7,4,4,0:25,4,1,0",
               # "7,4,4,0:25,5,1,0",
               # "7,4,4,0:30,3,1,0",
               # "7,4,4,0:30,4,1,0",
               # "7,4,4,0:30,5,1,0",
               #"7,1,1,0:7,6,6,0",
               #"7,1,1,0:7,5,5,0",
               #"7,1,1,0:7,4,4,0",
               #"7,1,1,0:7,3,3,0",
               #"7,1,1,0:7,2,2,0",
                "5,3,1,0:6,3,1,0",
                "7,3,1,0:5,3,1,0",
                "5,3,1,0:5,3,1,0",
                "5,4,1,0:5,4,1,0",
                "4,3,1,0:4,3,1,0",
                "5,3,1,0:5,3,3,0",
                "5,4,1,0:5,4,3,0",
               "4,3,1,0:4,3,3,0" #this one
              ]
    },
    
    {'name': 'dense_layers',
     'prefix': '--dlayers',
     'vals': [
         "none", 
         # "10", 
         # "20"
     ]
    },

    {'name': 'pooling',
     'prefix': '',
     'vals': ['--flatten']
    },
    
    {'name': 'do_qkeras',
     'prefix': '',
     'vals': []
    },
    
    {'name': 'qkeras_bits',
     'prefix': '--q-bits',
     'vals': []
    },
    
    {'name': 'qkeras_ibits',
     'prefix': '--q-ibits',
     'vals': []
    },
    
    {'name': 'l1_regularization',
     'prefix': '--l1reg',
     'vals': []
    },
    
    {'name': 'learning_rate',
     'prefix': '--lrate',
     #'vals': ["0.0005", "0.001", "0.005"]
     'vals': ["0.001"]
    },
    
    {'name': 'reference_trainiing',
     'prefix': '',
     #'vals': ["--bigger-model /sdf/home/r/rafaeltl/home/Muon/21062022/si-mu-lator/algorithms/withnoise_quantile_models/MyTCN_CL20.4.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"]
     'vals': ["--bigger-model /sdf/home/r/rafaeltl/home/Muon/21062022/si-mu-lator/algorithms/withnoise_quantile_models/MyTCN_CL25.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"]
    }
]

n_trials = 0
options = []
for cc in configs:
    if n_trials == 0:
        n_trials = len(cc['vals'])
        options = [ cc['prefix'] + ' ' + str(vv) for vv in cc['vals'] ]
    elif len(cc['vals']) > 0:
        n_trials *= len(cc['vals'])
        new_options = []
        for op in options:
            for vv in cc['vals']:
                new_options.append( op + ' ' + cc['prefix'] + ' ' + str(vv) )
        options = deepcopy(new_options)
    
print(n_trials, len(options))

while True:
    input_val = input(f"Accept {n_trials} jobs with dataset in {DATA_LOC}? [Y/n]\n")
    if input_val == 'Y':
        print('Accepted! Will submit jobs...')
        break
    elif input_val == 'n' or input_val == 'N':
        print('Not accepting jobs, bye...')
        sys.exit()
    else:
        print('Option not recognized, Y (accept) or n (not accept, quit)?')

        
        
command = f'python train.py -f "{DATA_LOC}"'

sbatch = '''sbatch --partition=atlas \
                   --job-name=JNAME --output=out/JNAME_o.txt \
                   --error=err/JNAME_e.txt --ntasks=1 \
                   --gpus-per-task=1 \
                   --cpus-per-task=4 --mem-per-cpu=4g \
                   --time=4:00:00 \
                   << EOF
#!/bin/sh
THIS_COMMAND
EOF
'''

if TEST:
    print("Test mode, only running on a few events...")
    command += ' --test'
        
for iop,op in enumerate(options):
    this_command = f'singularity exec --nv -B /sdf,/gpfs,/scratch {SING_IMG} ' + command + op + DCARD
    print(this_command)
        
    if LOCAL:
        print("Running job locally, will only run first one...")
        os.system(this_command)
        break
    else:
        jname = f'muon_nn_job_{iop}'
        sbatch_command = sbatch.replace('THIS_COMMAND', this_command).replace('JNAME', jname)
        os.system(sbatch_command)
        
