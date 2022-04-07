import numpy as np
import argparse
from detmodel.detector import Detector

import multiprocessing
import tqdm
import time
import sys
import h5py

import pandas

my_configs = 0

parser = argparse.ArgumentParser(description='Si-MU-late')
    
# general and required
parser.add_argument('-d', '--detector', dest='detcard', type=str, required=True,
                    help='Detector card')
parser.add_argument('-n', '--nevents', dest='nevs', type=int, required=True,
                    help='Number of events')
parser.add_argument('--minhits', dest='min_n_hits', type=int, required=False, default=1,
                    help='Minimum number of hits per event over all detector')
parser.add_argument('--override-n-noise-hits-per-event', dest='override_n_noise_hits_per_event', type=int, required=False, default=-1,
                    help='Override noise setting, requires exact number of noise hits in an event')                    
parser.add_argument('-o', '--outfile', dest='outf', type=str, required=True,
                    help='Out h5 file')
    
# muon simulation
parser.add_argument('-m', '--addmuon', dest='ismu', action='store_true', default=False,
                    help='Simulate muon')
parser.add_argument('-x', '--muonx', nargs=2, metavar=('muxmin', 'muxmax'), 
                    default=(0,0), type=float, help='Generated muon X')
parser.add_argument('-y', '--muony', nargs=2, metavar=('muymin', 'muymax'), 
                    default=(0,0), type=float, help='Generated muon Y')
parser.add_argument('-a', '--muona', nargs=2, metavar=('muamin', 'muamax'), 
                    default=(0,0), type=float, help='Generated muon angle')
    
# background simulation
parser.add_argument('-b', '--bkgrate', dest='bkgr', type=float, default=0,
                       help='Background rate scale factor')

# other
parser.add_argument('-r', '--randomseed', dest='randseed', type=int, default=42,
                       help='Set random seed')
    
my_configs = parser.parse_args()
    
my_detector = Detector()
my_detector.read_card(my_configs.detcard)

def run_event(randseed):
    
    my_detector.reset_planes()
    
    ## muon
    mu_config = None

    np.random.seed(randseed)

    if my_configs.ismu:
        mu_x = 0
        if my_configs.muonx[0] < my_configs.muonx[1]:
            mu_x = np.random.uniform(low=my_configs.muonx[0], high=my_configs.muonx[1])
        
        mu_y = 0
        if my_configs.muony[0] < my_configs.muony[1]:
            mu_y = np.random.uniform(low=my_configs.muony[0], high=my_configs.muony[1])
            
        mu_a = 0
        if my_configs.muona[0] < my_configs.muona[1]:
            mu_a = np.random.uniform(low=my_configs.muona[0], high=my_configs.muona[1])
        
        my_detector.add_muon(mu_x=mu_x, mu_y=mu_y, mu_theta=mu_a, mu_phi=0, mu_time=0, randseed=randseed)
        mu_config = [mu_x, mu_y, mu_a, 0, 0]
    
    ## background
    if my_configs.bkgr > 0:
        my_detector.add_noise(my_configs.bkgr, override_n_noise_hits_per_event=override_n_noise_hits_per_event, randseed=randseed+1)
    
    ## signals
    sigs_keys = my_detector.get_signals(my_configs.min_n_hits)
    
    if sigs_keys is not None:
        return (sigs_keys[0], sigs_keys[1], mu_config)
    else:
        return None
    

def make_signal_matrix(res):
    
    evs = []
   
    max_sigs = []
    ncols = -10
    for iiev,iev in enumerate(res):
        if iev.get() is not None:
            max_sigs.append(iev.get()[0].shape[0])
            if iiev == 0:
                ncols = iev.get()[0].shape[1]
            else:
                assert ncols == iev.get()[0].shape[1]

    n_nonzero_sigs = len(max_sigs)
    max_sigs = np.max(max_sigs)
    
    out_matrix = -99 * np.ones( ( n_nonzero_sigs, max_sigs, ncols ) )
    key = []
    mu_configs = []
    iiev = 0
    for iev in res:
        if iev.get() is not None:
            this_res, this_key, muconf = iev.get()
            if len(key) == 0:
                key = this_key[:]
            this_shape = this_res.shape
            out_matrix[iiev][ :this_shape[0], : ] = this_res
            mu_configs.append(muconf)
            iiev += 1
    
    return(out_matrix, key, mu_configs)

def make_event_dict(sig_mat, mu_configs, sig_keys):
    
    event_dict = {
        'n_signals': [],
        'n_mu_signals': [],
        'mu_x': [],
        'mu_y': [],
        'mu_theta': [],
        'mu_phi': [],
        'mu_time': []
    }
    
    for iev in range(sig_mat.shape[0]):
        
        indx_hit_type = sig_keys.index('is_muon')

        ## number of signals with is_muon that is not -99
        event_dict['n_signals'].append( np.sum(sig_mat[iev,:,indx_hit_type] > -1) ) 

        ## number of signals with is_muon == 1
        event_dict['n_mu_signals'].append( np.sum(sig_mat[iev,:,indx_hit_type] == 1) ) 

        ## injected mu x
        if mu_configs[iev] is not None:
            event_dict['mu_x'].append( mu_configs[iev][0] )
            event_dict['mu_y'].append( mu_configs[iev][1] )
            event_dict['mu_theta'].append( mu_configs[iev][2] )
            event_dict['mu_phi'].append( mu_configs[iev][3] )
            event_dict['mu_time'].append( mu_configs[iev][4] )
        else:
            event_dict['mu_x'].append( -99 )
            event_dict['mu_y'].append( -99 )
            event_dict['mu_theta'].append( -99 )
            event_dict['mu_phi'].append( -99 )
            event_dict['mu_time'].append( -99 )
    
    return event_dict
    
    
def main():
    
    print("Running events...")
    
    ncpu = multiprocessing.cpu_count()
    print(f"---> Using {ncpu} CPUs for parallelization")
    print(f"---> Using {my_configs.randseed} as random seed")
    np.random.seed(my_configs.randseed)

    pool = multiprocessing.Pool(ncpu)
    pbar = tqdm.tqdm(total=my_configs.nevs)

    def update(*a):
        pbar.update()

    #make array of random seeds
    random_seeds = np.random.randint(1, 2**30, size=my_configs.nevs, dtype=int)
    
    results = []
    for i in range(pbar.total):
        this_res = pool.apply_async(run_event, args=(random_seeds[i],), callback=update)
        results.append(this_res)
        
    pool.close()
    pool.join()

    if len(results) < 1 or results is None:
        print("No results found...")
        sys.exit()
  
    sig_matrix, sig_keys, mu_confs = make_signal_matrix(results)
    ev_dict = make_event_dict(sig_matrix, mu_confs, sig_keys)
    
    out_file_name = my_configs.outf.replace('.h5', 
                        f'_Rnd{my_configs.randseed}.h5')

    with h5py.File(out_file_name, 'w') as hf:
        hf.create_dataset("signals", data=sig_matrix)
        dt = h5py.special_dtype(vlen=str) 
        feature_names = np.array(sig_keys, dtype=dt) 
        hf.create_dataset("signal_keys", data=feature_names )
        for kk in ev_dict:
            hf.create_dataset('ev_'+kk, data=np.array(ev_dict[kk]) )
    print("Saved!")

if __name__== "__main__" :
    main()
