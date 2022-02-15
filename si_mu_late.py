import numpy as np
import argparse
from detmodel.detector import Detector

import multiprocessing
import tqdm
import time

import h5py

import pandas

my_configs = 0

parser = argparse.ArgumentParser(description='Si-MU-late')
    
# general and required
parser.add_argument('-d', '--detector', dest='detcard', type=str, required=True,
                    help='Detector card')
parser.add_argument('-n', '--nevents', dest='nevs', type=int, required=True,
                    help='Number of events')
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
                       help='Background rate (Hz) per module unit')

# other
parser.add_argument('-r', '--randomseed', dest='randseed', type=int, default=42,
                       help='Set random seed')
    
my_configs = parser.parse_args()

    
my_detector = Detector()
my_detector.read_card(my_configs.detcard)

def run_event(iev):
    np.random.seed(my_configs.randseed + iev)
    
    my_detector.reset_planes()
    
    ## muon
    mu_config = None
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
        
        my_detector.add_muon(mu_x=mu_x, mu_y=mu_y, mu_theta=mu_a, mu_phi=0, mu_time=0)
        mu_config = [mu_x, mu_y, mu_a, 0, 0]
    
    ## background
    if my_configs.bkgr > 0:
        my_detector.add_noise("constant", my_configs.bkgr)
    
    ## signals
    sigs_keys = my_detector.get_signals(iev)
    
    if sigs_keys is not None:
        return (sigs_keys[0], sigs_keys[1], mu_config)
    else:
        return None
    

def make_signal_matrix(res):
    
    evs = []

    ## get max number of hits
    max_sigs = np.max( [ iev.get()[0].shape[0] for iev in res ] )
    
    ## assert that  dicts have the same number of columns
    assert all(iev.get()[0].shape[1]==res[0].get()[0].shape[1] for iev in res)
    ncols = res[0].get()[0].shape[1]

    out_matrix = -99 * np.ones( ( len(res), max_sigs, ncols ) )
    key = []
    mu_configs = []
    for iiev,iev in enumerate(res):
        this_res, this_key, muconf = iev.get()
        if len(key) == 0:
            key = this_key[:]
        this_shape = this_res.shape
        out_matrix[iiev][ :this_shape[0], : ] = this_res
        mu_configs.append(muconf)
    
    return(out_matrix, key, mu_configs)

def make_event_dict(sig_mat, mu_configs):
    
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
        
        ## number of signals with z > 0
        event_dict['n_signals'].append( np.sum(sig_mat[iev,:,1] > 0) ) 
        ## number of signals with is_muon == True
        event_dict['n_mu_signals'].append( np.sum(sig_mat[iev,:,0] == True) ) 
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
    
    pool = multiprocessing.Pool(ncpu)
    pbar = tqdm.tqdm(total=my_configs.nevs)

    def update(*a):
        pbar.update()

    results = []
    for i in range(pbar.total):
        this_res = pool.apply_async(run_event, args=(i,), callback=update)
        if this_res is not None:
            results.append(this_res)
        
    pool.close()
    pool.join()

    if len(results) < 1:
        print("No results found...")
        sys.exit()
  
    sig_matrix, sig_keys, mu_confs = make_signal_matrix(results)
    ev_dict = make_event_dict(sig_matrix, mu_confs)
    
    out_file_name = my_configs.outf.replace('.h5', 
                        f'_Rnd{my_configs.randseed}.h5')

    with h5py.File(out_file_name, 'w') as hf:
        hf.create_dataset("signals", data=sig_matrix)
        hf.create_dataset("signal_keys", data=np.array(mu_confs))
        for kk in ev_dict:
            hf.create_dataset('ev_'+kk, data=np.array(ev_dict[kk]) )
    print("Saved!")
        


#     for i in range(my_configs.nevs):
#         a = run_event(i)
#         print(a)

if __name__== "__main__" :
    main()
