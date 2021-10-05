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
                    default=(0,0), help='Generated muon X')
parser.add_argument('-y', '--muony', nargs=2, metavar=('muymin', 'muymax'), 
                    default=(0,0), help='Generated muon Y')
parser.add_argument('-a', '--muona', nargs=2, metavar=('muamin', 'muamax'), 
                    default=(0,0), help='Generated muon angle')
    
# background simulation
parser.add_argument('-b', '--bkgrate', dest='bkgr', type=float, default=0,
                       help='Background rate (Hz) per module unit')

# other
parser.add_argument('-r', '--randomseed', dest='randseed', type=float, default=42,
                       help='Set random seed')
    
my_configs = parser.parse_args()

    
my_detector = Detector()
my_detector.read_card(my_configs.detcard)

def run_event(iev):
    np.random.seed(my_configs.randseed + iev)
    
    my_detector.reset_planes()
    
    ## muon
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
    
    ## background
    if my_configs.bkgr > 0:
        my_detector.add_noise("constant", my_configs.bkgr)
    
    ## signals
    return my_detector.get_signals(iev)
    

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
        results.append(this_res)
        
    pool.close()
    pool.join()
    
    final_results = [ pandas.DataFrame(res.get()) for res in results ]

    if len(final_results) < 1:
        print("No results found...")
        sys.exit()
    
    final_results = pandas.concat(final_results)
    
    final_results.to_hdf(my_configs.outf, 'signals')
    
#     print(final_results)
#     pd = pandas.DataFrame(final_results)
#     print(pd)
#     final_results = np.concatenate(final_results)
    
#     ## transform dict
#     transf_dict = {}
#     for kk in final_results[0].keys():
#         transf_dict[kk] = [ final_results[ir][kk] for ir in range(len(final_results)) ]

#     print(transf_dict)
    
#     with h5py.File(my_configs.outf, 'w') as hf:
#         hf.create_dataset("hits", data=transf_dict)
#         print("Saved!")


#     for i in range(my_configs.nevs):
#         a = run_event(i)
#         print(a)

if __name__== "__main__" :
    main()
