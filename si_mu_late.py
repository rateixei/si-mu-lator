import numpy as np
import argparse
from detmodel.detector import Detector

import multiprocessing
import tqdm

my_configs = 0


parser = argparse.ArgumentParser(description='Si-MU-late')
    
# general and required
parser.add_argument('-d', '--detector', dest='detcard', type=str, required=True,
                    help='Detector card')
parser.add_argument('-n', '--nevents', dest='nevs', type=int, required=True,
                    help='Number of events')
    
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
    
my_configs = parser.parse_args()
    
my_detector = Detector()
my_detector.read_card(my_configs.detcard)

def run_event(iev):
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
    return my_detector.get_signals()
    

def main():
    
    print("Running events...")
    
    pool = multiprocessing.Pool(4)
    pbar = tqdm.tqdm(total=my_configs.nevs)

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(run_event, args=(i,), callback=update)
        
    pool.close()
    pool.join()

#     for i in range(my_configs.nevs):
#         a = run_event(i)

if __name__== "__main__" :
    main()
