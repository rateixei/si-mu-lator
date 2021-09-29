import numpy as np
import argparse
import sys
sys.path.insert(0, './detmodel/')
import detector
import elements

def setup_detector(card):
    det = detector.Detector()
    det.read_card(card)
    return det

def event(det, confs):
    det.reset_planes()
    
    ## muon
    if confs.ismu:
        mu_x = 0
        if confs.muonx[0] < confs.muonx[1]:
            mu_x = np.random.uniform(low=confs.muonx[0], high=confs.muonx[1])
        
        mu_y = 0
        if confs.muony[0] < confs.muony[1]:
            mu_y = np.random.uniform(low=confs.muony[0], high=confs.muony[1])
            
        mu_a = 0
        if confs.muona[0] < confs.muona[1]:
            mu_a = np.random.uniform(low=confs.muona[0], high=confs.muona[1])
        
        det.add_muon(mu_x=mu_x, mu_y=mu_y, mu_theta=mu_a, mu_phi=0, mu_time=0)
    
    ## background
    if confs.bkgr > 0:
        det.add_noise("constant", confs.bkgr)
    
    ## signals
    signals = det.get_signals()
    
    return signals
    

def main():
    parser = argparse.ArgumentParser(description='Si-μ-late')
    
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
    
    args = parser.parse_args()
    print(args)
    
    det = setup_detector(args.detcard)
    
    for iev in range(0, args.nevs):
        print(event(det, args))
    
    

if __name__== "__main__" :
    main()