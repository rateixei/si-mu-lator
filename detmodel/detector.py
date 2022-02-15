import numpy as np
import sympy
import yaml
import sys
from detmodel.hit import Hit
from detmodel.signal import Signal
from detmodel.muon import Muon
from detmodel.plane import Plane

## ToDo:
### Add print method to summarize detector
### implement noise as a function of geometry

class Detector:
    def __init__(self):
        print("-- Initializing detector --")
        self.specs = {}
        self.planes = []
        self.mymu = 0
        self.has_mu = 0

    def reset_planes(self):
#         print("-- Resetting planes --")
        if len(self.planes) > 0:
            for p in self.planes:
                p.clear_hits()

    def add_muon(self, mu_x, mu_y, mu_theta, mu_phi=0, mu_time=0):
#         print("-- Adding muon --")
        self.has_mu = 1
        self.muinit = {'x': mu_x, 'y': mu_y, 'theta': mu_theta, 'phi': mu_phi, 'time': mu_time}
        self.mymu = Muon(x=mu_x, y=mu_y, theta=mu_theta, phi=mu_phi, time=mu_time)

        mu_res = []
        for p in self.planes:
            p_res = p.pass_muon(self.mymu)
            mu_res.append(p_res)

        return mu_res

    def add_noise(self, noise_type, noise_rate_per_module):
#         print("-- Adding noise --")
        
        '''
        p_width_t is the time window in which to integrate the signal (in nano seconds)
        therefore, the number of noise hits is:
        
             noise_rate_per_module (Hz) * p_width_t (ns) * 1e-9
        '''
        
        if self.specs['det_width_t'] is 0:
                print("det_width_t is set to 0, so you're trying to integrate noise over a window of 0 time, please specify time window")
                sys.exit()
                
        if noise_type=='constant':    

            n_noise = noise_rate_per_module * self.specs['det_width_t'] * 1e-9
                       
            for p in self.planes:
                p.add_noise(n_noise)
        else:
            print("TBI")

    def get_signals(self, iev=-1, summary=False):
#         print("-- Getting signals --")
        signals = []
        keys = []
        
        for ip,p in enumerate(self.planes):
            p_return = p.return_signal(summary)
            
            if p_return is not None:
                p_sig, p_keys = p_return
                signals.append(p_sig)
                
                if len(keys) == 0:
                    keys = p_keys[:]

        if len(signals) == 0:
            return None
        else:
            signals = np.concatenate(signals)
            return (signals,keys)
    
    def find_plane_par(self, par, iplane):
        
        if 'name' not in self.specs:
            print("Need to initialize specs first")
            return None
        
        if par in self.specs['planes'][iplane]:
            return self.specs['planes'][iplane][par]
        elif str('det_'+par) in self.specs:
            return self.specs[str('det_'+par)]
        else:
            return 0
        

    def read_card(self, detector_card):
        print("-- Reading card --")
        
        with open(detector_card) as f:
            # fyml = yaml.load(f, Loader=yaml.FullLoader) ## only works on yaml > 5.1
            fyml = yaml.safe_load(f)
            self.specs = fyml['detector']

        for p in self.specs['planes']:

            if 'z' not in self.specs['planes'][p]:
                print("Need to specify z for all planes")
                sys.exit()

            p_z = self.specs['planes'][p]['z']

            p_tilt    = self.find_plane_par('tilt', p)
            p_offset  = self.find_plane_par('offset', p)

            p_width_x = self.find_plane_par('width_x', p)
            p_width_y = self.find_plane_par('width_y', p)
            p_width_t = self.find_plane_par('width_t', p)

            p_n_x_seg = self.find_plane_par('n_x_seg', p)
            p_n_y_seg = self.find_plane_par('n_y_seg', p)
            p_n_t_seg = self.find_plane_par('n_t_seg', p)

            p_x_res   = self.find_plane_par('x_res', p)
            p_y_res   = self.find_plane_par('y_res', p)
            p_z_res   = self.find_plane_par('z_res', p)
            p_t_res   = self.find_plane_par('t_res', p)

            if p_width_x == 0 or p_width_y == 0 or p_width_t == 0 \
                or p_n_x_seg == 0 or p_n_t_seg == 0:
                print("Plane information not correctly set")
                print( f'p_width_x: {p_width_x}' )
                print( f'p_width_y: {p_width_y}' )
                print( f'p_width_t: {p_width_t}' )
                print( f'p_n_x_seg: {p_n_x_seg}' )
                print( f'p_n_y_seg: {p_n_y_seg}' )
                print( f'p_n_t_seg: {p_n_t_seg}' )
                sys.exit()

            ## Currently, the only supported types are MicroMegas and MDTs
            p_type = 'mm' if 'type' not in self.specs['planes'][p] else self.specs['planes'][p]['type']
            
            p_i = Plane(type=p_type, z=p_z,
            width_x=p_width_x, width_y=p_width_y, width_t=p_width_t,
            n_x_seg=p_n_x_seg, n_y_seg=p_n_y_seg, n_t_seg=p_n_t_seg,
            x_res=p_x_res, y_res=p_y_res, z_res=p_z_res, t_res=p_t_res,
            tilt=p_tilt, offset=p_offset)

            self.planes.append(p_i)
