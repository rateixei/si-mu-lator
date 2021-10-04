import numpy as np
import sympy
from detmodel.hit import Hit
from detmodel.signal import Signal, Segment
from detmodel.muon import Muon

class Plane:
    ## planes are aligned in z
    ## tilt only on x segmentation so far
    ## width_x: geometrical width of detector in x
    ## width_y: geometrical width of detector in y
    ## width_t: time window (in BCs = 25ns) to integrate signal
    ## n_?_seg: number of allowed segments in each coordinate
    ## segment size is then width_?/n_?_seg
    def __init__(self, z, width_x=10, width_y=10, width_t=10,
                          n_x_seg=10, n_y_seg=0, n_t_seg=10,
                          x_res=0, y_rex=0, z_res=0, t_res=0,
                          tilt=0):
        ## geometry
        self.z = z
        self.point = sympy.Point3D(0,0,z)
        self.plane = sympy.Plane(self.point, normal_vector=(0,0,1))

        ## detector plane tilt
        self.tilt = tilt

        ## detector geometrical boundaries, assuming squares now
        self.sizes = {
        'x': width_x,        'y': width_y,
        't': width_t
        }

        ## detector spatial segmentation in x and y,
        ## and timing segmentation in t
        ## Note: if you have a tilt in x, you need to increase the range
        ## of the segmentation to ensure full coverage
        tilt_width_x_min = -0.5*width_x
        tilt_width_x_max = 0.5*width_x
        if abs(self.tilt) > 0:
            tilt_dx = width_y*np.abs( np.tan(tilt) )
            tilt_width_x_max = 0.5*width_x + 0.5*tilt_dx
            tilt_width_x_min = -0.5*width_x - 0.5*tilt_dx

        self.segmentations = {
        'x': np.linspace( tilt_width_x_min, tilt_width_x_max, n_x_seg+1 ),
        'y': np.linspace( -0.5*width_y, 0.5*width_y, n_y_seg+1 ),
        't': np.linspace( -0.5*width_t, 0.5*width_t, n_t_seg+1 )
        }

        self.seg_mids = {}
        for coord in self.segmentations:
            if len(self.segmentations[coord]) > 1:
                self.seg_mids[coord] = 0.5*(self.segmentations[coord][:-1] + self.segmentations[coord][1:])
            else:
                self.seg_mids[coord] = self.segmentations[coord]
        
        ## plane segmentation lines (centers)
        self.seg_lines = {'x':[], 'y':[]}
        
        for this_x_center in self.seg_mids['x']:
            this_p1 = sympy.Point3D(this_x_center, 0, self.z)
            this_p2 = sympy.Point3D(this_x_center + 0.5*width_y*np.tan(tilt), 0.5*width_y, self.z )
            self.seg_lines['x'].append(Segment( sympy.Line3D(this_p1, this_p2), coord='x' ))
            
        for this_y_center in self.seg_mids['y']:
            this_p1 = sympy.Point3D(0, this_y_center, self.z)
            this_p2 = sympy.Point3D(0.5*width_x, this_y_center, self.z )
            self.seg_lines['y'].append(Segment( sympy.Line3D(this_p1, this_p2), coord='y' ))

        ## keeping position resolution as 0 for now
        ## timing resolution of 5 BCs
        ## Resolution smearings are applied to muon only, since noise is random
        self.resolutions = {
        'x': x_res,        'y': y_rex,
        'z': z_res,        't': t_res
        }

        ## raw hits
        self.hits = []
    
    def get_edge(self, edge):
        #            x
        #         __|__ y
        #           |
        #         top and bottom below refer to this orientation
        
        if 'right' in edge:
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z),
                                                  sympy.Point3D( 0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z) )
        elif 'left' in edge:
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z),
                                                  sympy.Point3D( 0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z) )
        elif 'bottom' in edge:
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z),
                                                  sympy.Point3D(-0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z) )
        elif 'top' in edge:
            return sympy.Line3D( sympy.Point3D( 0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z),
                                                  sympy.Point3D( 0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z) )
        elif 'midx' in edge:
            return sympy.Line3D( sympy.Point3D( 0, 0, self.z),
                                                  sympy.Point3D( 1, 0, self.z) )
        elif 'midy' in edge:
            return sympy.Line3D( sympy.Point3D( 0, 0, self.z),
                                                  sympy.Point3D( 0, 1, self.z) )
        else:
            print('Must specify: right, left, bottom, top, midx or midy')
            return -1

    def clear_hits(self):
        self.hits = []
        for slx in self.seg_lines['x']:
            slx.reset()
            
        for sly in self.seg_lines['y']:
            sly.reset()
            

    def smear(self, pos, coord):
        ## smear muon hit position and time

        if coord not in self.resolutions:
            print('Could not understand coordinate, must be x y z or t, but received', coord)
            return -99

        if self.resolutions[coord] > 0:
            return np.random.normal(pos, self.resolutions[coord])
        else:
            return pos

    def pass_muon(self, muon):
        ## find intersection of muon and detector plane

        pmu_intersect = self.plane.intersection(muon.line)

        if len(pmu_intersect) == 0 or len(pmu_intersect) > 1:
            print("There should always be one and only one muon-plane intersection. What's happening?")
            print(pmu_intersect)
            return -1

        intersection_point = pmu_intersect[0]
        mu_ip_x = self.smear(float(intersection_point.x), 'x')
        mu_ip_y = self.smear(float(intersection_point.y), 'y')
        mu_ip_z = self.smear(float(intersection_point.z), 'z')
        mu_ip_t = self.smear(muon.time, 't')

        ## if muon is outside the detector fiducial volume
        ## or outside the time window, return 0
        if np.abs(mu_ip_x) > 0.5*self.sizes['x']:
            return 0
        if np.abs(mu_ip_y) > 0.5*self.sizes['y']:
            return 0
        if np.abs(mu_ip_t) > 0.5*self.sizes['t']:
            return 0

        muhit = Hit(mu_ip_x,
                    mu_ip_y,
                    mu_ip_z,
                    mu_ip_t, True)

        self.hits.append(muhit)

        return 1


    def add_noise(self, n_noise):
        ## add uniform random noise hits

        noise_x = np.random.uniform(-0.5*self.sizes['x'], 0.5*self.sizes['x'], int(n_noise))
        noise_y = np.random.uniform(-0.5*self.sizes['y'], 0.5*self.sizes['y'], int(n_noise))
        noise_z = self.z*np.ones(int(n_noise))
        noise_t = np.random.uniform(-0.5*self.sizes['t'], 0.5*self.sizes['t'], int(n_noise))

        for inoise in range(int(n_noise)):
            noise_hit = Hit( noise_x[inoise],
                                noise_y[inoise],
                                noise_z[inoise],
                                noise_t[inoise],
                                False)

            self.hits.append(noise_hit)
    
    def find_signal(self, this_hit):
        
        ## find which segment this hit has activated
        
        hit_distancex_seg = None
        hit_hash_ix = -10
        hit_distancey_seg = None
        hit_hash_iy = -10
        
        hit_hash_ix = np.argmin( [ xseg.line.distance(this_hit.point()) for xseg in self.seg_lines['x'] ] )
        hit_hash_iy = np.argmin( [ yseg.line.distance(this_hit.point()) for yseg in self.seg_lines['y'] ] )
        
        ## if segment already has signal, skip (but set to muon if new signal is from muon)
        
        if self.seg_lines['x'][hit_hash_ix].is_sig and self.seg_lines['y'][hit_hash_iy].is_sig:
            if this_hit.is_muon:
                self.seg_lines['x'][hit_hash_ix].is_muon = True
                self.seg_lines['y'][hit_hash_iy].is_muon = True
                return None
        else:
            isig = Signal( hash_seg_line_x=hit_hash_ix, hash_seg_line_y=hit_hash_iy, 
                          time=this_hit.time, is_muon=this_hit.is_muon )
            
            self.seg_lines['x'][hit_hash_ix].add_signal(isig)
            self.seg_lines['y'][hit_hash_iy].add_signal(isig)
            
            return isig.get_info_wrt_plane(self)


    def hit_processor(self):
        ## decide on overlapping hits
        
        ## sorting hits by which one arrived first
        self.hits.sort(key=lambda hit: hit.time)
        
        out_signals = []
        
        for ihit in self.hits:
            isig_info = self.find_signal(ihit)
            if isig_info is not None:
                out_signals.append(isig_info)

        return out_signals

    
    def return_signal(self):

        out_signals = self.hit_processor()
        
        out_dict = {}
        
        if len(out_signals) < 1:
            return out_dict
        else:
            out_dict = dict.fromkeys(out_signals[0])
            
            for kk in out_dict: 
                out_dict[kk] = []            
                for osig in out_signals:
                    out_dict[kk].append( float(osig[kk]) )
        
        return out_signals