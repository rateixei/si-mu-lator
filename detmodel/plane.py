import numpy as np
import sympy
from enum import Enum
from detmodel.hit import Hit
from detmodel.signal import Signal, Segment
from detmodel.muon import Muon
from detmodel import util

class DetType(Enum):
    MM   = 'mm'
    MDT  = 'mdt'
    STGC = 'stgc'

    def asint(self):
        return {
            DetType.MM: 0,
            DetType.MDT: 1,
            DetType.STGC: 2
        }.get(self)

    def asstr(self):
        return {
            DetType.MM: 'mm',
            DetType.MDT: 'mdt',
            DetType.STGC: 'stgc'
        }.get(self)

class Plane:
    ## planes are aligned in z
    ## tilt only on x segmentation so far
    ## width_x: geometrical width of detector in x
    ## width_y: geometrical width of detector in y
    ## width_t: time window (in BCs = 25ns) to integrate signal
    ## n_?_seg: number of allowed segments in each coordinate
    ## segment size is then width_?/n_?_seg
    def __init__(self, type, z, width_x=10, width_y=10, width_t=10,
                          n_x_seg=10, n_y_seg=0, n_t_seg=10,
                          x_res=0, y_res=0, z_res=0, t_res=0,
                          tilt=0, offset=0, max_hits=0, sig_eff=0):
        ## type
        self.p_type = DetType(type)

        ## geometry
        self.z = z
        self.point = sympy.Point3D(0,0,z,evaluate=False)
        self.plane = sympy.Plane(self.point, normal_vector=(0,0,1))

        ## noise info
        self.noise_rate = 0
        self.noise_type = 'constant'
        self.noise_hits = 0

        ## detector plane tilt and offset
        self.tilt = tilt
        self.offset = offset
        self.max_hits = max_hits
        self.sig_eff = sig_eff

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
        'x': np.linspace( tilt_width_x_min+self.offset, tilt_width_x_max+self.offset, n_x_seg+1 ),
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
            this_p1 = sympy.Point3D(this_x_center, 0, self.z, evaluate=False)
            this_p2 = sympy.Point3D(this_x_center + 0.5*width_y*np.tan(tilt), 0.5*width_y, self.z, evaluate=False)
            self.seg_lines['x'].append(Segment( sympy.Line3D(this_p1, this_p2), coord='x', z=self.z ))
            
        for this_y_center in self.seg_mids['y']:
            this_p1 = sympy.Point3D(0, this_y_center, self.z, evaluate=False)
            this_p2 = sympy.Point3D(0.5*width_x, this_y_center, self.z, evaluate=False)
            self.seg_lines['y'].append(Segment( sympy.Line3D(this_p1, this_p2), coord='y', z=self.z ))

        ## keeping position resolution as 0 for now
        ## timing resolution of 5 BCs
        ## Resolution smearings are applied to muon only, since noise is random
        self.resolutions = {
        'x': x_res,        'y': y_res,
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
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z, evaluate=False),
                                                  sympy.Point3D( 0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z, evaluate=False) )
        elif 'left' in edge:
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z, evaluate=False),
                                                  sympy.Point3D( 0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z, evaluate=False) )
        elif 'bottom' in edge:
            return sympy.Line3D( sympy.Point3D(-0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z, evaluate=False),
                                                  sympy.Point3D(-0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z, evaluate=False) )
        elif 'top' in edge:
            return sympy.Line3D( sympy.Point3D( 0.5*self.sizes['x'], -0.5*self.sizes['y'], self.z, evaluate=False),
                                                  sympy.Point3D( 0.5*self.sizes['x'],  0.5*self.sizes['y'], self.z, evaluate=False) )
        elif 'midx' in edge:
            return sympy.Line3D( sympy.Point3D( 0, 0, self.z, evaluate=False),
                                                  sympy.Point3D( 1, 0, self.z, evaluate=False) )
        elif 'midy' in edge:
            return sympy.Line3D( sympy.Point3D( 0, 0, self.z, evaluate=False),
                                                  sympy.Point3D( 0, 1, self.z, evaluate=False) )
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

    def pass_muon(self, muon, randseed=42):
        np.random.seed(int(randseed + 10*(self.z)))

        ## apply signal efficiency
        if self.sig_eff > 0:
            rnd_number_eff = np.random.uniform(0.0, 1.0)
            if rnd_number_eff > self.sig_eff:
                return 0 ## missed muon signal

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

        # To compute the drift radius (for MDT detector), need to find detector element (i.e. wire) 
        # for which this muon has the smallest distance of closest approach to the wire
        # Caveat: the calculation below assumes tubes are exactly vertical
        mu_ix = -9999
        mu_rdrift = 9999.
        if self.p_type == DetType.MDT:
            for islx, slx in enumerate(self.seg_lines['x']):
                wirepos = sympy.Point(slx.line.p1.x, slx.line.p1.z, evaluate=False)
                muonpos1 = sympy.Point(muon.line.p1.x, muon.line.p1.z, evaluate=False)
                muonpos2 = sympy.Point(muon.line.p2.x, muon.line.p2.z, evaluate=False)
                muonline = sympy.Line(muonpos1, muonpos2)
                rdrift = muonline.distance(wirepos)
                if rdrift.evalf() < mu_rdrift:
                    mu_ix = islx
                    mu_rdrift = rdrift.evalf()
            # do not record hit if closest wire further than tube radius
            if mu_rdrift > 0.5*self.sizes['x']/len(self.seg_lines['x']): 
                return 0
            
        muhit = Hit(mu_ip_x,
                    mu_ip_y,
                    mu_ip_z,
                    mu_ip_t, 
                    mu_ix,
                    mu_rdrift,
                    True,
                    False)

        self.hits.append(muhit)
        
        return 1

    def set_noise(self, noise_rate, noise_type='constant'):
        self.noise_rate = noise_rate
        self.noise_type = noise_type

    def add_noise(self, noise_scale, override_n_noise_per_plane=-1, randseed=42):

        '''
        p_width_t is the time window in which to integrate the signal (in nano seconds)
        therefore, the number of noise hits is:
        
             noise_scale * noise_rate per strip (Hz) * number of strips * p_width_t (ns) * 1e-9
        '''

        if 'constant' not in self.noise_type:
            print('Only support constant noise now')
            return -1
        
        if self.sizes['t'] < 1e-15:
            print('Time integration width must be larger than 0')
            return -1

        n_noise_init = noise_scale * (len(self.segmentations['x']) -1) \
                        * self.noise_rate * self.sizes['t'] * 1e-9
        if override_n_noise_per_plane > 0:
            n_noise_init = override_n_noise_per_plane

        np.random.seed(int(randseed + (self.z)))
        n_noise = np.random.poisson(n_noise_init)
        self.noise_hits = n_noise

        noise_x = np.random.uniform(-0.5*self.sizes['x'], 0.5*self.sizes['x'], int(n_noise))
        noise_y = np.random.uniform(-0.5*self.sizes['y'], 0.5*self.sizes['y'], int(n_noise))
        noise_z = self.z*np.ones(int(n_noise))
        noise_t = np.random.uniform(-0.5*self.sizes['t'], 0.5*self.sizes['t'], int(n_noise))
        noise_r = 9999.*np.ones(int(n_noise))
        if self.p_type == DetType.MDT:
            noise_r = np.random.uniform(0.0, 0.5*self.sizes['x']/len(self.seg_lines['x']), int(n_noise))

        for inoise in range(int(n_noise)):
            # find detector element (segment) closest to each noise hit along x, as needed for MDT
            noise_ix = np.argmin( [ np.abs(noise_x[inoise]-xseg.line.p1.x) for xseg in self.seg_lines['x'] ] )

            noise_hit = Hit(noise_x[inoise],
                            noise_y[inoise],
                            noise_z[inoise],
                            noise_t[inoise],
                            noise_ix,
                            noise_r[inoise],
                            False,
                            False)

            self.hits.append(noise_hit)
    
    def find_signal(self, this_hit):
        
        ## find which detector segment this hit has activated
        
        hit_distancex_seg = None
        hit_hash_ix = -10
        hit_distancey_seg = None
        hit_hash_iy = -10

        if self.p_type == DetType.MDT: # association between hit and detector element (segment) already done according to rdrift
            hit_hash_ix = this_hit.seg_ix
        else:
            # if no tilt in this plane or the hit is in the middle of detector element along y,
            # speed up finding of nearest element
            #hit_hash_ix = np.argmin( [ xseg.line.distance(this_hit.point()) for xseg in self.seg_lines['x'] ] )            
            hit_hash_ix = np.argmin( [ util.distpoint2line(xseg, this_hit) for xseg in self.seg_lines['x'] ] )

            
        #hit_hash_iy = np.argmin( [ yseg.line.distance(this_hit.point()) for yseg in self.seg_lines['y'] ] )
        hit_hash_iy = np.argmin( [ util.distpoint2line(yseg, this_hit) for yseg in self.seg_lines['y'] ] )

        
        ## if segment already has signal, skip (but set to muon if new signal is from muon)
            
        if self.seg_lines['x'][hit_hash_ix].is_sig == False or \
            self.seg_lines['y'][hit_hash_iy].is_sig == False:
            
            if self.p_type == DetType.STGC and this_hit.skip:  # do not promote as signal
                return None
            
            isig = Signal( hash_seg_line_x=hit_hash_ix, hash_seg_line_y=hit_hash_iy,
                           x=this_hit.x, y=this_hit.y, z=this_hit.z,
                           time=this_hit.time, seg_ix=this_hit.seg_ix, rdrift=this_hit.rdrift,
                           is_muon=this_hit.is_muon )
            
            self.seg_lines['x'][hit_hash_ix].add_signal(isig)
            self.seg_lines['y'][hit_hash_iy].add_signal(isig)
            
            return isig.get_info_wrt_plane(self, display=False)
    
        else:
            
            if this_hit.is_muon:
                if self.seg_lines['x'][hit_hash_ix].is_sig and \
                    self.seg_lines['y'][hit_hash_iy].is_sig:
                    
                    self.seg_lines['x'][hit_hash_ix].sig.is_muon = True
                    self.seg_lines['y'][hit_hash_iy].sig.is_muon = True

            return None


    def hit_processor(self, summary=False):
        ## decide on overlapping hits
        
        ## sorting hits by which one arrived first
        if self.p_type == DetType.MDT:
            self.hits.sort(key=lambda hit: hit.rdrift)
        else:
            self.hits.sort(key=lambda hit: hit.time)

        ## apply additional position smearing by combining muon and noise hits if sTGC plane
        if len(self.hits) > 1 and self.p_type == DetType.STGC:
            self.combine_hits(False)
        
        out_signals = []
        
        if summary:
            print("Total number of hits:", len(self.hits) )
    
        for ihit in self.hits:
            
            isig_info = self.find_signal(ihit)
            if isig_info is not None:
                out_signals.append(isig_info)

            if self.max_hits > 0 and len(out_signals) == self.max_hits:
                    break

        n_sigs = len(out_signals)
        
        if n_sigs < 1:
            return None
        
        n_props = len(out_signals[0])
        sig_matrix = np.zeros( (n_sigs, n_props) )
        
        for ns in range(n_sigs):
            sig_matrix[ns][:] = list( out_signals[ns].values() )
            
        return (sig_matrix, list(out_signals[ns].keys()) )

    def combine_hits(self, summary=False):
        
        ## Combine hits in the same plane into one hit
        ## For the time being, only do so if a muon hit exists
        ## Background noise hit positions are averaged with muon hit position but with a reduced weight
        
        imu = -1
        ibkg = -1
        sumx = 0.
        sumw = 0.
        list_bkg_ix = []
        list_seg_ix = [] # store detector segment with hits
        for ihit, hit in enumerate(self.hits):
            
            if self.max_hits > 0 and ihit == self.max_hits:
                break
            
            hit_ix = np.argmin( [ util.distpoint2line(xseg, hit) for xseg in self.seg_lines['x'] ] )
            ## Here we rely on the hits being ordered to avoid multiple hits on same detector segment
            if summary:
                print('hit_ix',hit_ix,'is muon?',hit.is_muon)

            # comment out code below; removing it has the effect of not priviledging sTGC hits arriving earlier
            #if hit_ix in list_seg_ix:
            #    continue
            list_seg_ix.append(hit_ix)
            if hit.is_muon:
                imu = ihit
                weight = 1.0
            else:    # background noise hit
                list_bkg_ix.append(ihit)
                weight = 0.2
            sumx += weight*hit.x
            sumw += weight
        
        ## Update x position of muon hit (if one exists), otherwise pick one of the noise hits if several
        if imu >= 0:
            self.hits[imu].x = sumx/sumw
        else:    # if no muon hit pick one of the noise hits randomly to become signal
            ibkg = np.random.random_integers(0, len(list_bkg_ix)-1)
            self.hits[ibkg].x = sumx/sumw
        ## Flag background noise hits
        for i in list_bkg_ix:
            if i != list_bkg_ix[ibkg] or ibkg == -1:
                self.hits[i].skip = True # hit will be suppressed and not considered output signal
        if summary:
            print('imu, ibkg, list_bkg_ix', imu, ibkg, list_bkg_ix)
                
        return None
        
    def return_signal(self, summary=False):
        return self.hit_processor(summary)
  
