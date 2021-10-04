# import numpy as np
# import sympy

# class Muon:
#     def __init__(self, x, y, theta, phi, time=0):
#         self.org_x = x
#         self.org_y = y
#         self.org_z = 0
        
#         self.r = 1.
#         self.theta = theta
#         self.phi = phi
#         self.end_x = self.r*np.cos(self.phi)*np.sin(self.theta) + self.org_x
#         self.end_y = self.r*np.sin(self.phi)*np.sin(self.theta) + self.org_y
#         self.end_z = self.r*np.cos(self.theta)

#         ## muon is generated from 0,0,0 (IP), but this can be changed
#         self.point_org = sympy.Point3D( (self.org_x,self.org_y,0) )
#         self.point_end = sympy.Point3D( (self.end_x,self.end_y,self.end_z) )
#         self.line = sympy.Line3D(self.point_org, self.point_end)
#         self.time = time

# class Hit:
#     def __init__(self, x, y, z, time, is_muon):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.time = time
#         self.is_muon = is_muon
    
#     def point(self):
#         return sympy.Point3D(self.x, self.y, self.z)
    
#     def print(self):
#         hit_info = 'Hit information:\n\t'
#         hit_info += f'X = {self.x}' + '\n\t'
#         hit_info += f'Y = {self.y}' + '\n\t'
#         hit_info += f'Z = {self.z}' + '\n\t'
#         hit_info += f'T = {self.time}' + '\n\t'
#         hit_info += f'Is from muon = {self.is_muon}' + '\n'
#         print(hit_info)

# class Signal:
#     def __init__(self, this_plane, this_hit):
#         ## find which line segments in plane are closes to hit
#         ## then extrapolate either to the edges or to the center
#         ## of the detector
#         self.time = this_hit.time
#         self.is_muon = this_hit.is_muon
#         ## find closest line segment in x
#         hit_distancex = 99999999
#         self.hit_distancex_seg = None
        
#         for xseg in this_plane.seg_lines['x']:
#             xseg_dist = xseg.distance(this_hit.point())
#             if xseg_dist < hit_distancex:
#                 hit_distancex = xseg_dist
#                 self.hit_distancex_seg = xseg
                
#         ## find closest line segment in y
#         hit_distancey = 99999999
#         self.hit_distancey_seg = None
        
#         for yseg in this_plane.seg_lines['y']:
#             yseg_dist = yseg.distance(this_hit.point())
#             if yseg_dist < hit_distancey:
#                 hit_distancey = yseg_dist
#                 self.hit_distancey_seg = yseg
        
#         if hit_distancex == 99999999 \
#             or hit_distancey == 99999999:
#             return None
                
#         ## project at x=0, x=end, y=0, y=end
#         self.x_rightend = self.hit_distancex_seg.intersection(this_plane.edge_lines['right'])[0]
#         self.x_middle   = self.hit_distancex_seg.intersection(this_plane.edge_lines['midx'])[0]
#         self.y_topend   = self.hit_distancey_seg.intersection(this_plane.edge_lines['top'])[0]
#         self.y_middle   = self.hit_distancey_seg.intersection(this_plane.edge_lines['midy'])[0]
        
#     def set_muon(self, ismuon):
#         self.is_muon = ismuon
    
#     def print(self):
#         sig_info = 'Signal information:\n\t'
#         sig_info += f'projX_at_rightend_x = {self.x_rightend.x}' + '\n\t'
#         sig_info += f'projX_at_rightend_y = {self.x_rightend.y}' + '\n\t'
#         sig_info += f'projX_at_middle_x = {self.x_middle.x}' + '\n\t'
#         sig_info += f'projY_at_topend_x = {self.y_topend.x}' + '\n\t'
#         sig_info += f'projY_at_topend_y = {self.y_topend.y}' + '\n\t'
#         sig_info += f'projY_at_middle_x = {self.y_middle.x}' + '\n\t'
#         sig_info += f'projY_at_middle_y = {self.y_middle.y}' + '\n\t'
#         sig_info += f'hit_distancex_seg = {self.hit_distancex_seg}' + '\n\t'
#         sig_info += f'hit_distancey_seg = {self.hit_distancey_seg}' + '\n\t'
#         sig_info += f'time = {self.time}' + '\n\t'
#         sig_info += f'Is from muon = {self.is_muon}' + '\n'
#         print(sig_info)
    
#     def read_signal(self):
#         return {'projX_at_rightend_x': self.x_rightend.x,
#                    'projX_at_rightend_y': self.x_rightend.y,
#                    'projX_at_middle_x': self.x_middle.x,
#                    'projX_at_middle_y': self.x_middle.y,
#                    'projY_at_topend_x': self.y_topend.x,
#                    'projY_at_topend_y': self.y_topend.y,
#                    'projY_at_middle_x': self.y_middle.x,
#                    'projY_at_middle_y': self.y_middle.y,
#                    'time': self.time,
#                    'is_muon': self.is_muon
#                    }
        
# class Plane:
#     ## planes are aligned in z
#     ## tilt only on x segmentation so far
#     ## width_x: geometrical width of detector in x
#     ## width_y: geometrical width of detector in y
#     ## width_t: time window (in BCs = 25ns) to integrate signal
#     ## n_?_seg: number of allowed segments in each coordinate
#     ## segment size is then width_?/n_?_seg
#     def __init__(self, z, width_x=10, width_y=10, width_t=10,
#                           n_x_seg=10, n_y_seg=0, n_t_seg=10,
#                           x_res=0, y_rex=0, z_res=0, t_res=0,
#                           tilt=0):
#         ## geometry
#         self.z = z
#         self.point = sympy.Point3D(0,0,z)
#         self.plane = sympy.Plane(self.point, normal_vector=(0,0,1))

#         ## detector plane tilt
#         self.tilt = tilt

#         ## detector geometrical boundaries, assuming squares now
#         self.sizes = {
#         'x': width_x,        'y': width_y,
#         't': width_t
#         }

#         ## detector spatial segmentation in x and y,
#         ## and timing segmentation in t
#         ## Note: if you have a tilt in x, you need to increase the range
#         ## of the segmentation to ensure full coverage
#         tilt_width_x_min = -0.5*width_x
#         tilt_width_x_max = 0.5*width_x
#         if self.tilt > 0:
#             tilt_dx = 0.5*width_y*np.abs( np.tan(tilt) )
#             tilt_width_x_max = 0.5*width_x + tilt_dx
#         if self.tilt < 0:
#             tilt_dx = 0.5*width_y*np.abs( np.tan(tilt) )
#             tilt_width_x_min = -0.5*width_x - tilt_dx

#         self.segmentations = {
#         'x': np.linspace( tilt_width_x_min, tilt_width_x_max, n_x_seg+1 ),
#         'y': np.linspace( -0.5*width_y, 0.5*width_y, n_y_seg+1 ),
#         't': np.linspace( -0.5*width_t, 0.5*width_t, n_t_seg+1 )
#         }

#         self.seg_mids = {}
#         for coord in self.segmentations:
#             if len(self.segmentations[coord]) > 1:
#                 self.seg_mids[coord] = 0.5*(self.segmentations[coord][:-1] + self.segmentations[coord][1:])
#             else:
#                 self.seg_mids[coord] = self.segmentations[coord]
        
#         ## plane segmentation lines (centers)
#         self.seg_lines = {'x':[], 'y':[]}
        
#         for this_x_center in self.seg_mids['x']:
#             this_p1 = sympy.Point3D(this_x_center, 0, self.z)
#             this_p2 = sympy.Point3D(this_x_center + 0.5*width_y*np.tan(tilt), 0.5*width_y, self.z )
#             self.seg_lines['x'].append(sympy.Line3D(this_p1, this_p2))
            
#         for this_y_center in self.seg_mids['y']:
#             this_p1 = sympy.Point3D(0, this_y_center, self.z)
#             this_p2 = sympy.Point3D(0.5*width_x, this_y_center, self.z )
#             self.seg_lines['y'].append(sympy.Line3D(this_p1, this_p2))
        
#         self.edge_lines = {}
        
# #            x
# #         __|__ y
# #           |
# #         top and bottom below refer to this orientation
        
#         self.edge_lines['right']  = sympy.Line3D( sympy.Point3D(-0.5*width_x,  0.5*width_y, self.z),
#                                                   sympy.Point3D( 0.5*width_x,  0.5*width_y, self.z) )
        
#         self.edge_lines['left']   = sympy.Line3D( sympy.Point3D(-0.5*width_x, -0.5*width_y, self.z),
#                                                   sympy.Point3D( 0.5*width_x, -0.5*width_y, self.z) )
        
#         self.edge_lines['bottom'] = sympy.Line3D( sympy.Point3D(-0.5*width_x, -0.5*width_y, self.z),
#                                                   sympy.Point3D(-0.5*width_x,  0.5*width_y, self.z) )
        
#         self.edge_lines['top']    = sympy.Line3D( sympy.Point3D( 0.5*width_x, -0.5*width_y, self.z),
#                                                   sympy.Point3D( 0.5*width_x,  0.5*width_y, self.z) )
        
#         self.edge_lines['midx']   = sympy.Line3D( sympy.Point3D( 0, 0, self.z),
#                                                   sympy.Point3D( 1, 0, self.z) )
        
#         self.edge_lines['midy']   = sympy.Line3D( sympy.Point3D( 0, 0, self.z),
#                                                   sympy.Point3D( 0, 1, self.z) )

#         ## keeping position resolution as 0 for now
#         ## timing resolution of 5 BCs
#         ## Resolution smearings are applied to muon only, since noise is random
#         self.resolutions = {
#         'x': x_res,        'y': y_rex,
#         'z': z_res,        't': t_res
#         }

#         ## raw hits
#         self.hits = []

#     def clear_hits(self):
#         self.hits = []

#     def smear(self, pos, coord):
#         ## smear muon hit position and time

#         if coord not in self.resolutions:
#             print('Could not understand coordinate, must be x y z or t, but received', coord)
#             return -99

#         if self.resolutions[coord] > 0:
#             return np.random.normal(pos, self.resolutions[coord])
#         else:
#             return pos

#     def pass_muon(self, muon):
#         ## find intersection of muon and detector plane

#         pmu_intersect = self.plane.intersection(muon.line)

#         if len(pmu_intersect) == 0 or len(pmu_intersect) > 1:
#             print("There should always be one and only one muon-plane intersection. What's happening?")
#             print(pmu_intersect)
#             return -1

#         intersection_point = pmu_intersect[0]
#         mu_ip_x = self.smear(float(intersection_point.x), 'x')
#         mu_ip_y = self.smear(float(intersection_point.y), 'y')
#         mu_ip_z = self.smear(float(intersection_point.z), 'z')
#         mu_ip_t = self.smear(muon.time, 't')

#         ## if muon is outside the detector fiducial volume
#         ## or outside the time window, return 0
#         if np.abs(mu_ip_x) > 0.5*self.sizes['x']:
#             return 0
#         if np.abs(mu_ip_y) > 0.5*self.sizes['y']:
#             return 0
#         if np.abs(mu_ip_t) > 0.5*self.sizes['t']:
#             return 0

#         muhit = Hit(mu_ip_x,
#                     mu_ip_y,
#                     mu_ip_z,
#                     mu_ip_t, True)

#         self.hits.append(muhit)

#         return 1


#     def add_noise(self, n_noise):
#         ## add uniform random noise hits

#         noise_x = np.random.uniform(-0.5*self.sizes['x'], 0.5*self.sizes['x'], int(n_noise))
#         noise_y = np.random.uniform(-0.5*self.sizes['y'], 0.5*self.sizes['y'], int(n_noise))
#         noise_z = self.z*np.ones(int(n_noise))
#         noise_t = np.random.uniform(-0.5*self.sizes['t'], 0.5*self.sizes['t'], int(n_noise))

#         for inoise in range(int(n_noise)):
#             noise_hit = Hit( noise_x[inoise],
#                                 noise_y[inoise],
#                                 noise_z[inoise],
#                                 noise_t[inoise],
#                                 False)

#             self.hits.append(noise_hit)

#     def hit_processor(self):
#         ## decide on overlapping hits
        
#         ## sorting hits by which one arrived first
#         self.hits.sort(key=lambda hit: hit.time)
        
#         out_signals = []
        
#         for ihit in self.hits:
            
#             isig = Signal(self, ihit)
            
#             if isig is None:
#                 print("wrong signal")
#                 ihit.print()
#                 continue
            
#             isig.print()
            
#             if len(out_signals) == 0:
#                 out_signals.append(isig)
#                 continue
            
#             is_overlap = False
#             this_sig_ismu = isig.is_muon
            
#             for i_osig in range(0,len(out_signals)):
                
#                 osig = out_signals[i_osig]
            
#                 is_same_x = isig.hit_distancex_seg.is_similar(osig.hit_distancex_seg)
#                 is_same_y = isig.hit_distancey_seg.is_similar(osig.hit_distancey_seg)
            
#                 if is_same_x and is_same_y:
#                     ## two signals in the same detector readout element
#                     ## keep the oldest one
#                     ## set to muon if either one of them is from muon
#                     is_overlap = True
#                     this_sig_ismu = isig.is_muon or osig.is_muon
#                     out_signals.pop(i_osig)
#                     break

#             isig.is_muon = this_sig_ismu
#             out_signals.append(isig)

#         return out_signals
    
#     def return_signal(self):

#         out_signals = self.hit_processor()
        
#         out_dict = {}
        
#         if len(out_signals) < 1:
#             return out_dict
#         else:
#             out_dict = dict.fromkeys(out_signals[0].read_signal())
            
#             for kk in out_dict: 
#                 out_dict[kk] = []            
#                 for osig in out_signals:
#                     out_dict[kk].append( float(osig.read_signal()[kk]) )
        
#         return out_dict
