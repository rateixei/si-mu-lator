import numpy as np
import sympy
import sys
import copy

class Segment:
    def __init__(self, seg_line, coord, z):
        self.line=seg_line
        self.z=z
        if 'x' not in coord and 'y' not in coord:
            print('Segment coord must be x or y')
            sys.exit()
        else:
            self.coord = coord

        self.is_sig = False
        self.sig = None
        
    def reset(self):
        self.is_sig = False
        self.sig = None
        
    def add_signal(self,sig):
        self.sig = copy.deepcopy(sig)
        self.is_sig = True
    
class Signal:
    def __init__(self, hash_seg_line_x, hash_seg_line_y, z, time, seg_ix, rdrift, is_muon):
        self.hash_seg_x = hash_seg_line_x
        self.hash_seg_y = hash_seg_line_y
        self.z = z
        self.time = time
        self.seg_ix = seg_ix
        self.rdrift = rdrift
        self.is_muon = is_muon
        
    def get_info_wrt_plane(self, plane, display=False):
        x_rightend = plane.seg_lines['x'][self.hash_seg_x].line.intersection(plane.get_edge('right'))[0]
        x_middle   = plane.seg_lines['x'][self.hash_seg_x].line.intersection(plane.get_edge('midx'))[0]
        y_topend   = plane.seg_lines['y'][self.hash_seg_y].line.intersection(plane.get_edge('top'))[0]
        y_middle   = plane.seg_lines['y'][self.hash_seg_y].line.intersection(plane.get_edge('midy'))[0]
        
        hit_dict =  {'projX_at_rightend_x': float(x_rightend.x),
                   'projX_at_rightend_y': float(x_rightend.y),
                   'projX_at_middle_x': float(x_middle.x),
                   'projX_at_middle_y': float(x_middle.y),
                   'projY_at_topend_x': float(y_topend.x),
                   'projY_at_topend_y': float(y_topend.y),
                   'projY_at_middle_x': float(y_middle.x),
                   'projY_at_middle_y': float(y_middle.y),
                   'z': plane.z,
                   'time': self.time,
                   'seg_ix': self.seg_ix,
                   'rdrift': self.rdrift,
                   'is_muon': self.is_muon
                   }
        if display:
            print('Signal information:\n\t')
            print(hit_dict)
        
        return hit_dict
    
    def print(self):
        hit_info = 'Signal information:\n\t'
        hit_info += f'hash_seg_x = {self.hash_seg_x}' + '\n\t'
        hit_info += f'hash_seg_y = {self.hash_seg_y}' + '\n\t'
        hit_info += f'z = {self.z}' + '\n\t'
        hit_info += f'time = {self.time}' + '\n\t'
        hit_info += f'seg_ix = {self.seg_ix}' + '\n\t'
        hit_info += f'rdrift = {self.rdrift}' + '\n\t' 
        hit_info += f'is_muon = {self.is_muon}' + '\n'
        print(hit_info)