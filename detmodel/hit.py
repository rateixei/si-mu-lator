import numpy as np
import sympy

class Hit:
    def __init__(self, x, y, z, time, seg_ix, rdrift, is_muon):
        self.x = x
        self.y = y
        self.z = z
        self.time = time
        self.seg_ix = seg_ix
        self.rdrift = rdrift
        self.is_muon = is_muon
    
    def point(self):
        return sympy.Point3D(self.x, self.y, self.z)
    
    def print(self):
        hit_info = 'Hit information:\n\t'
        hit_info += f'X = {self.x}' + '\n\t'
        hit_info += f'Y = {self.y}' + '\n\t'
        hit_info += f'Z = {self.z}' + '\n\t'
        hit_info += f'T = {self.time}' + '\n\t'
        hit_info += f'ix = {self.seg_ix}' + '\n\t'
        hit_info += f'R = {self.rdrift}' + '\n\t'
        hit_info += f'Is from muon = {self.is_muon}' + '\n'
        print(hit_info)