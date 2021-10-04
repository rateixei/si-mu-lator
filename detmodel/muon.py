import numpy as np
import sympy

class Muon:
    def __init__(self, x, y, theta, phi, time=0):
        self.org_x = x
        self.org_y = y
        self.org_z = 0
        
        self.r = 1.
        self.theta = theta
        self.phi = phi
        self.end_x = self.r*np.cos(self.phi)*np.sin(self.theta) + self.org_x
        self.end_y = self.r*np.sin(self.phi)*np.sin(self.theta) + self.org_y
        self.end_z = self.r*np.cos(self.theta)

        ## muon is generated from 0,0,0 (IP), but this can be changed
        self.point_org = sympy.Point3D( (self.org_x,self.org_y,0) )
        self.point_end = sympy.Point3D( (self.end_x,self.end_y,self.end_z) )
        self.line = sympy.Line3D(self.point_org, self.point_end)
        self.time = time
