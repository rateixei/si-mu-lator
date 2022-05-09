import numpy as np

# geometry utilities (to avoid slow Sympy distance between point and line calculation)

def tline(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def distpoint2line(xseg, hit):
    p = np.array([ xseg.line.p1.x, xseg.line.p1.y, xseg.line.p1.z])
    q = np.array([ xseg.line.p2.x, xseg.line.p2.y, xseg.line.p2.z])
    r = np.array([ hit.x, hit.y, hit.z])
    dvecsq = tline(p, q, r)*(p-q)+q-r
    return sum(dvecsq*dvecsq)**0.5

