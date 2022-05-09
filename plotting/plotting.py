import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import sympy
from detmodel.plane import DetType
from detmodel import util

def plot_det_volume(det, ax, draw_muon=False):
    
    edges = {'top':0, 'bottom':0, 'left':0, 'right':0}
    
    max_x = 0
    max_y = 0
    max_z = 0
    min_z = 999999
    
    for ip,p in enumerate(det.planes):
        pz = p.z
        
        for e in edges:
            edges[e] = p.get_edge(e)
            
        verts = [ [   [-0.5*p.sizes['x'], -0.5*p.sizes['y'], pz],
                      [0.5*p.sizes['x'], -0.5*p.sizes['y'], pz],
                      [0.5*p.sizes['x'], 0.5*p.sizes['y'], pz],
                      [-0.5*p.sizes['x'], 0.5*p.sizes['y'], pz]   ]
                ]
        
        ax.add_collection3d(Poly3DCollection(verts, facecolors='gray', linewidths=0, edgecolors='gray', alpha=.1))
        
        if 0.5*p.sizes['x'] > max_x: max_x = 0.5*p.sizes['x']
        if 0.5*p.sizes['y'] > max_y: max_y = 0.5*p.sizes['y']
        if pz > max_z: max_z = pz
        if pz < min_z: min_z = pz
            
        for islx, slx in enumerate(p.seg_lines['x']):
#             print(slx.line)
            
            inter_left  = list(slx.line.intersect( edges['left'] ))[0]
            inter_right = list(slx.line.intersect( edges['right'] ))[0]
            
            lcolor = 'gray'
            alpha = 0.2
            
            if slx.is_sig:
                lcolor = 'm'
                alpha = 0.7
                if slx.sig.is_muon:
                    lcolor='green'
                    alpha = 1
                    print("Found muon signal, plane ", ip, " xseg ", islx, " time ",  slx.sig.time )
                else:
                    print("Found bkg signal, plane ", ip, " xseg ", islx, " time ",  slx.sig.time )

            ax.plot3D (
                xs=[ float(inter_left.x), float(inter_right.x) ],
                ys=[ float(inter_left.y), float(inter_right.y) ],
                zs=[ pz,pz ],
                color=lcolor, alpha=alpha
            )
        
        if len(p.seg_lines['y']) > 1:
            for sly in p.seg_lines['y']:
#                 print(sly.line)

                inter_top  = list(sly.line.intersect( edges['top'] ))[0]
                inter_bottom = list(sly.line.intersect( edges['bottom'] ))[0]
                
                lcolor = 'gray'
                alpha = 0.2
                if slx.is_sig:
                    lcolor = 'C0'
                    alpha = 0.7
                    if slx.sig.is_muon:
                        lcolor='green'
                        alpha = 1
            
                ax.plot3D (
                    xs=[ float(inter_bottom.x), float(inter_top.x) ],
                    ys=[ float(inter_bottom.y), float(inter_top.y) ],
                    zs=[ pz,pz ],
                    color=lcolor, alpha=alpha
                )
                
    if draw_muon:
        plane_init = sympy.Plane(sympy.Point3D(0,0,min_z-1,evaluate=False) , normal_vector=(0,0,1))
        plane_final = sympy.Plane(sympy.Point3D(0,0,max_z+1,evaluate=False) , normal_vector=(0,0,1))
        intersect_init = list(plane_init.intersection( det.mymu.line ))
        intersect_final = list(plane_final.intersection( det.mymu.line ))
        print(intersect_init, intersect_final)
        ax.plot3D(
            xs=[ float(intersect_init[0].x), float(intersect_final[0].x) ],
            ys=[ float(intersect_init[0].y), float(intersect_final[0].y) ],
            zs=[ float(intersect_init[0].z), float(intersect_final[0].z) ],
            color='red'
        )

    ax.set_xlim(-max_x*1, max_x*1)
    ax.set_ylim(-max_y*1, max_y*1)
    ax.set_zlim(min_z-1, max_z+1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    
def plot_det_xz(det, ax, draw_muon=False, draw_allhits=False):

    # Plot XZ projection for MDT chamber with vertical tubes (along y axis)

    edges = {'top':0, 'bottom':0, 'left':0, 'right':0}
    pitch = det.specs['det_width_x']/det.specs['det_n_x_seg']  # works for MDT

    max_x = 0
    max_y = 0
    max_z = 0
    min_z = 99999

    # Loop over detector planes
    for ip,p in enumerate(det.planes):
        pz = p.z

        for e in edges:
            edges[e] = p.get_edge(e)

        # Draw lines at location of detector planes
        xs=[ -0.5*p.sizes['x'], 0.5*p.sizes['x'] ]
        zs=[ pz, pz ]
        if p.p_type != DetType.MDT:
            pitch = p.sizes['x'] / len(p.segmentations['x'])

        if 0.5*p.sizes['x'] > max_x: max_x = 0.5*p.sizes['x']
        if pz > max_z: max_z = pz
        if pz < min_z: min_z = pz

        # Loop over the detector elements in this plane and draw circles for MDT (rectangles otherwise)
        for islx, slx in enumerate(p.seg_lines['x']):

            # 2 lines below are slow, not necessary in this projection that assumes y=0
            #inter_left  = list(slx.line.intersect( edges['left'] ))[0]
            #inter_right = list(slx.line.intersect( edges['right'] ))[0]

            lcolor = 'gray'
            alpha = 0.4
            # first point on the detector element segment corresponds to y=0
            wirepos = np.array([slx.line.p1.x, slx.line.p1.z])
            if p.p_type != DetType.MDT:
                # strip gap is 0.5 mm and strip pitch is 3.2 mm for sTGC
                rec = plt.Rectangle(wirepos-(0.42*pitch,0), 0.84*pitch, 0.8, color=lcolor, alpha=alpha)
                ax.add_patch(rec)
            else:
                cir = plt.Circle(wirepos, 0.5*pitch, color=lcolor, alpha=alpha, fill=False)
                ax.add_patch(cir)
                ax.plot(wirepos[0], pz, marker='o', markersize=2, color=lcolor, alpha=alpha)

        # Loop over the hits in this plane
        # Keep track of signal hits (i.e. hits with smallest drift radius in a given tube)
        # to remove non signal hits if desired
        list_seg_ix = [] # store signal hits
        for ihit,hit in enumerate(p.hits):

            # Here we rely on the hits being ordered from smallest to largest radius beforehand
            if not draw_allhits and hit.seg_ix in list_seg_ix:  # currently only works for MDT
                print('skip hit',ihit)
                continue

            list_seg_ix.append(hit.seg_ix)
            if hit.is_muon:  # muon hit
                lcolor = 'green'
                alpha = 1
            else:            # noise hit
                lcolor = 'm'
                alpha = 0.7

            rdrift = 0.20  # small fixed radius to mark hit position 
            hitpos = (0,0)
            if p.p_type == DetType.MM:
                #hit_ix = np.argmin( [ xseg.line.distance(hit.point()) for xseg in p.seg_lines['x'] ] )
                hit_ix = np.argmin( [ util.distpoint2line(xseg, hit) for xseg in p.seg_lines['x'] ] )
                if draw_allhits or p.seg_lines['x'][hit_ix].is_sig:
                    hitpos = sympy.Point(p.seg_lines['x'][hit_ix].line.p1.x, hit.z, evaluate=False)
                else:
                    continue
            elif p.p_type == DetType.MDT:
                rdrift = hit.rdrift
                hitpos = sympy.Point(p.seg_lines['x'][hit.seg_ix].line.p1.x, hit.z, evaluate=False)
            elif p.p_type == DetType.STGC:
                if hit.is_muon:
                    hitpos = sympy.Point(hit.x, hit.z, evaluate=False)
                else: # place noise hit in middle of strip
                    #hit_ix = np.argmin( [ abs(xseg-hit.x) for xseg in p.seg_mids['x'] ] )
                    hit_ix = np.argmin( [ util.distpoint2line(xseg, hit) for xseg in p.seg_lines['x'] ] )
                    hitpos = sympy.Point(p.seg_lines['x'][hit_ix].line.p1.x, hit.z, evaluate=False)
            
            cir = plt.Circle(hitpos, rdrift, color=lcolor, linewidth=2, fill=False)
            ax.add_patch(cir)

    if draw_muon:
        plane_init = sympy.Plane(sympy.Point3D(0,0,-10,evaluate=False) , normal_vector=(0,0,1))
        plane_final = sympy.Plane(sympy.Point3D(0,0,max_z+25,evaluate=False) , normal_vector=(0,0,1))
        intersect_init = list(plane_init.intersection( det.mymu.line ))
        intersect_final = list(plane_final.intersection( det.mymu.line ))
        print(intersect_init, intersect_final)
        xs=[ float(intersect_init[0].x), float(intersect_final[0].x) ]
        zs=[ float(intersect_init[0].z), float(intersect_final[0].z) ]
        ax.plot( xs, zs, color='red' )

    #ax.set_xlim(-max_x*1, max_x*1)
    #ax.set_ylim(-10, max_z+25)
    ax.set_xlim(-max_x*1, max_x*1)
    ax.set_ylim((min_z-20)*1, (max_z+20)*1)

    ax.set_xlabel('x (mm)', fontsize=15)
    ax.set_ylabel('z (mm)', fontsize=15)

    