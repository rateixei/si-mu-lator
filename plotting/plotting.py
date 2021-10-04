import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import sympy

def plot_det_volume(det, ax, draw_muon=False):
    
    edges = {'top':0, 'bottom':0, 'left':0, 'right':0}
    
    max_x = 0
    max_y = 0
    max_z = 0
    
    for ip,p in enumerate(det.planes):
        pz = p.z
        
        for e in edges:
            edges[e] = p.get_edge(e)
            
        verts = [ [   [-0.5*p.sizes['x'], -0.5*p.sizes['y'], pz],
                      [0.5*p.sizes['x'], -0.5*p.sizes['y'], pz],
                      [0.5*p.sizes['x'], 0.5*p.sizes['y'], pz],
                      [-0.5*p.sizes['x'], 0.5*p.sizes['y'], pz]   ]
                ]
        
        ax.add_collection3d(Poly3DCollection(verts, facecolors='gray', linewidths=1, edgecolors='gray', alpha=.10))
        
        if 0.5*p.sizes['x'] > max_x: max_x = 0.5*p.sizes['x']
        if 0.5*p.sizes['y'] > max_y: max_y = 0.5*p.sizes['y']
        if pz > max_z: max_z = pz
            
        for slx in p.seg_lines['x']:
#             print(slx.line)
            
            inter_left  = list(slx.line.intersect( edges['left'] ))[0]
            inter_right = list(slx.line.intersect( edges['right'] ))[0]
            
            lcolor = 'gray'
            alpha = 0.3
            
            if slx.is_sig:
                lcolor = 'C0'
                alpha = 0.6
                if slx.sig.is_muon:
                    lcolor='green'
                    alpha = 1
            
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
                alpha = 0.3
                if slx.is_sig:
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
        plane_init = sympy.Plane(sympy.Point3D(0,0,0) , normal_vector=(0,0,1))
        plane_final = sympy.Plane(sympy.Point3D(0,0,max_z+1) , normal_vector=(0,0,1))
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
    ax.set_zlim(0, max_z+1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')