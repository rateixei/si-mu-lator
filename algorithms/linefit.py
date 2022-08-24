import numpy as np

def get_mm_line_fit(hits, sig_keys, is_perfect=False, only_x=False):
    
    ## extract quantities
    hit_types = hits[:, sig_keys.index('is_muon')]
    valid_hits = hit_types > -90
    
    if valid_hits.sum() < 2:
        return [np.nan,np.nan,np.nan]
    
    ptype = hits[:, sig_keys.index('ptype')]
    ptilt = hits[:, sig_keys.index('ptilt')]
    
    is_mm = ptype == 0
    is_mmx = (ptype==0)*(np.abs(ptilt) < 1e-6)
    is_mmuv = (ptype==0)*(np.abs(ptilt) > 1e-6)
    is_stgc = ptype == 2
    
    zs = hits[:, sig_keys.index('z')]
    xs = hits[:, sig_keys.index('projX_at_middle_x')]
    xends = hits[:, sig_keys.index('projX_at_rightend_x')]
    
    xs[ptype==2] = hits[ptype==2, sig_keys.index('x')]

    unc_xs = np.zeros_like(xs)
    unc_xs[is_mmx] = 0.45
    unc_xs[is_mmuv] = 2*np.abs( xs[is_mmuv] - xends[is_mmuv] )
    unc_xs[is_stgc] = 0.2
    
    # print(zs)
    # print(xs)
    # print(unc_xs)
    
    ## only save the ones with valid hits
    indx_hit = None
    
    if is_perfect:
        indx_hit = (hit_types == 1)
    else:
        indx_hit = (valid_hits == 1)
    
    if only_x:
        indx_hit = indx_hit * ( is_mmx )
        
    if indx_hit.sum() < 2:
        return [np.nan,np.nan,np.nan]
    
    zs = zs[ indx_hit ]
    xs = xs[ indx_hit ]
    unc_xs = unc_xs[ indx_hit ]
    
    # print(zs)
    # print(xs)
    # print(unc_xs)
    # print(ptype[indx_hit])
    # print(ptilt[indx_hit])
    
    ## calculate coefficients
    _c1 = ( zs**2/unc_xs**2 ).sum() #beta
    _c2 = ( zs/unc_xs**2 ).sum() #gamma
    _c3 = ( zs*xs/unc_xs**2 ).sum() #omega
    _c4 = ( 1./unc_xs**2 ).sum() #lambda
    _c5 = ( xs/unc_xs**2 ).sum() #rho
        
    ## calculate line parameters
    A = ( _c5*_c2 - _c3*_c4 ) / ( _c2**2 - _c1*_c4 )
    B = ( _c3 - A*_c1 ) / _c2
    
    ## calculate chi2 value
    Chi2 = ( ( A*zs + B - xs )**2 / unc_xs**2 ).sum()
    
#     print(A, B, Chi2)
#     print()
#     print()
        
    return [A, B, Chi2]
    

def get_fits(events, sig_keys, overwrite=False, is_perfect=False, only_x=False):
    
    fits = np.zeros( (events.shape[0], 3) )
    for iev,hits in enumerate(events):
        lfit = get_mm_line_fit(hits, sig_keys, is_perfect=is_perfect, only_x=only_x)
        fits[iev,:] = lfit
        
        # if iev > 10: break
    
    return fits