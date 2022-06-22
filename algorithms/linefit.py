import numpy as np

def get_mm_line_fit(hits, sig_keys):
    
    ## extract quantities
    zs = hits[:, sig_keys.index('z')]
    xs = hits[:, sig_keys.index('projX_at_middle_x')]
    unc_xs = 2*np.abs( hits[:, sig_keys.index('projX_at_middle_x')] - 
                      hits[:, sig_keys.index('projX_at_rightend_x')] )
    unc_xs[unc_xs < 0.45] = 0.45
    
    ## only save the ones with valid hits
    zs = zs[ hits[:, sig_keys.index('is_muon')] > -1 ]
    xs = xs[ hits[:, sig_keys.index('is_muon')] > -1 ]
    unc_xs = unc_xs[ hits[:, sig_keys.index('is_muon')] > -1 ]
    
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
        
    return [A, B, Chi2]
    

def get_fits(events, sig_keys, overwrite=False):
    
    fits = np.zeros( (events.shape[0], 3) )
    for iev,hits in enumerate(events):
        lfit = get_mm_line_fit(hits, sig_keys)
        fits[iev,:] = lfit
    
    return fits
    

    