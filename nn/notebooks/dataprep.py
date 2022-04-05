import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py

def make_data_matrix(all_files, max_files=50, masking99 = False, sort_by='none'):
    
    data = {}
    signals = []
    sig_keys = []

    for ifile in range(min(max_files, len(all_files))):
        fname = all_files[ifile]
        print(fname)
    
        this_file = h5py.File( fname, 'r' )
        
        if ifile == 0:
            sig_keys = [ vv.decode("utf-8") for vv in np.array(this_file['signal_keys']) ]
            print(sig_keys)
    
        signals.append( np.array( this_file['signals'] ) )
        
        for kk in this_file.keys():
            if 'mu' in kk or '_n_' in kk:
                if kk not in data:
                    data[kk] = np.array( this_file[kk] )
                else:
                    data[kk] = np.concatenate( [data[kk], np.array( this_file[kk] )] )
    
    max_hits = max( [ als.shape[1] for als in signals ] )
    features = max( [ als.shape[2] for als in signals ] )
    tot_evts = np.sum( [als.shape[0] for als in signals] )
    
    dmat = -99*np.ones((tot_evts, max_hits, features))
    Y_mu = np.array(data['ev_mu_phi'] >= 0, dtype=int)
    
    tot_add = 0
    for fevt in signals:
        n_entries = fevt.shape[0]
        dmat[tot_add:tot_add+n_entries,:,:] = fevt[:]
        tot_add += n_entries
        # for evt in fevt:
            # hits_to_add = evt#[np.where(evt[:,sig_keys.index('is_muon')] > -990)][:]
            
            # if 'none' not in sort_by and sort_by in sig_keys:
                # hits_to_add = hits_to_add[ np.argsort( hits_to_add[:,sig_keys.index(sort_by)] ) ]
            
            # dmat[tot_add, :hits_to_add.shape[0],:] = hits_to_add
            # tot_add += 1
        
    Y_hit = dmat[:,:,0]
    
#     if masking99:
#         #for masking -99

#         for isig in range(dmat.shape[0]):
#             for jsig in range(dmat.shape[1]):
#                 if dmat[isig,jsig,8] == 0:
#                     dmat[isig,jsig] = np.full_like(dmat[isig,jsig], -99, dtype=int)
#                     Y_hit[isig,jsig] = -99
                
    Y = np.empty((Y_hit.shape[0], Y_hit.shape[1]+1), dtype=int)
    Y[:,0] = Y_mu[:]
    Y[:,1:] = Y_hit[:]
    
    return (data, dmat, Y, Y_mu, Y_hit, sig_keys)