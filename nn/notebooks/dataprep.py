import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py

def make_data_matrix(all_files, max_files=50):
    
    data = {}
    signals_mu = []
    signals_nomu = []

    for ifile in range(min(max_files, len(all_files))):
        fname = all_files[ifile]
    
        this_file = h5py.File( fname, 'r' )
    
        this_data = {}
    
        for kk in this_file.keys():
        
            if kk not in data:
                data[kk] = np.array( this_file[kk] )
            elif 'ev_' in kk:
                data[kk] = np.concatenate( [data[kk], np.array( this_file[kk] )] )
            elif 'signals' in kk:
                if 'NoMuon' in fname:
                    signals_nomu.append( np.array( this_file[kk] ) )
                else:
                    signals_mu.append( np.array( this_file[kk] ) )
    
    n_signals_mu = len(signals_mu)
    n_signals_nomu = len(signals_nomu)
    
    max_hits = max( [ als.shape[1] for als in [*signals_mu, *signals_nomu] ] )
    tot_evnts = np.sum( [ als.shape[0] for als in [*signals_mu, *signals_nomu] ] )
    features = max( [ als.shape[2] for als in [*signals_mu, *signals_nomu] ] )
    
    dmat = np.zeros((tot_evnts, max_hits, features))
    Y_mu = np.zeros((tot_evnts))

    tot_add = 0
    added_mu = 0
    
    for als in signals_mu:
        this_add = als.shape[0]
        dmat[added_mu:added_mu+this_add, :als.shape[1],:] = als[:]
        added_mu += this_add
        tot_add += this_add
        
    Y_mu[0:added_mu] = np.ones_like(Y_mu[0:added_mu])
    added_nomu = 0
    
    for als in signals_nomu:
        this_add = als.shape[0]
        dmat[tot_add:tot_add+this_add, :als.shape[1],:] = als[:]
        added_nomu += this_add
        tot_add += this_add
        
    assert added_mu+added_nomu == tot_evnts
    
    Y_hit = dmat[:,:,10]
    
    #for masking -99

    for isig in range(dmat.shape[0]):
        for jsig in range(dmat.shape[1]):
            if dmat[isig,jsig,8] == 0:
                dmat[isig,jsig] = np.full_like(dmat[isig,jsig], -99, dtype=int)
                Y_hit[isig,jsig] = -99
                
    Y = np.empty((Y_hit.shape[0], Y_hit.shape[1]+1))
    Y[:,0] = Y_mu[:]
    Y[:,1:] = Y_hit[:]
    
    return (dmat, Y, Y_mu, Y_hit)