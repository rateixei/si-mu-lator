import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import yaml

def make_data_matrix(all_files, max_files=50, masking99 = False, sort_by='none'):
    print('~~ Reading data... ~~')
    
    data = {}
    signals = []
    sig_keys = []

    for ifile in tqdm(range(min(max_files, len(all_files)))):
        fname = all_files[ifile]
    
        this_file = h5py.File( fname, 'r' )
        
        if ifile == 0:
            sig_keys = [ vv.decode("utf-8") for vv in np.array(this_file['signal_keys']) ]
            
    
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
    for ifevt in range(len(signals)):
        fevt = signals[ifevt]
        
        n_entries = fevt.shape[0]
        n_hits = fevt.shape[1]
        dmat[tot_add:tot_add+n_entries,:n_hits,:] = fevt[:]
        tot_add += n_entries
        # for evt in fevt:
            # hits_to_add = evt#[np.where(evt[:,sig_keys.index('is_muon')] > -990)][:]
            
            # if 'none' not in sort_by and sort_by in sig_keys:
                # hits_to_add = hits_to_add[ np.argsort( hits_to_add[:,sig_keys.index(sort_by)] ) ]
            
            # dmat[tot_add, :hits_to_add.shape[0],:] = hits_to_add
            # tot_add += 1
        
    Y_hit = dmat[:,:,0]
                
    Y = np.empty((Y_hit.shape[0], Y_hit.shape[1]+1), dtype=int)
    Y[:,0] = Y_mu[:]
    Y[:,1:] = Y_hit[:]
    
    print('~~ Calculating occupancy information... ~~')
    data['n_sig_mmx']  = np.zeros(tot_evts)
    data['n_sig_mmu']  = np.zeros(tot_evts)
    data['n_sig_mmv']  = np.zeros(tot_evts)
    data['n_sig_mm']   = np.zeros(tot_evts)
    data['n_sig_stgc'] = np.zeros(tot_evts)
    data['n_sig_mdt']  = np.zeros(tot_evts)
    for iev in tqdm(range(dmat.shape[0])):
        hits_tilt = dmat[iev,:,sig_keys.index('ptilt')]
        is_mm = dmat[iev,:,sig_keys.index('ptype')] == 0
        data['n_sig_mmu'][iev]  = ((hits_tilt > 0)&(is_mm)).sum()
        data['n_sig_mmv'][iev]  = ((hits_tilt < 0)&(hits_tilt > -90)&(is_mm)).sum()
        data['n_sig_mmx'][iev]  = (is_mm).sum() - data['n_sig_mmu'][iev] - data['n_sig_mmv'][iev]
        data['n_sig_mm'][iev]   = (dmat[iev,:,sig_keys.index('ptype')] == 0).sum()
        data['n_sig_stgc'][iev] = (dmat[iev,:,sig_keys.index('ptype')] == 2).sum()
        data['n_sig_mdt'][iev]  = (dmat[iev,:,sig_keys.index('ptype')] == 1).sum()
    

    print('!!')
    print(f'I read {dmat.shape[0]} events, of which {(Y_mu==1).sum()} have muon and {(Y_mu==0).sum()} do not')
    print('!!')
    return (data, dmat, Y, Y_mu, Y_hit, sig_keys)

def training_prep(X, sig_keys):
    print('~~ Preparing padded matrix ~~')
    
    X_out = np.zeros((X.shape[0], X.shape[1], X.shape[2]+1))
    
    if 'is_signal' not in sig_keys:
        sig_keys.append('is_signal')
    else:
        print('Data already prepared?')
        # return X
    
    for iev in tqdm(range( X.shape[0] )):
        ev = X[iev]
        
        hit_types = ev[:,sig_keys.index('is_muon')]
        valid_hits = hit_types > -90
        
        if valid_hits.sum() < 1: continue
        
        X_out[iev,valid_hits,:-1] = np.copy(ev[valid_hits,:])
        X_out[iev,valid_hits, -1] = np.ones_like(X_out[iev,valid_hits, -1])
        
        delta_z_valid_hits = X_out[iev,valid_hits,sig_keys.index('z')].max() - X_out[iev,valid_hits,sig_keys.index('z')].min()
        
        if delta_z_valid_hits < 1e-5:
            delta_z_valid_hits = X_out[iev,valid_hits,sig_keys.index('z')].max()
            
        X_out[iev,valid_hits,sig_keys.index('z')] = ( X_out[iev,valid_hits,sig_keys.index('z')] - delta_z_valid_hits)/delta_z_valid_hits
        
        X_out[iev,valid_hits,sig_keys.index('ptilt')] = X_out[iev,valid_hits,sig_keys.index('ptilt')]/0.02618
    
    print('Output data matrix shape:', X_out.shape)
    return X_out

def detector_matrix(X, sig_keys, detcard):
    print('~~ Preparing detector-based data matrix ~~')
    print('Using detector card:', detcard)
    
    specs = 0
    with open(detcard) as f:
        # fyml = yaml.load(f, Loader=yaml.FullLoader) ## only works on yaml > 5.1
        fyml = yaml.safe_load(f)
        specs = fyml['detector']
    
    n_planes = len(specs['planes'])
    z_hit_planes = []
    t_width_planes = []
    for p_name in specs['planes']:
        p = specs['planes'][p_name]
        n_hits = 0
        if 'max_hits' in p:
            n_hits = p['max_hits']
        elif p['type'] == 'mm':
                print('If this is a MM layer, you need to limit the number of hits per layer')
                return -1
        else:
            n_hits = 1
            
        z_hit_planes += n_hits*[p['z']]
        p_t_width = p['width_t'] if 'width_t' in p else specs['det_width_t']
        t_width_planes += n_hits*[p_t_width]
    
    X_out = np.zeros( (X.shape[0], len(z_hit_planes), X.shape[2]+1) )
    if 'is_signal' not in sig_keys:
        sig_keys.append('is_signal')
    
    # for iev,ev in enumerate(X):
    for iev in tqdm(range( X.shape[0] )):
        ev = X[iev]
        allhit=0
        
        for ipz,pz in enumerate(z_hit_planes):
            X_out[iev,ipz,sig_keys.index('z')]=pz

            for ihit,hit in enumerate(X[iev][allhit:]):
                if hit[sig_keys.index('z')]==pz:
                    X_out[iev,ipz,:-1] = hit
                    X_out[iev,ipz,-1] = 1
                    X_out[iev,ipz,sig_keys.index('time')]/t_width_planes[ipz]
                    allhit+=1
                    break
                    
            
    max_dz = max(z_hit_planes) - min(z_hit_planes)
    avg_dz = 0.5*(max(z_hit_planes) + min(z_hit_planes))
    
    X_out[:,:,sig_keys.index('z')] = (X_out[:,:,sig_keys.index('z')] - avg_dz)/(0.5*max_dz)
    X_out[:,:,sig_keys.index('ptilt')] = (X_out[:,:,sig_keys.index('z')] - avg_dz)/(0.5*max_dz)
    
    print('Output data matrix shape:', X_out.shape)
    return X_out