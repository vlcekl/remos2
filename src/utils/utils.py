import os
import string as s
import re
import numpy as np
from collections import Counter
from itertools import product, combinations
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import h5py

def multinomial_covariance(probs):

    probs = np.array(probs)

    mat = np.empty((len(probs), len(probs)), dtype=float)

    for i, j in product(range(len(probs)), repeat=2):
        if i == j:
            mat[i, i] = probs[i]*(1-probs[i])
        else:
            mat[i, j] = -probs[i]*probs[j]

    return mat


def corr_cov(data_matrix, output='corr'):
    """
    Calculates correlation matrix for n_exp multinomial experiments with n samples each
    resulting in k possible states.

    Parameters
    ----------
    data_matrix: ndarray((n_exp, k), dtype=float)
                 data matrix of n_exp experiments with results in k histogram bins (states)
    output: str
            defines whether correlation ('corr' - default) or covariance ('cov') matrix is returned

    Returns
    -------
    corr_mat: ndarray((k, k), dtype=float)
            correlation matrix 
                 
    """
    
    n_exp = data_matrix.shape[0]

    # calculate averages of the columns
    averages = np.stack([np.mean(data_matrix, axis=0)]*n_exp, axis=0)
    
    # data deviations from averages
    data_centered = data_matrix - averages
    
    # calculate covariance matrix
    cov_mat = data_centered.T.dot(data_centered)/(n_exp - 1.)
    
    if output == 'cov':
        return cov_mat

    # inverse square root variance diagonal matrix
    var_srinv = np.sqrt(np.diag(1.0/np.diag(cov_mat)))

    # correlation matrix
    corr_mat = var_srinv.dot(cov_mat).dot(var_srinv)
    
    return corr_mat



def n_effective(data_matrix):
    """
    Calculate the effective number of samples from a data matrix of a multinomial distribution.
    Calls corr_cov function to calculate covariance matrix
    
    Parameters
    ----------
    data_matrix: ndarray((n_exp, k), dtype=float)
                 data matrix of n_exp experiments with results in k histogram bins (states)

    Returns
    -------
    n_eff: float
           effective number of samples
    """
    
    v = np.diag(corr_cov(data_matrix, output='cov'))
    p = np.mean(data_matrix, axis=0)
    
    # vector of effective sample numbers for each histogram bin
    n_eff_vec = p*(1-p)/v
    
    # mean value (should it be minimum value?)
    n_eff = np.mean(n_eff_vec)
    #n_eff = np.min(n_eff_vec)
    
    return n_eff


def dist_calc(pt, pts):
    """Distance between a point 'pt' and a list of points 'pts'"""
    
    distances = []
    for ind in range(pts.shape[0]):
        distances.append((np.sqrt((pt[0] - pts[ind,0])**2 + (pt[1] - pts[ind,1])**2), ind))
    
    return distances

def hexagon(pos):
    """
    Scale neighbor positions so that they have the same distance from their center of mass.
    """
    
    com_x = sum([p[0] for p in pos])/len(pos)
    com_y = sum([p[1] for p in pos])/len(pos)
    
    pos_hex = []
    for p in pos:
        dx = p[0] - com_x
        dy = p[1] - com_y
        r = (dx*dx + dy*dy)**0.5
        pos_hex.append([dx/r, dy/r])
    
    return pos_hex

def class_triplets(c_id, pos, ids):
    """
    Assign unique identifiers to the neighbor configurations
    """
    
    # scale to make the neighbor hexagon more regular
    pos = hexagon(pos)

    
    # collect pair distances between neighbors and add the pair type (0, 1, 2)
    pair_list = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            r = np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            pair_list.append((r, ids[i]+ids[j]))
    
    assert len(pair_list) == 15, f'Wrong number of pair distances: {len(pair_list)}'

    
    # number of Re atoms in the neighbor list
    n_atom = sum([1 for elem in ids if elem == 1])
    
    # sort the pair list according to distances
    pair_list = sorted(pair_list, key=lambda x: x[0])
    
    triples = []
    for p in pair_list[0:6]:
        triples.append(c_id + p[1])

    #print(triples)

    return triples

def classify(pos, ids):
    """
    Assign unique identifiers to the neighbor configurations
    """
    
    # scale to make the neighbor hexagon more regular
    pos = hexagon(pos)
    
    # collect pair distances between neighbors and add the pair type (0, 1, 2)
    pair_list = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            r = np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            pair_list.append((r, ids[i]+ids[j]))
    
    assert len(pair_list) == 15, f'Wrong number of pair distances: {len(pair_list)}'
    
    # number of Re atoms in the neighbor list
    n_atom = sum([1 for elem in ids if elem == 1])
    
    # sort the pair list according to distances
    pair_list = sorted(pair_list, key=lambda x: x[0])
    
    # number of Re-Re pairs among the 6 shortest distances distances (ortho)
    n_ortho = sum([1 for elem in pair_list[0:6] if elem[1] == 2])
    
    # number of Re-Re pairs among the 6 medium distances distances (meta)
    n_meta = sum([1 for elem in pair_list[6:12] if elem[1] == 2])
    
    # number of Re-Re pairs among the 3 longest distances distances (para)
    n_para = sum([1 for elem in pair_list[12:15] if elem[1] == 2])

    return [n_atom, n_ortho, n_meta, n_para]

# read reference system parameters into an array
def read_mld(data_paths, read_func):
    pars_ref = {}
    for data_path in data_paths:
        params = read_func(os.path.join(data_path, 'lg.mld'))
        if isinstance(params['ref_params'], dict):
            pars_ref[data_path] = np.array([v for v in params['ref_params'].values()]).flatten(order='F')
            # check if the force field parameters agree between systems
            assert len(set([str(v) for v in pars_ref.values()])) == 1, "Different parameters"
        else:
            pars_ref[data_path] = params['ref_params']

    return pars_ref[data_paths[0]]

# read lattice simulation results: configuration statistics and energies
def read_outputs(names, dirnames, read_func, fname=None):
    trjs = {}
    for name, dname in zip(names, dirnames):
        if fname:
            trjs[name] = read_func(os.path.join(dname, fname))
        else:
            trjs[name] = read_func(dname)
    return trjs

# filter the results (discard initial imin number of configurations)
def trajectory_range(trjs, imin=10):

    for key in trjs.keys():
        trjs[key]['energy'] = trjs[key]['energy'][imin:]
        trjs[key]['temp'] = trjs[key]['temp'][imin:]
        trjs[key]['interaction_stats'] = trjs[key]['interaction_stats'][imin:]
        trjs[key]['config_stats'] = trjs[key]['config_stats'][imin:]

    return trjs

def make_reference_counts(trjs, trans_hist):
    """
    For given trajectories, convert to histograms of symmetrized configurations
    """
    # group symmetric configurations collected from a simulated system based on their unique label
    hsv = {}
    for name, trj in trjs.items():
        # convert to the histograms of symmetrized configurations
        old_hist = np.array(trj['config_stats'])
        hsv[name] = trans_hist.dot(old_hist.T).T
        # normalize = calculate relative frequencies of symmetrized configurations
        for i in range(hsv[name].shape[0]):
            sm = np.sum(hsv[name][i,:])
            hsv[name][i,:] = hsv[name][i,:]#/sm

    return hsv

def make_reference_histograms(trjs, trans_hist):
    """
    For given trajectories, convert to histograms of symmetrized configurations
    """
    # group symmetric configurations collected from a simulated system based on their unique label
    hsv = {}
    for name, trj in trjs.items():
        # convert to the histograms of symmetrized configurations
        old_hist = np.array(trj['config_stats'])
        hsv[name] = trans_hist.dot(old_hist.T).T
        # normalize = calculate relative frequencies of symmetrized configurations
        for i in range(hsv[name].shape[0]):
            sm = np.sum(hsv[name][i,:])
            hsv[name][i,:] = hsv[name][i,:]/sm

    return hsv

# filter the results (discard initial configurations)
def make_energy_statistics(trjs):
    hsu = {}
    for name, trj in trjs.items():
        # flatten the array column by column (Fortran style 'F')
        hsu[name] = np.array([np.array(ar).flatten(order='F') for ar in trj['interaction_stats']])
    return hsu

# get relative frequencies of symmetrized surface local configurations averaged over all 'image' configurations
def make_average_histograms(names):
    hsv_ave = {}
    for name in names:
        histave = []
        for i in range(hsv[name].shape[1]):
            histave.append(sum(hsv[name][:,i])/float(hsv[name].shape[0]))
        hsv_ave[name] = np.array(histave)

    return hsv_ave

# Assemble target data structures
def make_targets(gsv, num_samples):
    targets = {}
    for name in gsv.keys():
        # target
        targ = {}
        targ['config_stats'] = gsv[name]
        targ['weight'] = num_samples[name]/sum(num_samples.values())
        targets[name] = targ

    return targets

# Assemble reference data structures
def make_stats(trjs, hsu, hsv):
    stats = {}
    for name in trjs.keys():
        # reference data
        stat = {}
        stat['config_stats'] = hsv[name]
        stat['interaction_stats'] = hsu[name]
        stat['energy'] = np.array(trjs[name]['energy'])
        stat['temp'] = np.array(trjs[name]['temp'])
        stats[name] = stat

    return stats

def get_sd2(data):
    sd2 = data['loss']
    par_in = data['params']
    stat = data['stats']
    target = data['targets']
    loss = sd2(par_in, stat, target)
    return par_in, loss

def optimize_params(data):#par_in, stat, target):
    sd2 = data['loss']
    par_in = data['params']
    stat = data['stats']
    target = data['targets']
    #print('# Start sd2 =', sd2(par_in, stat, target))
    #print('# Starting parameters:', par_in, type(par_in), par_in.shape)
    output = fmin(sd2, par_in, args=(stat, target), maxiter=100000, maxfun=10000, disp=0, full_output=1)
    #print('\n# End sd2 =', output[1])
    p_out = output[0]
    #print('# Final parameters:', p_out)
    return p_out, output[1]

def get_profiles(data, ngrid=50, scale=[1.0, 1.0]):
    scale = np.array(scale)
    
    stats_opt = data[1]['stats']
    targets_opt = data[1]['targets']
    
    s2_prof = np.empty((2*ngrid+1, 2*ngrid+1), dtype=float)
    #s2ref_prof = np.empty((2*ngrid+1, 2*ngrid+1), dtype=float)

    for i, j in product(range(-ngrid, ngrid+1), repeat=2):
        x = float(i)/(ngrid)
        y = float(j)/(ngrid)
        pars = np.array([x, y])*scale
        #s2, s2ref = sd2(pars, stats_opt, targets_opt)
        s2 = sd2(pars, stats_opt, targets_opt)
        s2_prof[i+ngrid, j+ngrid] = s2
        #s2ref_prof[i+ngrid, j+ngrid] = s2ref

    return [data[0], s2_prof]  #, s2ref_prof]

def get_simulation_results(data_dir, names, nsim = 10):
    
    atom_pos = []
    atom_ids = []

    for name in names:
        npy_name = os.path.join(data_dir, name +'_sim.npy')
        mat = np.load(npy_name)
        atom_pos.append(mat[:, nsim:])
        atom_ids.append(mat[:, 0:nsim])
    
    return atom_pos, atom_ids


def get_atom_positions(data_dir, names, thresh):
    
    metal_atoms = []
    intensities = []

    for name in names:
        h5_name = os.path.join(data_dir, name +'.h5')
    
        h5_file = h5py.File(h5_name, 'r+')
    
        atom_pos_grp= h5_file['Measurement_000']['Channel_000']['Atom_Positions']
        atom_centroids_1 = atom_pos_grp['Atom_Centroids_1']
        atom_centroids_2 = atom_pos_grp['Atom_Centroids_2']
        cropped_image = atom_pos_grp['Cropped_Image'][:]
    
        # metal atoms stored only in atom_centroids_1?
        if name in ['re05', 're55']:
            atoms = atom_centroids_1
        else:
            atoms = np.vstack((atom_centroids_1, atom_centroids_2))
        
        metal_atoms.append(atoms)
        intensities.append(cropped_image)
        
        
    # store types of atoms
    atom_pos = []
    atom_ids = []

    for atoms, intensity, thrs in zip(metal_atoms, intensities, thresh):

        Re_atoms = []
        Mo_atoms = []

        for i in range(atoms.shape[0]):
            x, y = atoms[i,:]

            if intensity[int(x), int(y)] >= thrs:
                Re_atoms.append((x, y))
            else:
                Mo_atoms.append((x,y))
    
        Mo_atoms = np.array(Mo_atoms)
        Re_atoms = np.array(Re_atoms)

        all_atoms = np.vstack((Mo_atoms[:], Re_atoms[:])) #ignoring the sulfur atoms for now.
    
        all_atoms_ids = np.zeros(all_atoms.shape[0], dtype=int)
        all_atoms_ids[:Mo_atoms.shape[0]] = 0 #0 = Mo
        all_atoms_ids[Mo_atoms.shape[0]:] = 1 #1 = Re
    
        atom_pos.append(all_atoms)
        atom_ids.append(all_atoms_ids)
    
    return atom_pos, atom_ids, intensities

def collect_triplet_histograms(names, atom_pos, atom_ids, dist_thres):
    
    # Image specific distance thresholds

    triplet_hist = {}
    num_samples = {}

    # cycle over lists of atoms from different images
    for name, apos, aids, dthres in zip(names, atom_pos, atom_ids, dist_thres):

        configs = []
    
        # for each atom, find its neighbors, center it
        for i, (c_pos, c_id) in enumerate(zip(apos, aids)):
        
            x, y = c_pos
            distances = dist_calc([x, y], apos)

            neighbor_i = []
            neighbor_pos = []
            neighbor_id = []
            for k in range(len(distances)):
                if distances[k][0] <= dthres and distances[k][0] > 0:
                    j = distances[k][1]
                    neighbor_i.append(j)
                    neighbor_pos.append(apos[j])
                    neighbor_id.append(aids[j])
        
            if len(neighbor_i) < 6:
                continue
            
            assert len(neighbor_i) == 6, f"Incorrect number of neighbors: {len(neighbor_i)}, {name}"
        
            # classify configurations - assign unique ids based on counts of ortho, meta, and para distances
            #configs.append(tuple([c_id] + classify(neighbor_pos, neighbor_id)))
            configs.extend(class_triplets(c_id, neighbor_pos, neighbor_id))
            
        # get counts of distinct configuraions
        counts = Counter(configs)
        triplet_hist[name] = counts
        num_samples[name] = sum(counts.values())

        print('Image:', name)
        print('Number of configuration types:', len(counts))
        #print('Total number of configurations (atoms with 6 neighbors):', num_samples[name])  #, counts)
        hst = np.zeros(4)
        hst[0] = triplet_hist[name].get(0, 0)
        hst[1] = triplet_hist[name].get(1, 0)
        hst[2] = triplet_hist[name].get(2, 0)
        hst[3] = triplet_hist[name].get(3, 0)
        print(hst/np.sum(hst))
        
    return triplet_hist, num_samples


def collect_target_histograms(names, atom_pos, atom_ids, dist_thres):
    

    # Image specific distance thresholds

    target_hist = {}
    num_samples = {}

    # cycle over lists of atoms from different images
    for name, apos, aids, dthres in zip(names, atom_pos, atom_ids, dist_thres):

        configs = []
    
        # for each atom, find its neighbors, center it
        for i, (c_pos, c_id) in enumerate(zip(apos, aids)):
        
            x, y = c_pos
            distances = dist_calc([x, y], apos)

            neighbor_i = []
            neighbor_pos = []
            neighbor_id = []
            for k in range(len(distances)):
                if distances[k][0] <= dthres and distances[k][0] > 0:
                    j = distances[k][1]
                    neighbor_i.append(j)
                    neighbor_pos.append(apos[j])
                    neighbor_id.append(aids[j])
        
            if len(neighbor_i) < 6:
                continue
            
            assert len(neighbor_i) == 6, f"Incorrect number of neighbors: {len(neighbor_i)}, {name}"
        
            # classify configurations - assign unique ids based on counts of ortho, meta, and para distances
            configs.append(tuple([c_id] + classify(neighbor_pos, neighbor_id)))
            
        # get counts of distinct configuraions
        counts = Counter(configs)
        target_hist[name] = counts
        num_samples[name] = sum(counts.values())

        #print('Image:', name)
        #print('Number of configuration types:', len(counts))
        #print('Total number of configurations (atoms with 6 neighbors):', num_samples[name])  #, counts)
        
    return target_hist, num_samples


def rhenium_concentration(target_hist):
    x_re = {}
    for name in target_hist.keys():
        re_sum = sum([v for k, v in target_hist[name].items() if k[0]==1])
        mo_sum = sum([v for k, v in target_hist[name].items() if k[0]==0])
        c = re_sum/(re_sum + mo_sum)
        x_re[name] = c

    return x_re

def make_transform_matrix(cfg_types, conf_dict):
    """Make transformation matrix from old (128) to new histogram (26)"""

    nrows = len(cfg_types)
    ncols = len(conf_dict)

    th = np.zeros((nrows, ncols), dtype=float)

    for key, val in conf_dict.items():
     # find index for the first occurence of the value
     #row = [i for i, e in enumerate(cfg_types) if e == val][0]
        row = cfg_types.index(val)
        col = key - 1
        th[row, col] = 1.0

    # reorganize matrix (to fix discrepancy in assignment)
    #th[4], th[2] = (th[2], th[4])
    #th[4], th[2] = (th[2], th[4])

    return th


def config_symmetry_numbers(cfg_types):
    """
    Parameters
    ----------
    cfg_types : list of tuples
        sorted list of configuration id tuples
    """
    
    # Neighbor hexagon positions
    neighbor_pos = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [ np.cos(np.pi/3.0),  np.sin(np.pi/3.0)],
        [-np.cos(np.pi/3.0),  np.sin(np.pi/3.0)],
        [ np.cos(np.pi/3.0), -np.sin(np.pi/3.0)],
        [-np.cos(np.pi/3.0), -np.sin(np.pi/3.0)],
    ]

    conf_dict = {}
    ncols = 2**7
    nrows = len(cfg_types)

    th = np.zeros((nrows, ncols), dtype=float)

    i = 0
    for neighbor_id in product((0, 1), repeat=6):
        num_id = tuple(reversed(list(neighbor_id)))
        #nbr_id = neighbor_id

        for c_id in range(2): # pick center atom type                                           
            
            conf_num  = c_id 
            conf_num += num_id[0]*2**1
            conf_num += num_id[1]*2**2
            conf_num += num_id[2]*2**3
            conf_num += num_id[3]*2**4
            conf_num += num_id[4]*2**5
            conf_num += num_id[5]*2**6

            conf_id = tuple([c_id] + classify(neighbor_pos, num_id))
            
            i += 1                                                      
            conf_dict[i] = conf_id

            cg_num = cfg_types.index(conf_id)
            #print('cg_num', conf_id, num_id, conf_num, cg_num)

            th[cg_num, conf_num] = 1.0

        conf_counts = Counter(conf_dict.values())

    return conf_counts, conf_dict, th


def random_config_proba(names, conf_counts, x_re):
    # random configuration probabilities
    ntot = 7 # number of atoms in the configuration

    prob_conf = {}
    for name in names:
        x = x_re[name]
        p_c = {}
        for k, v in conf_counts.items():
        
            # number of Re atoms in the configuration
            n_re = k[0] + k[1]
        
            #probability of a configuration with n_re atoms
            prob = x**n_re * (1.0-x)**(ntot - n_re)
        
            # weight probability by symmetry numbers (v)
            p_c[k] = prob*v
            
        prob_conf[name] = p_c

    return prob_conf

def show_images(names, atom_pos, atom_ids, intensities, symbol_size=7):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi = 96)

    for i, (name, img, apos, aids) in enumerate(zip(names, intensities, atom_pos, atom_ids)):
        ir, ic = divmod(i, 2)
        axs[ir, ic].imshow(img)#, cmap = 'hot')
        axs[ir, ic].scatter(apos[aids==0,1], apos[aids==0,0], c = 'r', s = symbol_size)
        axs[ir, ic].scatter(apos[aids==1,1], apos[aids==1,0], c = 'k', s = symbol_size)
        axs[ir, ic].get_xaxis().set_visible(False)
        axs[ir, ic].get_yaxis().set_visible(False)
        axs[ir, ic].set_title(name)

    plt.tight_layout()
    plt.show()

def show_positions(names, atom_pos, atom_ids, symbol_size=7, isim = 0):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi = 96)

    for i, (name, apos, aidsall) in enumerate(zip(names, atom_pos, atom_ids)):
        aids = np.squeeze(aidsall[:, isim])
        ir, ic = divmod(i, 2)
        axs[ir, ic].scatter(apos[aids==0,1], apos[aids==0,0], c = 'r', s = symbol_size)
        axs[ir, ic].scatter(apos[aids==1,1], apos[aids==1,0], c = 'k', s = symbol_size)
        axs[ir, ic].get_xaxis().set_visible(False)
        axs[ir, ic].get_yaxis().set_visible(False)
        axs[ir, ic].set_title(name)

    plt.tight_layout()
    plt.show()


def apply_prior(names, conf_counts, cfg_types, target_hist, prob_conf):
    
    # total number of configurations
    n_conf = sum([v for v in conf_counts.values()])
    multiple = [conf_counts[cfg] for cfg in cfg_types]

    target_probs = {}
    for name in names:
        t_hist = [target_hist[name].get(cfg, 0) for cfg in cfg_types]
        probs = [prob_conf[name].get(cfg, 0) for cfg in cfg_types]
        n_tot = sum(t_hist)

        # apply multinomial Jeffrey's prior
        t_adjusted = [n_tot*(nc + p*n_conf/2)/(n_tot + n_conf/2) for nc, mult, p in zip(t_hist, multiple, probs)]
        target_probs[name] = { cfg:t_adjusted[i] for i, cfg in enumerate(cfg_types)}

    return target_probs


def plot_histograms(names, num_samples, target_hist, target_hist_adjusted, prob_conf, cfg_types, x_re):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14), dpi = 96)

    cfg_labels = [str(c) for c in cfg_types]
    for i, name in enumerate(names):
        all_counts = np.array([target_hist[name].get(cfg, 0) for cfg in cfg_types])
        all_counts = all_counts/float(num_samples[name])
        all_counts_a = np.array([target_hist_adjusted[name].get(cfg, 0) for cfg in cfg_types])
        all_counts_a = all_counts_a/float(num_samples[name])

        p_array = [(k, v) for k, v in prob_conf[name].items()]
        all_counts_p = np.array([prob_conf[name].get(cfg, 0) for cfg in cfg_types])

        nconf = len(cfg_labels)
        width = 0.4

        ir, ic = divmod(i, 2)
        axs[ir, ic].bar(np.arange(nconf)+0.2, np.sqrt(all_counts_p), width, color='r', label='random')
        axs[ir, ic].bar(np.arange(nconf)-0.2, np.sqrt(all_counts_a), width, color='b', label='image adjusted')
        axs[ir, ic].bar(np.arange(nconf), np.sqrt(all_counts), width/2, color='y', label='image raw')

        axs[ir, ic].legend()
        axs[ir, ic].set_xticks(range(nconf))
        axs[ir, ic].set_xticklabels(cfg_labels, rotation=90.0, fontsize=16)
        #axs[ir, ic].set_title('Image: ' + name + '\nActual Re concentration: ' + str(x_re[name]))#', Total number of samples: n = ' + str(n_sample))
        axs[ir, ic].set_ylabel(r'$\sqrt{p}$', fontsize=20)

        plt.tight_layout()
    plt.show()

def plot_histograms_all(names, num_samples, target_hist, model_hist, prob_conf, cfg_types, x_re):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), dpi = 96)

    cfg_labels = [str(c) for c in cfg_types]
    num_labels = [str(i) for i, c in enumerate(cfg_types)]
    for i, name in enumerate(names):
        all_counts = np.array([target_hist[name].get(cfg, 0) for cfg in cfg_types])
        all_counts = all_counts/float(num_samples[name])
        #all_counts_a = np.array([model_hist[name].get(cfg, 0) for cfg in cfg_types])
        #all_counts_a = all_counts_a/float(num_samples[name])

        p_array = [(k, v) for k, v in prob_conf[name].items()]
        all_counts_p = np.array([prob_conf[name].get(cfg, 0) for cfg in cfg_types])

        nconf = len(cfg_labels)
        width = 0.4

        ir, ic = divmod(i, 2)
        axs[ir, ic].bar(np.arange(nconf)+0.2, np.sqrt(all_counts_p), width, color='r', label='Null')
        axs[ir, ic].bar(np.arange(nconf)-0.2, np.sqrt(model_hist[name]), width, color='b', label='Equil.')
        axs[ir, ic].bar(np.arange(nconf), np.sqrt(all_counts), width/2, color='y', label='Target')
        #axs[ir, ic].bar(np.arange(nconf)+0.2, (all_counts_p), width, color='r', label='Null')
        #axs[ir, ic].bar(np.arange(nconf)-0.2, (model_hist[name]), width, color='b', label='Equil.')
        #axs[ir, ic].bar(np.arange(nconf), (all_counts), width/2, color='y', label='Target')

        if i == 0:
            axs[ir, ic].legend()

        if ic == 0:
            axs[ir, ic].set_ylabel(r'$\sqrt{p}$', fontsize=20)

        #axs[ir, ic].set_xticklabels(fontsize=18)
        #axs[ir, ic].set_yticklabels(fontsize=18)
        #axs[ir, ic].set_xticklabels(num_labels, rotation=90.0)
        if ir > 0:
            axs[ir, ic].set_xlabel(r'configuration number', fontsize=20)
        #    axs[ir, ic].set_xticks(range(nconf))
            #axs[ir, ic].set_xticklabels(cfg_labels, rotation=90.0)
        #axs[ir, ic].set_title('Image: ' + name + '\nActual Re concentration: ' + str(x_re[name]))#', Total number of samples: n = ' + str(n_sample))

        plt.tight_layout()
    plt.show()

def plot_cg_histograms(names, num_samples, target_hist, target_hist_adjusted, prob_conf, cfg_types, x_re):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14), dpi = 96)

    #cfg_labels = [str(c) for c in cfg_types]
    nconf = 8
    cfg_labels = [str(c) for c in range(nconf)]
    p_labels = [str(round(c/5, 1)) for c in range(6)]
    for i, name in enumerate(names):
        all_counts = np.zeros(nconf)
        for cfg in cfg_types:
            k = cfg[0] + cfg[1]
            all_counts[k] += target_hist[name].get(cfg, 0)
        all_counts = all_counts/float(num_samples[name])
            
        #all_counts = np.array([target_hist[name].get(cfg, 0) for cfg in cfg_types])
        #all_counts = all_counts/float(num_samples[name])
        #all_counts_a = np.array([target_hist_adjusted[name].get(cfg, 0) for cfg in cfg_types])
        #all_counts_a = all_counts_a/float(num_samples[name])

        #p_array = [(k, v) for k, v in prob_conf[name].items()]
        #all_counts_p = np.array([prob_conf[name].get(cfg, 0) for cfg in cfg_types])

        #nconf = len(cfg_labels)
        width = 0.4

        ir, ic = divmod(i, 2)
        #axs[ir, ic].bar(np.arange(nconf)+0.2, np.sqrt(all_counts_p), width, color='r', label='random')
        #axs[ir, ic].bar(np.arange(nconf)-0.2, np.sqrt(all_counts_a), width, color='b', label='image adjusted')
        axs[ir, ic].bar(np.arange(nconf), all_counts, width, color='b')

        #axs[ir, ic].legend()
        axs[ir, ic].set_xticks(range(nconf))
        #axs[ir, ic].set_xticklabels(cfg_labels, rotation=90.0, fontsize=16)
        axs[ir, ic].set_xticklabels(cfg_labels, fontsize=20)
        axs[ir, ic].set_yticklabels(p_labels, fontsize=20)
        #axs[ir, ic].set_title('Image: ' + name + '\nActual Re concentration: ' + str(x_re[name]))#', Total number of samples: n = ' + str(n_sample))
        #axs[ir, ic].set_ylabel(r'$\p$', fontsize=20)
        axs[ir, ic].set_ylim(0, 1.0)

        if ic == 0:
            axs[ir, ic].set_ylabel(r'P', fontsize=20)

        if ir > 0:
            axs[ir, ic].set_xlabel(r'Number of Re atoms', fontsize=20)

        plt.tight_layout()
    plt.show()

def plot_cg_histograms_singles(names, num_samples, target_hist, model_hist, prob_conf, cfg_types, x_re):
    #fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14), dpi = 96)
    plt.figure(figsize=(12,8))

    #cfg_labels = [str(c) for c in cfg_types]
    nconf = 8
    cfg_labels = [str(c) for c in range(nconf)]
    p_labels = [str(round(c/5, 1)) for c in range(6)]
    x_val = ['x = 0.05', 'x = 0.55', 'x = 0.78', 'x = 0.95']
    for i, name in enumerate(names):
        all_counts = np.zeros(nconf)
        for cfg in cfg_types:
            k = cfg[0] + cfg[1]
            all_counts[k] += target_hist[name].get(cfg, 0)
        all_counts = all_counts/float(num_samples[name])
            
        #all_counts = np.array([target_hist[name].get(cfg, 0) for cfg in cfg_types])
        #all_counts = all_counts/float(num_samples[name])
        #all_counts_a = np.array([target_hist_adjusted[name].get(cfg, 0) for cfg in cfg_types])
        #all_counts_a = all_counts_a/float(num_samples[name])

        #p_array = [(k, v) for k, v in prob_conf[name].items()]
        #all_counts_p = np.array([prob_conf[name].get(cfg, 0) for cfg in cfg_types])

        #nconf = len(cfg_labels)
        width = 0.25

        ir, ic = divmod(i, 2)
        #axs[ir, ic].bar(np.arange(nconf)+0.2, np.sqrt(all_counts_p), width, color='r', label='random')
        #axs[ir, ic].bar(np.arange(nconf)-0.2, np.sqrt(all_counts_a), width, color='b', label='image adjusted')
        #plt.bar(np.arange(nconf)-0.375+0.25*i, all_counts, width, label=x_val[i])
        plt.plot(np.arange(nconf), all_counts, 'o-', label=x_val[i], mfc='white', mew=2, markersize=9, linewidth=4)

        #axs[ir, ic].legend()
    plt.xticks(range(nconf), fontsize=20)
        #axs[ir, ic].set_xticklabels(cfg_labels, rotation=90.0, fontsize=16)
        #plt.xticklabels(cfg_labels, fontsize=20)
    plt.yticks(fontsize=20)
        #plt.yticklabels(p_labels, fontsize=20)
        #axs[ir, ic].set_title('Image: ' + name + '\nActual Re concentration: ' + str(x_re[name]))#', Total number of samples: n = ' + str(n_sample))
        #axs[ir, ic].set_ylabel(r'$\p$', fontsize=20)
    plt.ylim(0, 1.0)
    plt.xlim(-0.1, 7.1)
    plt.ylabel(r'P', fontsize=20)
    plt.xlabel(r'Number of Re atoms', fontsize=20)
    plt.legend(fontsize=20)

    plt.show()


def get_probability_histogram(names, cfg_types, target_hist):
    gsv = {}
    for i, name in enumerate(names):
        counts = target_hist[name]
        all_counts = np.array([counts.get(cfg, 0) for cfg in cfg_types])
        n_sample = np.sum(all_counts)
        all_counts = all_counts/float(n_sample)
        gsv[name] = all_counts
        
    return gsv
