#!/usr/bin/env python
from __future__ import print_function
from cyvcf2 import VCF

import pandas as pd
import numpy as np
import csv
import time
import argparse

#or mpl.use('PDF')
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plot
from common.locals import *
from common.pga4gh import *


def filter_genotype(gts, samples):
    #unknown genotype
    badrows = np.where(gts > 2.)[0]
    gts = np.delete(gts, badrows, axis=0)
    mask = np.all(gts > 0., axis = 1)
    gts = gts[~mask, :]

    mask = np.all(gts == 0., axis = 1)
    gts = gts[~mask, :]
   
    #remove individuals with ref for all variants
    mask = np.all(gts == 0., axis = 0)
    gts = gts[:, ~mask]
    indices = np.where(mask)[0]
    samples = [sample for j, sample in enumerate(samples) if j not in indices]
   
    #remove singletons 
    G = np.where(gts > 0, 1, 0)
    bad_vars = np.where(np.sum(G,axis=1) == 1)[0]
    gts = np.delete(gts, bad_vars, axis=0)

    return gts, samples


def normalize_rows(mat):
    for i in range(mat.shape[0]):
        mat[i, :] = mat[i, :]/np.sum(mat[i, :])
    return mat

def liklihood(G, Q, F):
    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)
    L = np.sum(np.multiply(G, np.log(QF)) + np.multiply((2.-G), np.log(QF2)))
    return L


def initialize_Q(G, F, rounds=10):
    K, J = F.shape

    # seeds for symmetric dirichlet.
    seeds = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.7, 0.8, 0.9, 1.]
    I, J = G.shape
    alpha = np.zeros(K)
    alpha[:] = 1./K
    Q = np.random.dirichlet(alpha, size=I)

    Q_max = Q.copy()
    L_0 = liklihood(G, Q, F)
    print ("L_0 is:{}".format(L_0))

    for x in range(rounds):
        alpha = np.zeros(K)
        alpha[:] = seeds[x]
        Q = np.random.dirichlet(alpha, size=I)

        Q = normalize_rows(Q)

        L_1 = liklihood(G, Q, F)
        print ("init {}, likelihood:{}".format(x, L_1))
        if L_1-L_0 > 0.:
            Q_max = np.matrix.copy(Q)

    return Q_max


def Qupdate(G, F, Q):
    """
    single update step for Q
    """
    I, J = G.shape

    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)

    # E-Step: update a_ijk and b_ijk
    a = np.einsum('ik,kj->ijk', Q, F)
    a = a / QF[:, :, np.newaxis]
    b = np.einsum('ik,kj->ijk', Q, 1.-F)
    b = b / QF2[:, :, np.newaxis]

    # M-Step:updating Q
    term1 = np.einsum('ij,ijk->ik', G, a)
    term2 = np.einsum('ij,ijk->ik', 2.-G, b)
    Q = (term1 + term2)/(2.*J)

    # prevents accumulation of precision-errors at a single value
    Q = normalize_rows(Q)

    return Q


def Fupdate(G, F, Q):
    """
    single update step for F
    """
    I, J = G.shape

    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)

    # E-Step: update a_ijk and b_ijk
    a = np.einsum('ik,kj->ijk', Q, F)
    a = a / QF[:, :, np.newaxis]
    b = np.einsum('ik,kj->ijk', Q, 1.-F)
    b = b / QF2[:, :, np.newaxis]

    # M-Step: Updating F
    term1 = np.einsum('ij,ijk->kj', G, a)
    term2 = np.einsum('ij,ijk->kj', 2.-G, b)
    F = term1/(term1+term2)
    F = np.clip(F, 0., 1.)

    return F


def SQUARED_Update(M0, M1, M2):
    # Compute steplength using S3.
    R = M1 - M0
    V = M2 - M1 - R
    a = -1.*np.sqrt((R*R).sum()/(V*V).sum())

    if a > -1:
        a = -1.

    a_adjusted = False
    while not a_adjusted:
        M = (1+a)**2*M0 - 2*a*(1+a)*M1 + a**2*M2
        if (M < 0.).any() or (M > 1.).any():
            a = (a-1)/2.
        else:
            a_adjusted = True

    if np.abs(a+1) < 1e-4:
        a = -1.

    # if this accelerated step fails for some reason,
    # stick with the first non-accelerated step.
    if np.isnan(M).any():
        M = M1.copy()

    return M


def Q_SQUARED_Update(G, F, Q0):
    Q1 = Qupdate(G, F, Q0)
    Q2 = Qupdate(G, F, Q1)

    Q = SQUARED_Update(Q0, Q1, Q2)
    Q = normalize_rows(Q)

    return Q


def F_SQUARED_Update(G, F0, Q):
    F1 = Fupdate(G, F0, Q)
    F2 = Fupdate(G, F1, Q)

    F = SQUARED_Update(F0, F1, F2)

    return F


def SQUAREM(G, F0, Q0,  K, e, maxiters):

    I, J = G.shape
    notconverged = True
    iteration = 1

    L_0 = L_1 = 0.0

    # Computing likelihood for the first time
    L_0 = liklihood(G, Q0, F0)
    print("Initial Likelihood is :{}".format(L_0))

    itertime = time.time()

    Q_max = Q0.copy()
    while notconverged:
        # update F, given Q
        F = F_SQUARED_Update(G, F0, Q0)
        F0 = Fupdate(G, F, Q0)

        # update Q, given F
        Q = Q_SQUARED_Update(G, F0, Q0)
        Q0 = Qupdate(G, F0, Q)
        
         
        if not np.isnan(Q).any(): Q_max = Q.copy()  
          
        # Compute likelihood once every 10 iterations
        if iteration % 10 == 0: 
           L_0 = liklihood(G, Q, F)
           L_1 = liklihood(G, Q0, F0)

           itertime, lasttime = time.time(), itertime

           notconverged = (abs(L_1 - L_0) >= e)
           print ("{}, Likelihood:{}, delta:{}, time:{}".format(iteration, L_1, abs(L_1-L_0), itertime-lasttime))

        if maxiters and iteration >= maxiters:
            break
        iteration += 1

    return Q_max


def plot_admixture(Q, population_indices, population_labels):

    N,K = Q.shape
    colors = [ 'g','b', 'm', 'r', 'y']
    
    text_color = 'k'
    bg_color = 'w'
    fontsize = 12

    figure = plot.figure(figsize=(5,3))

    xmin = 0.13
    ymin = 0.2
    height = 0.6
    width = 0.74
    indiv_width = width/N
    subplot = figure.add_axes([xmin,ymin,width,height])
    [spine.set_linewidth(0.001) for spine in subplot.spines.values()]

    for k in xrange(K):
        if k:
            bottoms = Q[:,:k].sum(1)
        else:
            bottoms = np.zeros((N,),dtype=float)

        lefts = np.arange(N)*indiv_width
        subplot.bar(lefts, Q[:,k], width=indiv_width, bottom=bottoms, facecolor=colors[k], edgecolor=colors[k], linewidth=0.4)

        subplot.axis([0, N*indiv_width, 0, 1])
        subplot.tick_params(axis='both', top=False, right=False, left=False, bottom=False)
        xtick_labels = tuple(map(str,['']*N))
        subplot.set_xticklabels(xtick_labels)
        ytick_labels = tuple(map(str,['']*K))
        subplot.set_yticklabels(ytick_labels)

  
    for p,popname in enumerate(population_labels):
        indices = np.where(population_indices==p)[0]
        if indices.size>0:
            vline_pos = (indices.max()+1)*indiv_width 
            subplot.axvline(vline_pos, linestyle='-', linewidth=0.2, c='#888888')
            label_position = (xmin+(2*indices.min()+indices.size)*0.5*indiv_width, ymin-0.01)
            figure.text(label_position[0], label_position[1], popname, fontsize=6, color='k', \
                horizontalalignment='right', verticalalignment='top', rotation=70)

    return figure

def get_admixture_proportions(Q, sample_wise_subpops ,population_labels ):
     
    # get population labels
    seen_labels = list(set(sample_wise_subpops)) 
    population_labels = [label for label in population_labels if label in seen_labels]

    population_cluster = [np.mean(Q[[i for i,p in enumerate(sample_wise_subpops) if p==label],:],0).argmax() for label in population_labels] 
    population_indices = np.array([population_labels.index(pop) for pop in sample_wise_subpops])
     
    # re-order samples in admixture matrix
    order = np.argsort(population_indices)
    population_indices = population_indices[order]
    Q = Q[order,:]

    for ind in range(len(seen_labels)):
        indices  = [i for i, val in enumerate(population_indices) if val==ind]
        Q[indices] = Q[indices][Q[indices][:, population_cluster[ind] ].argsort()]
  
    return Q, population_indices, population_labels


                
def  graph_admixture(Q, samples, subpops, outputfile):

    sample_wise_subpops = [subpops[sample] for sample in samples]

    ordered_subpops = ['LWK','ESN','YRI','MSL','GWD','ACB','ASW','CLM','MXL','PUR','PEL','TSI','IBS','GBR','CEU','FIN','PJL','GIH','ITU','STU','BEB','CDX','KHV','CHS','CHB','JPT']
        
    # get the data to be plotted
    admixture, population_indices, population_labels = get_admixture_proportions(Q, sample_wise_subpops, ordered_subpops )
   
    # plot the data
    figure = plot_admixture(admixture, population_indices, population_labels)
    figure.savefig(outputfile, dpi=500)


def parse_args():
    parser = argparse.ArgumentParser(description='An SQUAREM implementation to infer population structure')
    parser.add_argument('--K', help='The number of populations(clusters)', type=int, default=5)
    parser.add_argument('--iterations', help='The maximum number of iterations', type=int)
    parser.add_argument('--epsilon', help='Convergence relaxation', type=float, default=1.e-4)
    parser.add_argument('--regions', help='A file that specifies genomic regions to query ')
    parser.add_argument('--outputfile', help='A file to plot the output', default="admixture_out")
    parser.add_argument('--vcfs', help='vcf paths if to query locally', default = ["./data/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"])
    parser.add_argument('--ga4gh', help='Use ga4gh search_variants instead of vcf', action='store_true')
    parser.add_argument('--snps_only', help='Only include SNPs', action='store_true', default= False)
    parser.add_argument('--servers', nargs='*', help='A list of servers to query', default = ["https://ga4gh.ccm.sickkids.ca/ga4gh/", "http://ga4gh.pmgenomics.ca/ga4gh/", "http://10.9.208.74/ga4gh/"])
    parser.add_argument('--I',  help='Number of samples to consider', type=int)
    parser.add_argument('--J',  help='Number of variants to consider', type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    servers = args.servers
    regions = args.regions
    snps_only = args.snps_only
    I = args.I
    J = args.J
    K = args.K
    epsilon = args.epsilon
    max_iters = args.iterations
    vcfs = args.vcfs
    outputfile = args.outputfile
 
    #for display only 
    np.set_printoptions(threshold='nan')
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_rows', 3000)


    regions = []
    if regions: 
        regions  = get_regions(regions)  
    else: 
        print("No regions file given, setting defaults now.") 
	regions = [{'chr':'20', 'start':10000, 'end': 100000}] 
   
    print("Total regions to query:{}".format(len(regions)))

    G = np.array((0,0))
    samples = None
    ancestry_info = {}
      
    if args.ga4gh: 
	 df = get_ga4gh_variants(servers, regions[:], snps_only).fillna(0)
         G = df.values[:, 1:]
         samples = df.columns.values[1:]
         ancestry_info = get_ancestry_info(servers)
    else:
         if len(vcfs) == 1: vcfs = [vcfs[0]*len(regions)]
	 G = Get_variants_from_files(vcfs, regions[:], snps_only, I = I, J = J)
         samples = fill_samples_from_file(vcfs[0])
         pops, ancestry_info = population_dictionaries('./data/samples-pops-subpops.csv')

    if G.size == 0: print("G is empty.Exitting!!!"); sys.exit(0)

    G, samples = filter_genotype(G, samples)
    G =  np.transpose(G).astype(float)
    
    F =  np.random.rand(K,G.shape[1])              
    Q = initialize_Q(G, F, rounds=10)

    Q = SQUAREM(G, F, Q, K, epsilon, max_iters)

    graph_admixture(Q, samples, ancestry_info, outputfile)
    print("Admixture plotted in ", outputfile)
    

