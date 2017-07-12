#!/usr/bin/env python
# This script will compute the number of variants which are rare (i.e. <0.5%) globally but > 5% in a sub-population. 
 
from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys
import pickle
import argparse

#or mpl.use('PDF')
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common.locals import *
from common.pga4gh import *

def graph_rare_variants(rv, filename):

    pops = ["BEB", "PJL", "STU", "ITU", "GIH", "TSI", "IBS", "GBR", "CEU",
            "FIN", "CHS", "CHB", "KHV", "CDX", "JPT", "PUR", "CLM", "MXL",
            "PEL", "ASW", "ACB", "YRI", "ESN", "GWD", "MSL", "LWK"]
    values = []

    mpl.rc('ytick', labelsize=6.5)


    for k in pops:
        values.append(rv[k])
    
    colors = population_to_colors(pops)

    plt.barh(range(len(values), 0, -1), values, color=colors, tick_label=pops, edgecolor='k')
    plt.savefig(filename)

def rare_variants(gts, samples, subpops):

    #convert to values {1,0} with 1 for a value > 0
    gts = np.where(gts > 0, 1, 0)

    pop_size = len(samples)

    subpop_dist = {}


    #create a count of samples in each subpop
    for sample in samples:
    	if subpops[sample] in subpop_dist: subpop_dist[subpops[sample]]+= 1
	else: subpop_dist[subpops[sample]] = 1

   
    min_subpop = min(subpop_dist.values())
   
    #filter out the rows which do not sum to (pop_size*0.05) and which do not appear in > min_pop*0.05 times at least.   
    bad_rows = np.where(np.sum(gts,axis=1) >= (pop_size*0.005) )[0] 
    gts = np.delete(gts, bad_rows, axis=0)
    
    bad_rows = np.where(np.sum(gts,axis=1) <= (min_subpop*0.05) )[0] 
    gts = np.delete(gts, bad_rows, axis=0)

    if gts.size == 0:
	return None
    J,I = gts.shape
     
    rare_variants = {}
    rare_variants = rare_variants.fromkeys(subpop_dist.keys(),0)

    for j in range(J):
	entries = np.where(gts[j] > 0)[0]
        local_count = {}
        local_count = local_count.fromkeys(subpop_dist.keys(),0)
	for ind in entries:
		local_count[subpops[samples[ind]]] += 1

        for k,v in local_count.items():
		if float(v) > subpop_dist[k]*0.05:
			rare_variants[k] += 1

    return rare_variants


def parse_args():
	parser = argparse.ArgumentParser(description='compute the number of variants which are rare (i.e. <0.5%) globally but > 5% in a sub-population.')

	parser.add_argument('--regions', help='A file that specifies genomic regions to query ')
        parser.add_argument('--outputfile', help='output file to graph population-wise frequent variants', default="Rare_Variants_out")
	parser.add_argument('--vcfs', help='vcf file paths, if to query locally', default = ["./data/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"])
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

    stats = rare_variants(G, samples, ancestry_info )
    graph_rare_variants(stats, outputfile)
    print("output plotted in ", outputfile)
