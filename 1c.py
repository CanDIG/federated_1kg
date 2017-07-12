#!/usr/bin/env python
# This script will find all singletons from a given region and the samples. 
 
from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys

import argparse

#or mpl.use('PDF')
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common.locals import *
from common.pga4gh import *

    
def singletons_only(G):
        G = np.where(G > 0, 1, 0)
	bad_var = np.where(np.sum(G,axis=1) <> 1)[0]
	G = np.delete(G, bad_var, axis=0)
        return G



def graph(G, samples, subpops, filename):



	#transpose to get individuals as rows so the sum(i_th row) gives No. of singletons in i_th individual 
	G = np.transpose(G)

	Sgl =  zip(samples, np.sum(G,axis=1))
          
	ordered_subpops = ['LWK', 'GWD', 'MSL', 'ACB','ASW','YRI', 'ESN','BEB', 'STU','ITU', 'PJL','GIH','CHB','KHV','CHS','JPT','CDX','TSI','CEU','IBS','GBR','FIN','PEL','MXL','CLM','PUR']    
	subpop_sum = {}
	subpop_count = {}


	subpop_sum = subpop_sum.fromkeys(ordered_subpops, 0)
	subpop_count = subpop_count.fromkeys(ordered_subpops, 0)


	for key, val in enumerate(Sgl):
	    subpop_sum[subpops[val[0]]] += val[1]
	    subpop_count[subpops[val[0]]] += 1 
	  
	averages = []
	for i, key in enumerate(ordered_subpops):
	    averages.append(float(subpop_sum[key])/float(subpop_count[key]))  


	colors = population_to_colors(ordered_subpops)
	mpl.rc('xtick', labelsize=10)
	mpl.rcParams['axes.formatter.useoffset'] = False
	plt.bar(range(0, len(averages), 1), averages, color=colors, tick_label=ordered_subpops, edgecolor='k')
	plt.xticks(rotation=90)
	plt.savefig(filename)

 

    
def parse_args():

	parser = argparse.ArgumentParser(description='Computes the average number of singletons in a population')

	parser.add_argument('--regions', help='A file that specifies genomic regions to query ')
        parser.add_argument('--outputfile', help='A file to plot the output', default="singletons_out")
	parser.add_argument('--vcfs', help='vcf files if to query locally', default = ["./data/ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"])
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

    #compute singletons and and graph results 
    G = singletons_only(G)
    graph(G, samples, ancestry_info, outputfile)
    print("Output plotted in ", outputfile)
    print("Plotted in ", args.outputfile)
