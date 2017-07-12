#!/usr/bin/env python
# compute number of variants per individual. 
 
from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys

import argparse

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common.locals import *
from common.pga4gh import * 


def variants(gts, samples):
  
    
    #convert to bool with true for the entries > 0
    gts = np.where(gts > 0, 1, 0)

    bad_rows = np.where(np.sum(gts,axis=1) == 0)[0] 
    gts = np.delete(gts, bad_rows, axis=0)
 
    gts = np.transpose(gts)

    variant_count = {}
    variant_count.fromkeys(samples, 0)

    for i in range(gts.shape[0]):
	variant_count[samples[i]] = sum(gts[i])

        
   
    return variant_count
    
def graph_variants(variants, subpops, filename):

	y = variants.values()


	ordered_pops = ["FIN", "GBR", "CEU", "IBS", "TSI", "CHS", "CDX","CHB","JPT", "KHV",
		    "GIH", "STU", "PJL", "ITU", "BEB", "PEL", "MXL", "CLM", "PUR",
		    "ASW", "ACB", "GWD", "YRI", "LWK", "ESN", "MSL"]

	colors = population_to_colors(ordered_pops)
	
	plt.xlabel('Individual')
	plt.ylabel('Variant sites per genome')

	offset =19500.
	plt.tick_params(axis='x',which='both', bottom='off', top='off', labelbottom='off') 
	ax = plt.axes()

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	for i,curr_pop in enumerate(ordered_pops):
	    curr_samples = np.array([count for sample,count in variants.items() if subpops[sample]==curr_pop ])
	    curr_samples.sort()
	    
	    curr_offset = (11.*offset)*i+offset
	    x_values=[]
	    for j in range(0,len(curr_samples)) :
		x_values.append(curr_offset+offset*(float(j)/float(len(curr_samples)))) 
	   
	    
	    
	    plt.scatter(x_values, curr_samples, marker='+', s = 50., color = colors[i])
	    if i%2 == 0:
		plt.text(min(x_values), min(curr_samples)-(list(curr_samples).count(min(curr_samples)))-1, curr_pop, va='top', ha='center', size=8)
	    else:
		plt.text(max(x_values), max(curr_samples)+1, curr_pop, va='bottom', ha='center', size=8)
	#plt.axes().set_aspect('equal')
	plt.savefig(filename)

def parse_args():

	parser = argparse.ArgumentParser(description='Computes the average number of variants in a population')

	parser.add_argument('--regions', help='A file that specifies genomic regions to query ')
        parser.add_argument('--outputfile', help='A file to plot the output', default="variants_out")
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


    #compute average number of variants 
    var = variants(G, samples)

    #plot variant counts
    graph_variants(var, ancestry_info, args.outputfile)

    print("Output plotted in ", outputfile)
    

