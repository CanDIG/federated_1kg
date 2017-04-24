#!/usr/bin/env python
# This script will compute the number of variants which are rare (i.e. <0.5%) globally but > 5% in a sub-population. 
#Requirements: samples-pops-subpops.csv, popdicts.py
 
from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys
import pickle
import popdicts as pd
from collections import Counter

import argparse

def Get_rare_variants_from_file(filename, I = None, J = None, _curr_start=0, _curr_end=8000000, chrom="20"):

  try:
    vcf = VCF(filename)

    region = str(chrom)+":"+str(_curr_start)+"-"+str(_curr_end)

    gts = np.array([np.copy(v.gt_types) for v in vcf(region)])

    if gts.size == 0:
	return None

    #convert to values {1,0} with 1 for an element > 0
    gts = np.where(gts > 0, 1, 0)

    samples = vcf.samples
    pop_size = len(samples)

    nrows, ncols = gts.shape
    if I:
    	gts = gts[ :, np.random.choice(ncols, I, replace=False)-1 ]
	pop_size = I
	samples = samples[:I]
    if J:
    	gts = gts[ np.random.choice(nrows, J, replace=False)-1, : ]
 
   
    subpop_dist = {}
    pops, subpops = pd.population_dictionaries('samples-pops-subpops.csv')

    
    
  
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
  except:
        print ("ERROR! Something went wrong. Check if the datafile(s) exist.")
        sys.exit(1) 

def parse_args():
	parser = argparse.ArgumentParser(description='The number of variants which are rare (i.e. <0.5%) globally but > 5% in a sub-population')
	parser.add_argument('--metafile', nargs=1, help='a file containing: filename, chrom, start, end', default = ["co-ords.txt"])
        parser.add_argument('--size', nargs=1, help='The number of consequtive variants to select from region', type=int, default =  [1000])
        parser.add_argument('--freq', nargs=1, help='The number of times a region must be sampled', type=int, default = [1])
        parser.add_argument('--I', nargs=1, help='The number of bio_samples to choose', type=int, default = [2054])
        parser.add_argument('--J', nargs=1, help='The number of random variants to choose from a sample', type=int, default = [None])
 	args = parser.parse_args()
        return args


if __name__ == "__main__": 
    
    np.set_printoptions(threshold='nan')

    args = parse_args()


    try:
    	handle =  open(args.metafile[0], 'r')
    	lines = [line.rstrip('\n') for line in handle]
        handle.close()        
       
        variants = Counter() 
    	for line in lines:
        	filename, chrnum, start, end = line.split(',')
                start = float(start)
                end = float(end)
                _overall_sampling_freq  = args.freq[0]                 #Used this for exp: *int(np.log(end-start))
                gap = int((end-start)/_overall_sampling_freq) 

                for it in range(_overall_sampling_freq):
                        print ("querying {}:{}-{}".format(chrnum, it*gap , (it*gap+min(gap, args.size[0]))))
                        variants_local = Get_rare_variants_from_file(filename, args.I[0], args.J[0], _curr_start=it*gap, _curr_end=(it*gap+min(gap, args.size[0])), chrom=chrnum)
                        if variants_local:	   
                		variants +=  Counter(variants_local)



        if dict(variants):
        	with open('variants.pickle', 'wb') as fstream: pickle.dump(variants, fstream)
                fstream.close()  
                print ("Output written to the file variants.pickle")

        else: print ("No variants found")
    except:
        print ("ERROR! Something went wrong. Check if the metafile exist.")
        sys.exit(1)  
