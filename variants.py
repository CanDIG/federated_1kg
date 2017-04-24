#!/usr/bin/env python
# This script will compute the average number of variants, grouped by (sub)population.  
# 1) fetch data from servers and do the analysis in a federated way. 

from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys
import pickle
import popdicts as pd
from collections import Counter

import argparse


def variants_from_file(filename, I = None, J = None, _curr_start=0, _curr_end=8000000, chrom="20"):

    try:
    	vcf = VCF(filename)

    	region = str(chrom)+":"+str(_curr_start)+"-"+str(_curr_end)
       
    	gts = np.array([np.copy(v.gt_types) for v in vcf(region)])
    
    	#convert to bool with true for the entries > 0
    	gts = np.where(gts > 0, 1, 0)

    	samples = vcf.samples
    
    
    	if gts.size == 0: return None; 
    	nrows, ncols = gts.shape
    	if I:
    		gts = gts[ :, np.random.choice(ncols, I, replace=False)-1 ]
		samples = samples[:I]
    	if J:
    		gts = gts[ np.random.choice(nrows, J, replace=False)-1, : ]
 
    
    	bad_rows = np.where(np.sum(gts,axis=1) == 0)[0] 
    	gts = np.delete(gts, bad_rows, axis=0)
    	if gts.size == 0: return None;
 
    	gts = np.transpose(gts)
 
    	I, J = gts.shape
    
    	variant_count = {}
    	variant_count.fromkeys(samples, 0)

    	for i in range(I):
		variant_count[samples[i]] = sum(gts[i])
    
        return variant_count
    except:
	print("ERROR! Check if the data file(s) exist.")
        sys.exit(1)
             

def parse_args():
	parser = argparse.ArgumentParser(description=' the average number of variants, grouped by (sub)population')
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
                        variants_local = variants_from_file(filename, args.I[0], args.J[0], _curr_start=it*gap, _curr_end=(it*gap+min(gap, args.size[0])), chrom=chrnum)
                        print ("querying {}:{}-{}".format(chrnum, it*gap , (it*gap+min(gap, args.size[0]))))
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
            

