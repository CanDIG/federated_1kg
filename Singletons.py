#!/usr/bin/env python
# This script will find all singletons from a given region and the samples. 
 
from __future__ import print_function
import ga4gh.client.client as client

from cyvcf2 import VCF
import numpy as np

import sys
import pickle

import argparse



def Get_Singletons_from_file(filename = "chr20.vcf.gz", I = None, J = None, mystart=0, myend=8000000, chrom="20"):

    try:
    	vcf = VCF(filename)
    
     
    	region = chrom+":"+str(mystart)+"-"+str(myend)

    	gts = np.array([np.copy(v.gt_types) for v in vcf(region)])
    

    	samples = vcf.samples
    
    	nrows, ncols = gts.shape
    	if I:
    		gts = gts[ :, np.random.choice(ncols, I, replace=False)-1 ]
		samples = samples[:I]
    	if J:
		gts = gts[ np.random.choice(nrows, J, replace=False)-1, : ]
 
  

     
    	#convert to bool with true for the entries > 0
    	gts = np.where(gts > 0, 1, 0)
   
    	#filter out the rows which do not sum to 1 
    	bad_rows = np.where(np.sum(gts,axis=1) <> 1)[0] 
    	gts = np.delete(gts, bad_rows, axis=0)

    	#transpose to get individuals as rows so the sum(i_th row) gives No. of singletons in i_th individual 
    	gts = np.transpose(gts)
    
        vcf.close()
        return zip(samples, np.sum(gts,axis=1))
    except:
    	print("Errror! Check if the file exist.")
      	sys.exit(1)
    
def parse_args():
	parser = argparse.ArgumentParser(description='Computes the average number of singletons in a population')
	parser.add_argument('--inputfile', nargs=1, help='input file name', default = "chr20.vcf.gz")
        parser.add_argument('--region', nargs=3, help='chromosome_number start end', type=int, default=[20, 0, 8000000])
        parser.add_argument('--I', nargs=1, help='Number of samples to consider', type=int, default =  [None])
        parser.add_argument('--J', nargs=1, help='Number of variants to consider', type=int, default = [None])
 	args = parser.parse_args()
        return args


if __name__ == "__main__": 
    
    args = parse_args()

    np.set_printoptions(threshold='nan')

    singletons = Get_Singletons_from_file(filename = args.inputfile, chrom = str(args.region[0]), I = args.I[0], J=args.J[0], mystart = args.region[1], myend = args.region[2])
   
    if singletons is not None:
    	print ("Writing the output to Singletons.pickle")
    	with open('Singletons.pickle', 'wb') as handle:
    		pickle.dump(singletons, handle)
