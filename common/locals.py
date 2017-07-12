import numpy as np
from cyvcf2 import VCF
import csv
import pandas as pda

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.colors as color



#************Samples***************

def fill_samples_from_file(filename):
        """
        Read samples from vcf file 
        """
	vcf = VCF(filename)
        return vcf.samples


#************Design Genomic Queries***************

def get_regions(filename):
    """
    Read the filename from csv containing filtered regions of interest and populate G   
    """ 
    df = pda.DataFrame.from_csv(filename, index_col=None)
    print (df)
    return df.to_dict('records')


def sample_regions(filename ='coord.txt' , basic_sample_times = 50, sample_size = 1000000, chrom_lst = [19, 20, 21,22]  ):
    """
    Create regions by sampling chromosomes 
    """   

    handle =  open(filename, 'r')
    lines = handle.readlines()
    chr_length = {}
    for line in lines:
    	num, length = line.split(',')
        chr_length[num] = int(length)
    handle.close()

    regions = []
    for chrnum in chrom_lst:
        sample_times = basic_sample_times*int(np.log(float(chr_length[str(chrnum)])))
        gap = chr_length[str(chrnum)]/sample_times  
        for it in range(sample_times):
	 	regions.append({'chr':chrnum, 'start':(it*gap), 'end': (it*gap+sample_size)})	
    return regions


#************Query Genomic Regions***************

def query_region_from_file(filename, mystart, myend, chrom, snps_only):
        """
        Query a region from local file system 
        """   

    	vcf = VCF(filename)
    
    	region = str(chrom)+":"+str(mystart)+"-"+str(myend)

        if snps_only: return np.array([np.copy(v.gt_types) for v in vcf(region) if len(v.ALT) == 1 and v.num_unknown == 0 ])
    	else: return  np.array([np.copy(v.gt_types) for v in vcf(region)])
        
        
def Get_variants_from_files(filenames, regions, snps_only, I = None, J = None ):
        
        """
        Query multiple regions from local file system 
        """ 

        G = np.empty((0,0))
     
        if len(filenames) != len(regions): print("Can't query locally. Number of vcf files must match the number of regions."); return G

         
	for ind, region in enumerate(regions):
        	G_local = query_region_from_file(filenames[ind],  region['start'] ,  region['end'], region['chr'], snps_only)
                if not G_local.size: continue
		if G.size: G = np.vstack((G,G_local)) 
		else: G = G_local.copy()   
       
        if G.size > 0:
		nrows, ncols = G.shape
		if I: print("shuffling individuals"); G = G[ :, np.random.choice(ncols, I, replace=False)-1 ]
		if J: print("shuffling variants"); G = G[ np.random.choice(nrows, J, replace=False)-1, : ]

        return G


#************Graphs***************
def population_to_colors(populations):
    pop_to_rgb = { 'ACB': (0.84, 0.52, 0.13, 1.0), 'GWD': (0.96, 0.92, 0.18, 1.0),
                   'BEB': (0.37, 0.07, 0.43, 1.0), 'PEL': (0.71, 0.02, 0.1, 1.0),
                   'LWK': (0.72, 0.6, 0.3, 1.0), 'MSL': (0.8, 0.67, 0.15, 1.0),
                   'GBR': (0.48, 0.72, 0.79, 1.0), 'IBS': (0.35, 0.43, 0.66, 1.0),
                   'ASW': (0.77, 0.32, 0.11, 1.0), 'TSI': (0.12, 0.13, 0.32, 1.0),
                   'KHV': (0.39, 0.64, 0.22, 1.0), 'CEU': (0.17, 0.23, 0.53, 1.0),
                   'SAS': (0.52, 0.27, 0.54, 1.0), 'EAS': (0.67, 0.76, 0.15, 1.0),
                   'AMR': (0.45, 0.13, 0.11, 1.0), 'YRI': (0.92, 0.75, 0.36, 1.0),
                   'CHB': (0.67, 0.77, 0.16, 1.0), 'CLM': (0.62, 0.14, 0.16, 1.0),
                   'CHS': (0.45, 0.67, 0.19, 1.0), 'ESN': (0.94, 0.77, 0.14, 1.0),
                   'FIN': (0.39, 0.68, 0.74, 1.0), 'AFR': (0.97, 0.92, 0.24, 1.0),
                   'GIH': (0.32, 0.19, 0.5, 1.0), 'PJL': (0.69, 0.0, 0.45, 1.0),
                   'EUR': (0.53, 0.73, 0.84, 1.0), 'STU': (0.5, 0.25, 0.54, 1.0),
                   'MXL': (0.69, 0.0, 0.16, 1.0), 'ITU': (0.53, 0.13, 0.31, 1.0),
                   'CDX': (0.32, 0.56, 0.2, 1.0), 'JPT': (0.25, 0.49, 0.2, 1.0),
                   'PUR': (0.62, 0.14, 0.16, 1.0)}

    if type(populations) is list:
        colors = [ pop_to_rgb[pop] for pop in populations ] 
    else:
        colors = pop_to_rgb[populations]

    return colors


def population_dictionaries(filename):
    popdict = {}
    subpopdict = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample = row['sample']
            popdict[sample] = row['population']
            subpopdict[sample] = row['subpopulation']

    return popdict, subpopdict
            
