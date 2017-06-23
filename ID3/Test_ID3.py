from __future__ import print_function


from cyvcf2 import VCF

import numpy as np
import pandas as pda
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from BCBio import GFF

import dtreediff
import csv

import argparse

from ga4gh_merge import *


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

def Get_Genotypes_from_file(filename, mystart=0, myend=90000, chrom="20"):

    vcf = VCF(filename)
    chrom = chrom.replace('chr','')
    region = chrom+":"+str(mystart)+"-"+str(myend)


    gts = [np.copy(v.gt_types) for v in vcf(region) if v.is_snp and len(v.ALT) == 1]
   


    #only return the bi-allelic variants
    bad_rows = np.where(gts == 2)[0]

    gts = np.delete(gts, bad_rows, axis=0)
 
    gts = np.clip(gts, 0, 2)
    
   
    return gts




def Get_Genotype(regions, I=2504):

	G = np.zeros((0,I))

    	for reg in regions:
        	filename = "ALL."+str(reg['chr'])+".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" 
		G_local = Get_Genotypes_from_file(filename, mystart=reg['start'], myend=reg['end'], chrom=reg['chr'])

                if G_local.size > 0: 
			G_local= G_local[:,:I]
        		G = np.vstack([G, G_local])
        	
   
         #remove singletons and unseen variants
        G2 = np.where(G>=1, 1, 0) 

        bad_rows = np.where(np.sum(G2,axis=1) <= 1)[0] 
        G = np.delete(G, bad_rows, axis=0) 

  
        return G


def filter_genotype(G, I = 2504):

   
    pops, subpops = population_dictionaries('samples-pops-subpops.csv')
    distinct_pops = set(pops.values())

    #get samples from the a vcf file. samples-pops-subpops.csv has higher No. of records
    filename = "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" 
    vcf = VCF(filename)
    samples = vcf.samples[:I]


    #computing No. of samples in each population
    total_counts = {}
    total_counts = total_counts.fromkeys(distinct_pops, 0)
   

    #Compute the overall distribution of the samples among diff: pops
    for sample in samples:
	total_counts[pops[sample]] += 1

     
    SNPs = 0
    indices = []

    for ind,g in enumerate(G):
        g = list(g)

        pop_count = {}
        pop_count = pop_count.fromkeys(distinct_pops, 0)

        #now computing pop-wise freq: so if a sample has gen in {1,2} we add a 1 to corresponding pop
        
        for loc, val in enumerate(g):
		if val > 0: pop_count[pops[samples[loc]]] += 1


        freq = {}
        freq = freq.fromkeys(distinct_pops, 0.0)
        for p in distinct_pops:
                if float(total_counts[p]) > 0.0 and pop_count[p] > 0.0:    
			freq[p] = float(pop_count[p])/float(total_counts[p]) 

        participating_counts = 0
        zeros = 0
        for k,v in freq.items():
		if v > 0.1: participating_counts += 1
                if v == 0.0: zeros += 1


        if participating_counts == 1 and zeros == len(freq.items())-1: 
                SNPs += 1
        else: indices.append(ind)         

    
    G = np.delete(G, indices, axis=0)  
    return G


def get_SNP_regions(my_genes, in_file = "gencode.v24.annotation.gff3" ):
        
	in_handle = open(in_file)
        regions = []
	for index, rec in enumerate(GFF.parse(in_handle,target_lines=130)):
    		
    		for x in rec.features:
                	if x.type == 'gene':
                                      
				for y in my_genes:
                                        if y == x.qualifiers['gene_name'][0]:
	                                        regions.append({'chr': rec.id, 'start': x.location.start, 'end': x.location.end})   
				
                      
	in_handle.close()
        return regions



def Bind_Data(G, I, filename = "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"):

    rows, cols = G.shape
    attr = []

    #generating pseudo-headers
    for x in range(cols):
    	attr.append("SNP"+str(x)) 
   
    data = [zip(attr, G[ind]) for ind, datum in enumerate(G)]
     
    #reading target_class
    pops, subpops = population_dictionaries('samples-pops-subpops.csv')

    
    vcf = VCF(filename)
    samples = vcf.samples[:I]

    pop_lst = [pops[sample] for sample in samples]
    
    data  = [dict(d, pop=n) for d, n in zip(data, pop_lst)]
   

    attr.append('pop')
    target_attr =  attr[-1]
 
    return data, attr, target_attr


def Confusion_Matrix(classification, actual_values):

        actu = pda.Series(actual_values, name='Actual')
	pred = pda.Series(classification, name='Predicted')
	df_confusion = pda.crosstab(actu, pred)
        return df_confusion 



#using the map of (pop:num), convert the predicted numbers into pop_names	
def replace(predicted, targets_map):
	predicted2 = [] 
    	for j in range( len(predicted)):
    		for ind, label in enumerate(targets_map):
          	 	if predicted[j] == ind: predicted2.append(label)
	return predicted2


def Accuracy(actual, predicted):

	count = 0
    	for i, label in enumerate(predicted):
    		if label == actual[i]: count += 1
 
        return count 


def encode_target(df, target_column):
   
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)


def Test_SK_Learn(data, train_sz):

    	df = pda.DataFrame.from_records(data[:int(I*(train_sz))])
    	test_data = pda.DataFrame.from_records(data[int(I*(train_sz)):])
 
         
    	df2, targets = encode_target(df, "pop")

    	
    	features = list(df2.columns[:len(df2.columns)-2])
    
    	y = df2["Target"]
    	X = df2[features]
  
    	test_X = test_data[features]
     
    	dt = DecisionTreeClassifier()
    	dt.fit(X, y)

    	predicted = dt.predict(test_X)
        predicted2 = replace(predicted, targets)
        
    	#print (zip(list(test_data['pop']), predicted2))

    	actual = list(test_data['pop']) 
    	
        count = Accuracy(actual, predicted2)
 
        #print ("TP:{}, Accuracy[0,1]:{}".format(count, float(count)/len(predicted)))
        print (float(count)/len(predicted))
    	output = Confusion_Matrix(predicted2, actual)
    	print (output)


def Split_Data(data, train_sz, option='cn'):

        train_data = []
        test_data = []
        I = len(data)
	if option == 'random':
		train_indices = random.sample(range(I),int(I*train_sz))
                for ind in range(I):
			if ind in train_indices:
				train_data.append(data[ind])
			else:
				test_data.append(data[ind])
         

        #by default takes consequtive records for split int train and test.           
        else:
        	train_data = data[:(int(I*train_sz))]
		test_data = data[(int(I*train_sz)):] 



        targets = [rec['pop'] for rec in train_data]
        dis_tar = set(targets)

        targets = [rec['pop'] for rec in test_data[:]]
        dis_tar = set(targets)
        
	return train_data, test_data


def remove_Nones(actual_values, classification):
	for i, label in enumerate(classification):
		if label == None:
			actual_values.pop(i)
			classification.pop(i)

	return actual_values, classification


def parse_args():
	parser = argparse.ArgumentParser(description='A differentially-private Classification of individuals to populations')
        parser.add_argument('--I', nargs=1, help='The number of individuals to choose', type=int, default = [1000])
        parser.add_argument('--train', nargs=1, help='Training samples size', type=int, default = [0.85])
        parser.add_argument('--ga4gh', help='Use ga4gh serch_variants instead of vcf', action='store_true')
 	args = parser.parse_args()
        return args


if __name__ == "__main__":
    
     
    np.set_printoptions(threshold='nan') 

    args = parse_args()
    I = args.I[0]
    train_sz = args.train[0]
    ga4gh = args.ga4gh
    
    #Genes overlapping which the SNPs to be used as features
    my_genes = ['TYR', 'TYRP1', 'OCA2', 'MCIR', 'DCT', 'AP3B1', 'CYP3A4', 'CYP2C8', 'CYP2D6', 'CYP2C9', 'CYP1A1','AHR']
 
    #this will return the chrom:start-end for each gene in the arg 
    #regions = get_SNP_regions(my_genes)

    # Use ga4gh search_variants instead of a vcf
    if ga4gh:

        regions = get_filter_regions()
        servers = ["https://ga4gh.ccm.sickkids.ca/ga4gh/", "http://ga4gh.pmgenomics.ca/ga4gh/"]

        G = get_ga4gh_variants(servers, regions)
        data, attr, target_attr = ga4gh_Bind_Data(G, I)

    # Use a local vcf
    else:

        # don't need to recompute with every run. We can pickle that up but this will increase file-dependency for the program.    
        regions =  [{'start': (94938657), 'chr': 'chr10', 'end': (94989390)}, {'start': (95036771), 'chr': 'chr10', 'end': (95069497)}, {'start': (89177451), 'chr': 'chr11', 'end': (89295759)}, {'start': (94436807), 'chr': 'chr13', 'end': (94479682)}, {'start': (27754874), 'chr': 'chr15', 'end': (28099358)}, {'start': (74719541), 'chr': 'chr15', 'end': (74725610)}, {'start': (42126498), 'chr': 'chr22', 'end': (42130906)}, {'start': (78000524), 'chr': 'chr5', 'end': (78294755)}, {'start': (17298621), 'chr': 'chr7', 'end': (17346152)}, {'start': (99756959), 'chr': 'chr7', 'end': (99784265)}, {'start': (12685438), 'chr': 'chr9', 'end': (12710290)}]

        #If we have regions already, we can pull that up from a    
        #regions = read_regions(filename)

        G = Get_Genotype(regions, I)

        #we don't want all because this may make a model overfit to our data 
        G = filter_genotype(G, I)

        #now we need each row as an individual's genotype 
        G = np.transpose(G)



        #This file will be used to pull up the samples. One in the Samples-Subpops-Pops.csv file has a super-set of samples in vcfs 
        filename = "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"

        #bind the genotypes(features), target_attr(pops) and returns a list of (feature:val)dictionaries
        data, attr, target_attr = Bind_Data(G, I, filename)

    #test the data with Python's
    #Test_SK_Learn(data, train_sz) 

  
    #Now with our implementation
    train_data, test_data = Split_Data(data, train_sz, option = 'cn') 
    #print ("pop_lst:{}\nattr:{}, target:{}\n data:{}\n".format(pop_lst, attr,target_attr, data))
    tree = dtreediff.create_decision_tree(train_data, attr, target_attr)
    
    # Classify the records in the test_data
    classification = dtreediff.classify(tree, test_data[:])

    #extract actual labels and remove the records which were unseen to our classifier
    actual_values = [rec['pop'] for rec in test_data]
    actual_values, classification = remove_Nones(actual_values, classification)

    accuracy   =  Accuracy(classification, actual_values)
    print ("{}".format(float(accuracy)/len(actual_values)))
 
    print(Confusion_Matrix(classification, actual_values))
