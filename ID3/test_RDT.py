from __future__ import print_function


from cyvcf2 import VCF

import numpy as np
import pandas as pda
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from BCBio import GFF

import dtreediff
from Test_ID3 import *
from RDT import *
import csv

import argparse

from ga4gh_merge import *

from sklearn.decomposition import PCA


def parse_args():
	parser = argparse.ArgumentParser(description='A differentially-private Classification of individuals to populations')
        parser.add_argument('--I', nargs=1, help='The number of individuals to choose', type=int, default = [1000])
        parser.add_argument('--train', nargs=1, help='Training samples size', type=float, default = [0.85])
        parser.add_argument('--ga4gh', help='Use ga4gh serch_variants instead of vcf', action='store_true')
 	args = parser.parse_args()
        return args


if __name__ == "__main__":

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
        # regions = regions[:2]
        servers = ["https://ga4gh.ccm.sickkids.ca/ga4gh/", "http://ga4gh.pmgenomics.ca/ga4gh/"]
        # servers = servers[:1]

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
    # Test_SK_Learn(data, train_sz)


    df = pda.DataFrame.from_records(data)
    pop = df['pop']
    df.drop('pop', 1, inplace=True)
    print(df.shape)
    print(pop.shape)

    pop_set = set(pop)

    results_data = []

    for replicate in range(3):

        for max_h in [10]:

            # for eps in [-1, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 5, 10]:
            # for eps in [0.01, 0.02, 0.05, 0.1, 0.15, 0.3]:
            # for eps in [0.01]:
            for eps in [5]:

                # for ncomponents in [1, 5, 10, 20, 40, 80, 160]:
                for ncomponents in [40]:
                    pca = PCA(n_components=ncomponents)
                    pca.fit(df)
                    X = pca.transform(df)

                    Xdf = pda.DataFrame(X)
                    Xdf['pop'] = pop

                    Xnew = pda.DataFrame()
                    for col in Xdf.columns:
                        if col == 'pop':
                            Xnew[str(col)] = Xdf[col]
                        else:
                            Xnew[str(col)] = Xdf[col] > Xdf[col].mean()
                            Xnew[str(col)] = Xnew[str(col)].astype(int)

                    Xrecords = Xnew.to_dict(orient='records')

                    attr = list(Xnew.columns)
                    attr = [str(a) for a in attr]

                    train_data, test_data = Split_Data(Xrecords, train_sz, option = 'cn')

                    # for ntrees in [1, 5, 10, 15]:
                    for ntrees in [10]:
                        limit = 300
                        X = attr[:]
                        X.remove('pop')
                        
                        D = train_data[:]

                        Rs = []
                        for i in range(ntrees):
                                Rs.append(BuildTreeStructure(X[:], max_h, 0))
                        for R in Rs:
                                UpdateStatistics(R, D)
                                
                        for R in Rs:
                                AddNoise(R, pop_set, eps)

                        classification = []
                        for d in test_data[:]:
                                pred = rdtClassifyEnsemble(Rs, d)
                                classification.append(pred)

                        actual_values = [rec['pop'] for rec in test_data]
                        # actual_values, classification = remove_Nones(actual_values, classification)

                        accuracy = Accuracy(classification, actual_values)
                        print('ntrees: ', ntrees, 'ncomponents: ', ncomponents, 'eps:', eps, 'max_h:', max_h, 'replicate', replicate)
                        print ("TP:{}, Accuracy:{}".format(accuracy, float(accuracy)/len(actual_values)))

                        results_data.append({'ntrees':ntrees, 'ncomponents':ncomponents, 'tp':accuracy,
                                             'accuracy':float(accuracy)/len(actual_values), 'eps':eps, 'max_h':max_h,
                                             'replicate':replicate})

                        print(len(train_data))
                        print(len(test_data))
                        print(Confusion_Matrix(classification, actual_values))

    # results table
    results_df = pda.DataFrame.from_records(results_data)
    results_pt = pda.pivot_table(results_df, values=['accuracy'], aggfunc=[np.mean, np.std], columns=['ncomponents', 'ntrees', 'max_h'], index='eps')
    results_pt = results_pt*100
    # print(results_df)
    print(results_pt)
