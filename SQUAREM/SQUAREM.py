from __future__ import print_function
from cyvcf2 import VCF

import mpmath as mp
import numpy as np 
import sys
import pickle
import random 
import fileinput
import csv
import time
import argparse

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



#read genotype into a matrix G and filter out the variants with all ones or zeros.  
def Get_Genotypes_from_file(filename, I = None, J = None, mystart=0, myend=90000, chrom="20"):

    vcf = VCF(filename)
    region = chrom+":"+str(mystart)+"-"+str(myend)



    gts = np.array([np.copy(v.gt_types) for v in vcf(region) if len(v.ALT) == 1 and v.num_unknown == 0 ])

    nrows, ncols = gts.shape

    if I:
    	gts = gts[ :, np.random.choice(ncols, I, replace=False)-1 ]

    if J:
	gts = gts[ np.random.choice(nrows, J, replace=False)-1, : ]
 
    

    badrows = np.where(gts == 2)[0]
    gts = np.delete(gts, badrows, axis=0)
    gts = np.clip(gts, 0, 2)


    mask = np.all(gts > 0., axis = 1)
    gts = gts[~mask, :]
    mask = np.all(gts == 0., axis = 1)
    gts = gts[~mask, :]
    mask = np.all(gts == 0., axis = 0)
    gts = gts[:, ~mask]
    #print ("the size of gts:{}".format(gts.shape))
    return gts


#generate F and Q. 
def initialize_F(G, K = 5,filename = "filteredchr20.recode.vcf.gz"):


    vcf = VCF(filename)

    #convert to values {1,0} with 1 for an element > 0
    gts = G.copy()
    gts =  gts.transpose()

    gts = np.where(gts > 0, 1, 0)

    samples = vcf.samples
    pop_size = len(samples)

    nrows, ncols = gts.shape

    pop_dist = {}
    pops, subpops = population_dictionaries('samples-pops-subpops.csv')


    #create a count of samples in each subpop
    for sample in samples:
        if pops[sample] in pop_dist: pop_dist[pops[sample]]+= 1
        else: pop_dist[pops[sample]] = 1



    #comute the f matrix now
    F = np.empty((K, gts.shape[0]), dtype=np.float)

    J,I = gts.shape

    for j in range(J):
        entries = np.where(gts[j] > 0)[0]
        #print("entries:{}".format(entries))
        local_count = {}
        local_count = local_count.fromkeys(pop_dist.keys(),0)
        for ind in entries:
                local_count[pops[samples[ind]]] += 1


        #print("local_counts:{}".format(local_count))
        for k, pop  in enumerate(local_count.keys()):
                F[k,j]  =  local_count[pop]/float(pop_dist[pop])

      
    
    return F

def initialize_Q(G, F, rounds = 10 ):


        K, J = F.shape

        #seeds for symmetric dirichlet. 
        seeds = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.7,0.8, 0.9, 1.]
        I, J = G.shape
        alpha = np.zeros(K)
        alpha[:]  = 1./K
        Q = np.random.dirichlet(alpha, size = I)
       
 
        Q_max = Q.copy()
        QF = np.matmul(Q,F)
    	QF2 = np.matmul(Q,1-F)
  
        L_0 = np.sum(np.multiply(G, np.log(QF)) + np.multiply( (2.-G), np.log(QF2) ) )
        print ("L_0 is:{}".format(L_0))

	for x in range(rounds):
 		alpha = np.zeros(K)
        	alpha[:]  = seeds[x] #np.random.choice(seeds, 1, replace=True)
        	Q = np.random.dirichlet(alpha, size = I)
       
                for i in range(I):
        		Q[i,:] = Q[i,:]/np.sum(Q[i,:])
 	        QF = np.matmul(Q,F)
    		QF2 = np.matmul(Q,1-F)
                

		L_1  = np.sum(np.multiply(G, np.log(QF)) + np.multiply( (2.-G), np.log(QF2) ) ) 

                print ("init {}, likelihood:{}".format(x, L_1))
 		if L_1-L_0 > 0. :
			Q_max = np.matrix.copy(Q)

	return Q_max		


#single update step for Q
def Qupdate(G,F,Q):

	I, J = G.shape
    
        QF = np.matmul(Q,F)
    	QF2 = np.matmul(Q,1-F)
	
 #E-Step: update a_ijk and b_ijk
        a = np.einsum('ik,kj->ijk', Q, F)  
        a = a / QF[:,:,np.newaxis]
        b = np.einsum('ik,kj->ijk', Q, 1.-F)
        b = b / QF2[:,:,np.newaxis]

        #M-Step:updating Q 
        term1 = np.einsum('ij,ijk->ik', G, a)
        term2 = np.einsum('ij,ijk->ik', 2.-G, b)
        Q = (term1 + term2)/(2.*J)

        #prevents accumulation of precision-errors at a single value  
        for i in range(I):
            Q[i,:] = Q[i,:]/np.sum(Q[i,:])
        
        return Q

#single update step for F
def Fupdate(G,F,Q):

	I, J = G.shape
    
        QF = np.matmul(Q,F)
    	QF2 = np.matmul(Q,1-F)
	
 #E-Step: update a_ijk and b_ijk
        a = np.einsum('ik,kj->ijk', Q, F)   # Einstein summation notation lets you express multiply-and-sum operations in a very general way
        a = a / QF[:,:,np.newaxis]
        b = np.einsum('ik,kj->ijk', Q, 1.-F)
        b = b / QF2[:,:,np.newaxis]
        										
        #M-Step: Updating F..
        term1 = np.einsum('ij,ijk->kj', G, a ) 
        term2 = np.einsum('ij,ijk->kj', 2.-G, b ) 
        F = term1/(term1+term2)
        F = np.clip(F, 0., 1.)
        
        return F


def Q_SQUARED_Update(G, F, Q0):

    
        Q1 = Qupdate(G,F,Q0)
	Q2 = Qupdate(G,F,Q1)
  

        #Compute steplength using S3.
        R = Q1 - Q0
        V = Q2 - Q1 - R
        a = -1.*np.sqrt((R*R).sum()/(V*V).sum())

        if a>-1:
            a = -1.
 
        a_adjusted = False
        while not a_adjusted:
        	Q = (1+a)**2*Q0 - 2*a*(1+a)*Q1 + a**2*Q2
            	if (Q < 0.).any() or (Q > 1.).any():
                	a = (a-1)/2.

                else:
                	a_adjusted = True
          
        if np.abs(a+1)<1e-4:
        	a = -1.


        # if this accelerated step fails for some reason, stick with the first non-accelerated step.
        if np.isnan(Q).any():
        	Q = Q1.copy()

      

        #print("Qsteplength:{}".format(a))
        
        

	for i in range(Q.shape[0]):
            Q[i,:] = Q[i,:]/np.sum(Q[i,:])

        return Q 


def F_SQUARED_Update(G, F0, Q):

    
        F1 = Fupdate(G,F0,Q)
	F2 = Fupdate(G,F1,Q)
  

        #Compute steplength using S3.
        R = F1 - F0
        V = F2 - F1 - R
        a = -1.*np.sqrt((R*R).sum()/(V*V).sum())

        if a>-1:
            a = -1.
 
        a_adjusted = False
        while not a_adjusted:
        	F = (1+a)**2*F0 - 2*a*(1+a)*F1 + a**2*F2
            	if (F < 0.).any() or (F > 1.).any():
                	a = (a-1)/2.

                else:
                	a_adjusted = True
          
        if np.abs(a+1)<1e-4:
        	a = -1.


        # if this accelerated step fails for some reason, stick with the first non-accelerated step.
        if np.isnan(F).any():
        	F = F1.copy()

      

       # print("steplength:{}".format(a))
        
        

        return F 



def SQUAREM(G, F0, Q0,  K = 5, e = 10**-4, maxiters=None):

    I, J = G.shape
    
    notconverged = True
    iteration = 0

   
    L_0 = L_1 = 0.0
    
    #Computing likelihood for the first time
  
    QF = np.matmul(Q0,F0)
    QF2 = np.matmul(Q0,1-F0)
    L_0 = np.sum( np.multiply(G, np.log(QF)) + np.multiply( (2.-G), np.log(QF2) ) )
    print("Initial Likelihood is :{}".format(L_0))
    #print("G:{}, F:{}, Q:{}, QF:{}, QF2:{}".format(G.shape, F.shape, Q0.shape, QF.shape, QF2.shape))


    itertime = time.time()
    
    while notconverged and iteration <= maxiters:
        
        #update F, given Q
        F = F_SQUARED_Update(G, F0, Q0)
        F0 = Fupdate(G,F,Q0)
        
        #update Q, given F
        Q = Q_SQUARED_Update(G, F0, Q0)
        Q0 = Qupdate(G,F0,Q)

   
        # Compute likelihood once every 10 iterations
        if (iteration+1)%10==0:

		QF = np.matmul(Q,F)
        	QF2 = np.matmul(Q,1-F)
        	L_0 = np.sum( np.multiply(G, np.log(QF)) + np.multiply( (2.-G), np.log(QF2) ) )
  
        	QF = np.matmul(Q0,F0)
   		QF2 = np.matmul(Q0,1-F0)
		L_1 = np.sum( np.multiply(G, np.log(QF)) + np.multiply( (2.-G), np.log(QF2) ) ) 
                
                itertime = time.time()-itertime

        	notconverged = (abs(L_1 - L_0) >= e)
        	print ("{}, Likelihood:{}, delta:{}, time:{}".format(iteration, L_1, abs(L_1-L_0), itertime))    
                

        if maxiters and iteration >= maxiters:
            break
        iteration += 1
            
            

    return Q     


def parse_args():
	parser = argparse.ArgumentParser(description='An SQUAREM implementation to infer population structure and assign individuals to populations using genotype data')
	parser.add_argument('--outputfile', nargs=1, help='a file where admixture co-efficients to be written', default = ["Q.log.txt"])
        parser.add_argument('--K', nargs=1, help='The number of populations(clusters)', type=int, default =  [5])
        #parser.add_argument('--I', nargs=1, help='The number of bio_samples to choose', type=int, default = [2504]) **add these later**
        #parser.add_argument('--J', nargs=1, help='The number of random variants to choose from a sample', type=int, default = [None])
 	args = parser.parse_args()
        return args

if __name__ == "__main__":

    np.set_printoptions(threshold='nan')
    args = parse_args()
    outputfilename =  args.outputfile[0]
    I = 2504
    J = None
    K = args.K[0]
    eps = 1.e-4
    #random.seed(50)
    
    G  = Get_Genotypes_from_file(filename="filteredchr20.recode.vcf.gz", mystart=0, myend=63025520, chrom=str(20), I = I)  
    G = np.transpose(G)
   
    F  = initialize_F(G, K,filename = "filteredchr20.recode.vcf.gz") 
    Q =  initialize_Q(G, F, rounds = 10)
 
    Q = SQUAREM(G, F, Q, K, e = eps, maxiters= 11)
  
    np.savetxt( outputfilename, Q, fmt='%.6f') 
    print("Output written to the file:{}".format( outputfilename))
    




