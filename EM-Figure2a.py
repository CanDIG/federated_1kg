from __future__ import print_function
import ga4gh.client.client as client

import numpy as np 
import sys

callset_names = []

def Get_Genotypes(I = 10, J = 5,mystart=0,myend=90000):

	baseURL = "http://1kgenomes.ga4gh.org"

	cl = client.HttpClient(baseURL)


	release = None
	for variant_set in cl.search_variant_sets(dataset_id=cl.search_datasets().next().id):
    		if variant_set.name == "phase3-release":
        		release = variant_set
        

	all_call_sets = list(cl.search_call_sets(release.id))
	
	call_set_ids = []
	#Callsets IDs for first I samples
	for  x in range(I):
		callset_names.append(all_call_sets[x].name)
    		call_set_ids.append(str(all_call_sets[x].id))
    
	reference_set = cl.search_reference_sets().next()

	references = [r for r in cl.search_references(reference_set_id=reference_set.id)]

	chr1 = filter(lambda x: x.name == "20", references)[0]

	example_variants = list(cl.search_variants(variant_set_id=release.id, start=mystart,end=myend, reference_name=chr1.name, call_set_ids=call_set_ids))
        
        
        
         #taking first J SNPs where a SNP/MNP must appear at least in a single individual and Alt > 1
        SNPs = filter(lambda x: len(x.alternate_bases) == len(x.reference_bases) and len(x.alternate_bases) == 1 and ([(call.genotype[0] + call.genotype[1]) for call in x.calls]).count(0) < I and ([(call.genotype[0] + call.genotype[1])  for call in x.calls].count(2) < I), example_variants)[:J]
       
        G = np.zeros((I,min(J,len(SNPs))))
 
	for j in range(len(SNPs)):
               SNP = SNPs[j] 
               for i in range(I):
                G[i, j] = (SNP.calls[i].genotype[0]+SNP.calls[i].genotype[1])
               	
                  
        
	return G



def EM(I = 10, J = 5, K = 3, e = 10**-4, mystart = 0, myend = 100000):


 #-- initialize F^0 
 F = np.random.rand(K,J)

 #-- initialize Q^0
 Q = np.empty((I,K))


 for i in range(I):
  Q[i] = np.random.dirichlet(np.ones(K), size=1)

 
 G = Get_Genotypes(I,J,mystart,myend)
 J = len(G[0])
 
#Computing initially L^0(Q,F)
 L_0 = L_1 =  0.0

 QF = np.matmul(Q,F)
 QF2 = np.matmul(Q,1-F)

 for i in range(I): 
  for j in range(J): 
   L_0 += (G[i,j]*np.log(QF[i,j]))+((2-G[i,j])*np.log(QF2[i,j]))
  
      
 a = np.zeros((I,J,K))
 b = np.zeros((I,J,K))
 
 notconverged = True

 while notconverged:
  
  #E-Step: update a_ijk and b_ijk
  for i in range(I):
   for j in range(J):
    for k in range(K):  
     a[i,j,k] =  (Q[i,k]*F[k,j])/QF[i,j]
     b[i,j,k] =  (Q[i,k]*(1-F[k,j]))/QF2[i,j]
  
 
  #M-Step: Updating F and Q 
  for k in range(K):
   for j in range(J):
    term1 = sum([G[i,j]*a[i,j,k] for i in range(I)])
    F[k,j] = term1/(term1 + (sum ( [(2-G[i,j])*b[i,j,k] for i in range(I)])) )
    if(F[k,j] > 1.0 and F[k,j] < 1.1):
    	F[k,j] = 1.0
    
  for i in range(I):
   for k in range(K):
    Q[i,k] = (1.0/(2.0*J))*(sum([ (G[i,j]*a[i,j,k])+((2-G[i,j])*b[i,j,k]) for j in range(J)])) 
   if(sum(Q[i]) > 1.0):
    Q[i] = Q[i]/sum(Q[i])	


  #Pre-computing sum_k(Q[i,k]*F[k,j] and sum_k(Q[i,k]*1-F[k,j]
  QF = np.matmul(Q,F)
  QF2 = np.matmul(Q,1-F)
  

  #Computing the likelihood with updated estimations
  L_1 = 0.0
  for i in range(I): 
   for j in range(J):
        #dealing with -inf or NaN
        if(QF[i,j] < 10**-300):
        	term1 = mp.log(mp.mpf(QF[i,j]))
        else: 
		term1 = np.log(QF[i,j])
        if(QF2[i,j] < 10**-300):
        	term2 = mp.log(mp.mpf(QF2[i,j]))
        else: 
		term2 = np.log(QF2[i,j])

	L_1 += (G[i,j]*term1)+((2-G[i,j])*term2)
    
  notconverged = ((L_1 - L_0) >= e)
  print ("L_1:{}, L_0:{}, L1-L0:{}, ntconverged:{}".format(L_1, L_0,(L_1-L_0), notconverged))	
  L_0 = L_1

				 
 return Q 

if __name__ == "__main__": 
    np.set_printoptions(threshold='nan')
    Q = EM(I = 60, J = 550, K = 5, e = 10**-5, mystart =  int(sys.argv[1]), myend = int(sys.argv[2]))
    print("Resultant Q:{}".format(Q))
    


