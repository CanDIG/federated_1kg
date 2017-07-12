from __future__ import print_function
from ga4gh.client import client

import pandas as pda
import numpy as np

#from threading import Thread
from multiprocessing import Process, Manager

# Silence https warnings
# https://stackoverflow.com/questions/27981545/suppress-insecurerequestwarning-unverified-https-request-is-being-made-in-pytho
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import time


def get_ga4gh_variants_dataframe(url, chrom, start, end, results, snps_only):
    """
    Returns a DataFrame of genotypes within the requested coordinates for all
    callsets.
    
    e.g.
                            index  HG00099  HG001031
    0    10_94951137_94951138_C_A      0.0      0.0    
    1    10_94951708_94951709_C_T      0.0      0.0    
    2    11_89179334_89179335_T_C      0.0      0.0    
    3    11_89183935_89183936_G_A      0.0      0.0    
    4    11_89207230_89207231_T_A      0.0      0.0    
    5    11_89207617_89207618_T_A      0.0      0.0    
    6    11_89207714_89207715_C_A      0.0      0.0    
    7    11_89216311_89216312_A_C      0.0      0.0    
    8    11_89219122_89219123_T_A      0.0      0.0
    (...)
    [XX rows x YY columns]

    XX variants x YY callsets.

    index = <chrom>_<start>_<end>_<ref>_<alt>

    :param str url: The url of the ga4gh server.
    :param str chrom: The chromosome for the region of interest.
    :param int start: The start position for the region of interest.
    :param str end: The end position for the region of interest.

    :return A DataFrame of genotypes within the requested coordinates for all
    callsets.
    rtype: DataFrame
    """

    chrom = chrom.replace('chr','')
    region = chrom+":"+str(start)+"-"+str(end)
    print ("server:{}, region {}:{}-{}".format(url, chrom, start, end))

    try:
	    httpClient = client.HttpClient(url)
	    
	    # Get the datasets on the server.
	    datasets = list(httpClient.search_datasets())
	    
	    # TODO: Assumption - uses the first dataset.
	    # Get the variantSets in the first dataset.
	    variantSets = list(httpClient.search_variant_sets(
		dataset_id=datasets[0].id))

	    # TODO: Assumption - uses the first variantset.
	    # Get the variants in the interval [<start>, <end>) on chromosome <chrom>
	    # in the first variantSet.
	    callSets = list(httpClient.search_call_sets(variantSets[0].id))
	    
	    iterator = httpClient.search_variants(
		variant_set_id=variantSets[0].id,
		reference_name=chrom, start=start, end=end,
		call_set_ids=[callset.id for callset in callSets])

	    all_gts = []
	    
	    for variant in iterator:

		if snps_only and len(variant.reference_bases) > 1 and len(variant.alternate_bases) > 1:
		# Only return the bi-allelic snps
		     continue
	       

		# Use var_id as the index for the DataFrame
		# This will be used as the key to join on
		# var_id = <chrom>_<start>_<end>_<ref>_<alt>
		var_id = "_".join([
		    variant.reference_name, str(variant.start), str(variant.end),
		    variant.reference_bases, ",".join(variant.alternate_bases)])

		# Since genotypes are restricted to bi-allelic snps, the possible
		# genotypes should be 0/0, 0/1, 1,1
		# Summing this -> 0, 1, 2 are the possible genotype values
		# gts = row of the DataFrame
		#     = [var_id, genotype_callset1, genotype_callset2, ...]
		gts = [var_id] + [int(sum(call.genotype)) for call in variant.calls]
		all_gts.append(gts)


	    # columns = [var_id, callset1, callset2, ...] 
	    #print("key:{}".format(url+region))
	    df =  pda.DataFrame(all_gts,columns=['index'] + [callset.name for callset in callSets])
	    results[url+region]  = df
  
    except: print("Can not query the region:{} from server:{}".format(region,url)); raise

def get_ga4gh_variants(servers, regions, snps_only):
    """
    Returns a DataFrame of genotypes within the requested coordinates for all
    callsets. The data is merged from multiple ga4gh servers and multiple
    regions.

    :param list servers: A list of ga4gh servers <str> to query.
    :param list regions: A list of regions to query.
    e.g. [region1, region2, ...]
         region1 = {'chrom':<chrom>, 'start':<start>, 'end':<end>}
         ...

    return: a DataFrame of genotypes within the requested coordinates for all
    callsets.
    rtype: DataFrame
    """

    # When init == True,
    init = True


    # for each server, query all regions then merge

    manager = Manager()
    results = manager.dict()   
    jobs  = {}

    for i, reg in enumerate(regions):

        num_regions = len(regions) - 1
        
        for j, server in enumerate(servers):
            # create jobs from regions
            d = {'url':server, 'chr':str(reg['chr']), 'start':reg['start'], 'end':reg['end']}
            #convert region to a dict key
            key = tuple(sorted(d.items()))
            jobs[key] = None

        print(len(jobs), ' parallel jobs created', '\nQuerying:')
     
        while len(jobs):
            for key in jobs:
                    job = dict( key )
        	    p = Process(target=get_ga4gh_variants_dataframe, args=(job['url'], job['chr'], job['start'], job['end'],  results, snps_only))
                    jobs[key] = p
                    p.start()
                        
    	    for j in jobs.values(): j.join()
         
            #remove all the jobs which exitted with no error 
            for k,v in jobs.items():
                    if v.exitcode == 0:del jobs[k] 
            print(len(jobs), ' jobs left')   
         

        # G is a merged DataFrame for all regions for one server
        # Merge each G into AllG (i.e. a merge across callsets for each server).
        # Merge on the index = <chrom>_<start>_<end>_<ref>_<alt>
        # If the same callset exists on multiple servers, all will be kept.
            
        # e.g.
        # server1
        #                            index  HG00099
        #    0    10_94951137_94951138_C_A      0.0
        #    1    10_94951708_94951709_C_T      0.0
        # server2
        #                            index  HG001031
        #    0    10_94951137_94951138_C_A      0.0    
        #    1    10_94951708_94951709_C_T      0.0    
        # AllG = server1 + server2
        #                            index  HG00099  HG001031
        #    0    10_94951137_94951138_C_A      0.0      0.0    
        #    1    10_94951708_94951709_C_T      0.0      0.0 

          
     

    
    for server in servers:
        G = pda.DataFrame()
        for key, val in results.items():
        	if server in key: G = G.append(val)

        if init: AllG = G; init = False
        else: AllG = AllG.merge(G, how='outer', left_on='index', right_on='index')

    
    return AllG


#query ancestry info:
def get_ga4gh_subpops(baseURL):

    httpclient = client.HttpClient(baseURL)
    datasets = list(httpclient.search_datasets())
    datasetId=datasets[0].id
    individuals = httpclient.search_individuals(datasetId)
    ancestry_dict = { i.name: i.description for i in individuals }
    return ancestry_dict


def get_ancestry_info(servers):
     ancestry_dict = {}
     for server in servers:       
            ancestry_part = get_ga4gh_subpops(server)
            ancestry_dict.update(ancestry_part)
 
     return ancestry_dict
