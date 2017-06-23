from __future__ import print_function
from ga4gh.client import client

import pandas as pda
import numpy as np
from Test_ID3 import *


# Silence https warnings
# https://stackoverflow.com/questions/27981545/suppress-insecurerequestwarning-unverified-https-request-is-being-made-in-pytho
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)



def get_filter_regions():
    """
    Returns a list of regions. These regions are the SNPs that are kept
    after filtering [i.e. filter_genotype(G, I)].

    Querying this filtered set is much faster than querying the unfiltered
    regions and then filtering after.

    :return a list of regions.
    """
    
    filter_regions = [{'start': 94951137, 'chr': 'chr10', 'end': 94951138},
                      {'start': 94951708, 'chr': 'chr10', 'end': 94951709},
                      {'start': 89179334, 'chr': 'chr11', 'end': 89179335},
                      {'start': 89183935, 'chr': 'chr11', 'end': 89183936},
                      {'start': 89207230, 'chr': 'chr11', 'end': 89207231},
                      {'start': 89207617, 'chr': 'chr11', 'end': 89207618},
                      {'start': 89207714, 'chr': 'chr11', 'end': 89207715},
                      {'start': 89216311, 'chr': 'chr11', 'end': 89216312},
                      {'start': 89219122, 'chr': 'chr11', 'end': 89219123},
                      {'start': 89226144, 'chr': 'chr11', 'end': 89226145},
                      {'start': 89245479, 'chr': 'chr11', 'end': 89245480},
                      {'start': 89292354, 'chr': 'chr11', 'end': 89292355},
                      {'start': 94465006, 'chr': 'chr13', 'end': 94465007},
                      {'start': 27756081, 'chr': 'chr15', 'end': 27756082},
                      {'start': 27762242, 'chr': 'chr15', 'end': 27762243},
                      {'start': 27770275, 'chr': 'chr15', 'end': 27770276},
                      {'start': 27770519, 'chr': 'chr15', 'end': 27770520},
                      {'start': 27770688, 'chr': 'chr15', 'end': 27770689},
                      {'start': 27773344, 'chr': 'chr15', 'end': 27773345},
                      {'start': 27773546, 'chr': 'chr15', 'end': 27773547},
                      {'start': 27774025, 'chr': 'chr15', 'end': 27774026},
                      {'start': 27774139, 'chr': 'chr15', 'end': 27774140},
                      {'start': 27774223, 'chr': 'chr15', 'end': 27774224},
                      {'start': 27775395, 'chr': 'chr15', 'end': 27775396},
                      {'start': 27775542, 'chr': 'chr15', 'end': 27775543},
                      {'start': 27776830, 'chr': 'chr15', 'end': 27776831},
                      {'start': 27776933, 'chr': 'chr15', 'end': 27776934},
                      {'start': 27780205, 'chr': 'chr15', 'end': 27780206},
                      {'start': 27782225, 'chr': 'chr15', 'end': 27782226},
                      {'start': 27783389, 'chr': 'chr15', 'end': 27783390},
                      {'start': 27783656, 'chr': 'chr15', 'end': 27783657},
                      {'start': 27784332, 'chr': 'chr15', 'end': 27784333},
                      {'start': 27785103, 'chr': 'chr15', 'end': 27785104},
                      {'start': 27785640, 'chr': 'chr15', 'end': 27785641},
                      {'start': 27787160, 'chr': 'chr15', 'end': 27787161},
                      {'start': 27787250, 'chr': 'chr15', 'end': 27787251},
                      {'start': 27787597, 'chr': 'chr15', 'end': 27787598},
                      {'start': 27788131, 'chr': 'chr15', 'end': 27788132},
                      {'start': 27788715, 'chr': 'chr15', 'end': 27788716},
                      {'start': 27788723, 'chr': 'chr15', 'end': 27788724},
                      {'start': 27793664, 'chr': 'chr15', 'end': 27793665},
                      {'start': 27794165, 'chr': 'chr15', 'end': 27794166},
                      {'start': 27797066, 'chr': 'chr15', 'end': 27797067},
                      {'start': 27819629, 'chr': 'chr15', 'end': 27819630},
                      {'start': 27821100, 'chr': 'chr15', 'end': 27821101},
                      {'start': 27821101, 'chr': 'chr15', 'end': 27821102},
                      {'start': 27821298, 'chr': 'chr15', 'end': 27821299},
                      {'start': 27824706, 'chr': 'chr15', 'end': 27824707},
                      {'start': 27826226, 'chr': 'chr15', 'end': 27826227},
                      {'start': 27827888, 'chr': 'chr15', 'end': 27827889},
                      {'start': 27832922, 'chr': 'chr15', 'end': 27832923},
                      {'start': 27834622, 'chr': 'chr15', 'end': 27834623},
                      {'start': 27834954, 'chr': 'chr15', 'end': 27834955},
                      {'start': 27835585, 'chr': 'chr15', 'end': 27835586},
                      {'start': 27836397, 'chr': 'chr15', 'end': 27836398},
                      {'start': 27836908, 'chr': 'chr15', 'end': 27836909},
                      {'start': 27837934, 'chr': 'chr15', 'end': 27837935},
                      {'start': 27838106, 'chr': 'chr15', 'end': 27838107},
                      {'start': 27841157, 'chr': 'chr15', 'end': 27841158},
                      {'start': 27846285, 'chr': 'chr15', 'end': 27846286},
                      {'start': 27849873, 'chr': 'chr15', 'end': 27849874},
                      {'start': 27851695, 'chr': 'chr15', 'end': 27851696},
                      {'start': 27855407, 'chr': 'chr15', 'end': 27855408},
                      {'start': 27863026, 'chr': 'chr15', 'end': 27863027},
                      {'start': 27876806, 'chr': 'chr15', 'end': 27876807},
                      {'start': 27877351, 'chr': 'chr15', 'end': 27877352},
                      {'start': 27880730, 'chr': 'chr15', 'end': 27880731},
                      {'start': 27889773, 'chr': 'chr15', 'end': 27889774},
                      {'start': 27916819, 'chr': 'chr15', 'end': 27916820},
                      {'start': 27917875, 'chr': 'chr15', 'end': 27917876},
                      {'start': 27922040, 'chr': 'chr15', 'end': 27922041},
                      {'start': 27923058, 'chr': 'chr15', 'end': 27923059},
                      {'start': 27926546, 'chr': 'chr15', 'end': 27926547},
                      {'start': 27928793, 'chr': 'chr15', 'end': 27928794},
                      {'start': 27933111, 'chr': 'chr15', 'end': 27933112},
                      {'start': 27961520, 'chr': 'chr15', 'end': 27961521},
                      {'start': 27961725, 'chr': 'chr15', 'end': 27961726},
                      {'start': 27962943, 'chr': 'chr15', 'end': 27962944},
                      {'start': 27963179, 'chr': 'chr15', 'end': 27963180},
                      {'start': 27970358, 'chr': 'chr15', 'end': 27970359},
                      {'start': 27970502, 'chr': 'chr15', 'end': 27970503},
                      {'start': 27971212, 'chr': 'chr15', 'end': 27971213},
                      {'start': 27971944, 'chr': 'chr15', 'end': 27971945},
                      {'start': 27987566, 'chr': 'chr15', 'end': 27987567},
                      {'start': 27989431, 'chr': 'chr15', 'end': 27989432},
                      {'start': 27990143, 'chr': 'chr15', 'end': 27990144},
                      {'start': 27990991, 'chr': 'chr15', 'end': 27990992},
                      {'start': 27991647, 'chr': 'chr15', 'end': 27991648},
                      {'start': 28003951, 'chr': 'chr15', 'end': 28003952},
                      {'start': 28008205, 'chr': 'chr15', 'end': 28008206},
                      {'start': 28015727, 'chr': 'chr15', 'end': 28015728},
                      {'start': 28015771, 'chr': 'chr15', 'end': 28015772},
                      {'start': 28021762, 'chr': 'chr15', 'end': 28021763},
                      {'start': 28023060, 'chr': 'chr15', 'end': 28023061},
                      {'start': 28032089, 'chr': 'chr15', 'end': 28032090},
                      {'start': 28033724, 'chr': 'chr15', 'end': 28033725},
                      {'start': 28035649, 'chr': 'chr15', 'end': 28035650},
                      {'start': 28039754, 'chr': 'chr15', 'end': 28039755},
                      {'start': 28039966, 'chr': 'chr15', 'end': 28039967},
                      {'start': 28040215, 'chr': 'chr15', 'end': 28040216},
                      {'start': 28040586, 'chr': 'chr15', 'end': 28040587},
                      {'start': 28040660, 'chr': 'chr15', 'end': 28040661},
                      {'start': 28040910, 'chr': 'chr15', 'end': 28040911},
                      {'start': 28041242, 'chr': 'chr15', 'end': 28041243},
                      {'start': 28041399, 'chr': 'chr15', 'end': 28041400},
                      {'start': 28041472, 'chr': 'chr15', 'end': 28041473},
                      {'start': 28041514, 'chr': 'chr15', 'end': 28041515},
                      {'start': 28041718, 'chr': 'chr15', 'end': 28041719},
                      {'start': 28072828, 'chr': 'chr15', 'end': 28072829},
                      {'start': 28082916, 'chr': 'chr15', 'end': 28082917},
                      {'start': 28090886, 'chr': 'chr15', 'end': 28090887},
                      {'start': 28099178, 'chr': 'chr15', 'end': 28099179},
                      {'start': 42126818, 'chr': 'chr22', 'end': 42126819},
                      {'start': 42128342, 'chr': 'chr22', 'end': 42128343},
                      {'start': 42130277, 'chr': 'chr22', 'end': 42130278},
                      {'start': 78001595, 'chr': 'chr5', 'end': 78001596},
                      {'start': 78001689, 'chr': 'chr5', 'end': 78001690},
                      {'start': 78016363, 'chr': 'chr5', 'end': 78016364},
                      {'start': 78016364, 'chr': 'chr5', 'end': 78016365},
                      {'start': 78020226, 'chr': 'chr5', 'end': 78020227},
                      {'start': 78030772, 'chr': 'chr5', 'end': 78030773},
                      {'start': 78038104, 'chr': 'chr5', 'end': 78038105},
                      {'start': 78040417, 'chr': 'chr5', 'end': 78040418},
                      {'start': 78044123, 'chr': 'chr5', 'end': 78044124},
                      {'start': 78050754, 'chr': 'chr5', 'end': 78050755},
                      {'start': 78051115, 'chr': 'chr5', 'end': 78051116},
                      {'start': 78056184, 'chr': 'chr5', 'end': 78056185},
                      {'start': 78057408, 'chr': 'chr5', 'end': 78057409},
                      {'start': 78057474, 'chr': 'chr5', 'end': 78057475},
                      {'start': 78057821, 'chr': 'chr5', 'end': 78057822},
                      {'start': 78059332, 'chr': 'chr5', 'end': 78059333},
                      {'start': 78061914, 'chr': 'chr5', 'end': 78061915},
                      {'start': 78064548, 'chr': 'chr5', 'end': 78064549},
                      {'start': 78076375, 'chr': 'chr5', 'end': 78076376},
                      {'start': 78077044, 'chr': 'chr5', 'end': 78077045},
                      {'start': 78078456, 'chr': 'chr5', 'end': 78078457},
                      {'start': 78078498, 'chr': 'chr5', 'end': 78078499},
                      {'start': 78078607, 'chr': 'chr5', 'end': 78078608},
                      {'start': 78083698, 'chr': 'chr5', 'end': 78083699},
                      {'start': 78084120, 'chr': 'chr5', 'end': 78084121},
                      {'start': 78084934, 'chr': 'chr5', 'end': 78084935},
                      {'start': 78085543, 'chr': 'chr5', 'end': 78085544},
                      {'start': 78085903, 'chr': 'chr5', 'end': 78085904},
                      {'start': 78086854, 'chr': 'chr5', 'end': 78086855},
                      {'start': 78087349, 'chr': 'chr5', 'end': 78087350},
                      {'start': 78103783, 'chr': 'chr5', 'end': 78103784},
                      {'start': 78105298, 'chr': 'chr5', 'end': 78105299},
                      {'start': 78136052, 'chr': 'chr5', 'end': 78136053},
                      {'start': 78162503, 'chr': 'chr5', 'end': 78162504},
                      {'start': 78166472, 'chr': 'chr5', 'end': 78166473},
                      {'start': 78240689, 'chr': 'chr5', 'end': 78240690},
                      {'start': 78272458, 'chr': 'chr5', 'end': 78272459},
                      {'start': 78274127, 'chr': 'chr5', 'end': 78274128},
                      {'start': 78274224, 'chr': 'chr5', 'end': 78274225},
                      {'start': 78275019, 'chr': 'chr5', 'end': 78275020},
                      {'start': 78275045, 'chr': 'chr5', 'end': 78275046},
                      {'start': 78275833, 'chr': 'chr5', 'end': 78275834},
                      {'start': 78282751, 'chr': 'chr5', 'end': 78282752},
                      {'start': 78293558, 'chr': 'chr5', 'end': 78293559},
                      {'start': 17306828, 'chr': 'chr7', 'end': 17306829},
                      {'start': 17310144, 'chr': 'chr7', 'end': 17310145},
                      {'start': 17315981, 'chr': 'chr7', 'end': 17315982},
                      {'start': 17317738, 'chr': 'chr7', 'end': 17317739},
                      {'start': 17328306, 'chr': 'chr7', 'end': 17328307},
                      {'start': 17338391, 'chr': 'chr7', 'end': 17338392},
                      {'start': 17344552, 'chr': 'chr7', 'end': 17344553},
                      {'start': 99764273, 'chr': 'chr7', 'end': 99764274},
                      {'start': 99772862, 'chr': 'chr7', 'end': 99772863},
                      {'start': 99773559, 'chr': 'chr7', 'end': 99773560},
                      {'start': 99773878, 'chr': 'chr7', 'end': 99773879},
                      {'start': 99778316, 'chr': 'chr7', 'end': 99778317},
                      {'start': 99782141, 'chr': 'chr7', 'end': 99782142},
                      {'start': 12692363, 'chr': 'chr9', 'end': 12692364},
                      {'start': 12701749, 'chr': 'chr9', 'end': 12701750},
                      {'start': 12705431, 'chr': 'chr9', 'end': 12705432}]

    return filter_regions

def get_ga4gh_variants_dataframe(url, chrom, start, end):
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
    print ("chrom:{}, start:{}, end:{}".format(chrom, start, end))


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

        # Only return the bi-allelic snps
        if len(variant.reference_bases) > 1:
            continue
        if len(variant.alternate_bases) > 1:
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
    return pda.DataFrame(all_gts,columns=['index']
                         + [callset.name for callset in callSets])


def get_ga4gh_variants(servers, regions):
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

    num_servers = len(servers) - 1

    # for each server, query all regions then merge
    for j, server in enumerate(servers):

        G = pda.DataFrame()

        num_regions = len(regions) - 1

        # query a DataFrame of genotypes for all regions
        # G is the merged DataFrame for all regions

        # e.g.
        # region1
        #                            index  HG00099  HG001031
        #    0    10_94951137_94951138_C_A      0.0      0.0    
        # region2
        #                            index  HG00099  HG001031
        #    0    10_94951708_94951709_C_T      0.0      0.0    
        # G = region1 + region2
        #                            index  HG00099  HG001031
        #    0    10_94951137_94951138_C_A      0.0      0.0    
        #    1    10_94951708_94951709_C_T      0.0      0.0    
        for i, reg in enumerate(regions):
            # merge regions
            df = get_ga4gh_variants_dataframe(server, reg['chr'], reg['start'], reg['end'])
            G = G.append(df)
            print(j, '/', num_servers, '|',  i, '/', num_regions, '|', 'G:', G.shape, '\n')

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
        if init:
            AllG = G
            init = False
        else:
            AllG = AllG.merge(G, how='outer', left_on='index', right_on='index')
        print('AllG:', AllG.shape, '\n')

    return AllG

def ga4gh_Bind_Data(G, I):
    """
    Modified Bind_Data(G, I, filename) to use a DataFrame from merged ga4gh data
    """

    from Test_ID3 import *

    # Remove variants/callsets where there is no genotype information
    G = G.dropna(axis=0)

    # Drop the "index" column -> just the genotypes
    G = G.drop('index', 1)

    # List of callsets/samples
    samples = [str(sample) for sample in G.columns]

    G = G.as_matrix()
    G = np.transpose(G)
    print(G.shape)
    

    rows, cols = G.shape
    attr = []

    #generating pseudo-headers
    for x in range(cols):
    	attr.append("SNP"+str(x)) 
   
    data = [zip(attr, G[ind]) for ind, datum in enumerate(G)]
     
    #reading target_class
    pops, subpops = population_dictionaries('samples-pops-subpops.csv')
    
    samples = samples[:I]

    pop_lst = [pops[sample] for sample in samples]
    
    data  = [dict(d, pop=n) for d, n in zip(data, pop_lst)]
   

    attr.append('pop')
    target_attr =  attr[-1]
 
    return data, attr, target_attr
