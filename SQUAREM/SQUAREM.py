#!/usr/bin/env python
from __future__ import print_function
from cyvcf2 import VCF

import numpy as np
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


def normalize_rows(mat):
    for i in range(mat.shape[0]):
        mat[i, :] = mat[i, :]/np.sum(mat[i, :])
    return mat


def Get_Synthetic_Genotypes(nindividuals, nvars, npops):
    populations = list(range(npops))
    popnames = ["pop_"+str(pop) for pop in populations]
    subpopnames = ["subpop_"+str(pop) for pop in populations]
    samplenames = ["ind_"+str(i) for i in range(nindividuals)]
    nnoise = int(0.1 * nvars)

    # simple case - everyone just gets one ancestry
    # more complex case - mixture
    pop_assignments = np.random.choice(populations,
                                       nindividuals, replace=True)
    print("Pop_assignments:")
    print(pop_assignments)

    # create the ancestral genotypes; eg w/ 3 ancestries
    # and 9 variants we'll get:
    # [ 1,1,1, 0,0,0, 0,0,0 ]
    # [ 0,0,0, 1,1,1, 0,0,0 ]
    # [ 0,0,0, 0,0,0, 1,1,1 ]

    ancestral_genotypes = np.zeros((npops, nvars), dtype=np.int)
    nvar_per_pop = (nvars + npops-1)//npops
    for pop in populations:
        s = pop*nvar_per_pop
        e = (pop+1)*nvar_per_pop
        ancestral_genotypes[pop, s:e] = 1

    G = np.zeros((nvars, nindividuals), dtype=np.int)
    popdict = {}
    subpopdict = {}
    for i in range(nindividuals):
        pop = pop_assignments[i]
        G[:, i] = ancestral_genotypes[pop, :]

        popdict[samplenames[i]] = popnames[pop]
        subpopdict[samplenames[i]] = subpopnames[pop]

        # add random noise
        randomize = np.random.choice(range(nvars), nnoise, replace=False)
        G[randomize, i] = 1 - G[randomize, i]

    return G, samplenames, popdict, subpopdict


def Get_Genotypes_from_file(filename, I=None, J=None,
                            mystart=0, myend=90000, chrom="20"):
    """
    read genotype into a matrix G and filter out the variants
    with all ones or zeros.
    """
    vcf = VCF(filename)
    samples = vcf.samples
    region = chrom+":"+str(mystart)+"-"+str(myend)

    gts = np.array([np.copy(v.gt_types) for v in vcf(region)
                    if len(v.ALT) == 1 and v.num_unknown == 0])
    nrows, ncols = gts.shape

    if I:
        gts = gts[:, np.random.choice(ncols, I, replace=False)-1]
    if J:
        gts = gts[np.random.choice(nrows, J, replace=False)-1, :]

    badrows = np.where(gts == 2)[0]
    gts = np.delete(gts, badrows, axis=0)
    gts = np.clip(gts, 0, 2)

    mask = np.all(gts > 0., axis=1)
    gts = gts[~mask, :]
    mask = np.all(gts == 0., axis=1)
    gts = gts[~mask, :]
    mask = np.all(gts == 0., axis=0)
    gts = gts[:, ~mask]
    return gts, samples


def initialize_F(G, K, pops, samples):
    """
    Create an initial random F
    """
    # convert to values {1,0} with 1 for an element > 0
    gts = G.copy()
    gts = gts.transpose()

    gts = np.where(gts > 0, 1, 0)

    nrows, ncols = gts.shape

    pop_dist = {}
    # create a count of samples in each subpop
    for sample in samples:
        if pops[sample] in pop_dist:
            pop_dist[pops[sample]] += 1
        else:
            pop_dist[pops[sample]] = 1

    # comute the f matrix now
    F = np.empty((K, gts.shape[0]), dtype=np.float)

    J, I = gts.shape

    for j in range(J):
        entries = np.where(gts[j] > 0)[0]
        local_count = {}
        local_count = local_count.fromkeys(pop_dist.keys(), 0)
        for ind in entries:
            local_count[pops[samples[ind]]] += 1

        for k, pop in enumerate(local_count.keys()):
            F[k, j] = local_count[pop]/float(pop_dist[pop])

    return F


def liklihood(G, Q, F):
    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)
    L = np.sum(np.multiply(G, np.log(QF)) + np.multiply((2.-G), np.log(QF2)))
    return L


def initialize_Q(G, F, rounds=10):
    K, J = F.shape

    # seeds for symmetric dirichlet.
    seeds = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.7, 0.8, 0.9, 1.]
    I, J = G.shape
    alpha = np.zeros(K)
    alpha[:] = 1./K
    Q = np.random.dirichlet(alpha, size=I)

    Q_max = Q.copy()
    L_0 = liklihood(G, Q, F)
    print ("L_0 is:{}".format(L_0))

    for x in range(rounds):
        alpha = np.zeros(K)
        alpha[:] = seeds[x]
        Q = np.random.dirichlet(alpha, size=I)

        Q = normalize_rows(Q)

        L_1 = liklihood(G, Q, F)
        print ("init {}, likelihood:{}".format(x, L_1))
        if L_1-L_0 > 0.:
            Q_max = np.matrix.copy(Q)

    return Q_max


def Qupdate(G, F, Q):
    """
    single update step for Q
    """
    I, J = G.shape

    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)

    # E-Step: update a_ijk and b_ijk
    a = np.einsum('ik,kj->ijk', Q, F)
    a = a / QF[:, :, np.newaxis]
    b = np.einsum('ik,kj->ijk', Q, 1.-F)
    b = b / QF2[:, :, np.newaxis]

    # M-Step:updating Q
    term1 = np.einsum('ij,ijk->ik', G, a)
    term2 = np.einsum('ij,ijk->ik', 2.-G, b)
    Q = (term1 + term2)/(2.*J)

    # prevents accumulation of precision-errors at a single value
    Q = normalize_rows(Q)

    return Q


def Fupdate(G, F, Q):
    """
    single update step for F
    """
    I, J = G.shape

    QF = np.matmul(Q, F)
    QF2 = np.matmul(Q, 1-F)

    # E-Step: update a_ijk and b_ijk
    a = np.einsum('ik,kj->ijk', Q, F)
    a = a / QF[:, :, np.newaxis]
    b = np.einsum('ik,kj->ijk', Q, 1.-F)
    b = b / QF2[:, :, np.newaxis]

    # M-Step: Updating F
    term1 = np.einsum('ij,ijk->kj', G, a)
    term2 = np.einsum('ij,ijk->kj', 2.-G, b)
    F = term1/(term1+term2)
    F = np.clip(F, 0., 1.)

    return F


def SQUARED_Update(M0, M1, M2):
    # Compute steplength using S3.
    R = M1 - M0
    V = M2 - M1 - R
    a = -1.*np.sqrt((R*R).sum()/(V*V).sum())

    if a > -1:
        a = -1.

    a_adjusted = False
    while not a_adjusted:
        M = (1+a)**2*M0 - 2*a*(1+a)*M1 + a**2*M2
        if (M < 0.).any() or (M > 1.).any():
            a = (a-1)/2.
        else:
            a_adjusted = True

    if np.abs(a+1) < 1e-4:
        a = -1.

    # if this accelerated step fails for some reason,
    # stick with the first non-accelerated step.
    if np.isnan(M).any():
        M = M1.copy()

    return M


def Q_SQUARED_Update(G, F, Q0):
    Q1 = Qupdate(G, F, Q0)
    Q2 = Qupdate(G, F, Q1)

    Q = SQUARED_Update(Q0, Q1, Q2)
    Q = normalize_rows(Q)

    return Q


def F_SQUARED_Update(G, F0, Q):
    F1 = Fupdate(G, F0, Q)
    F2 = Fupdate(G, F1, Q)

    F = SQUARED_Update(F0, F1, F2)

    return F


def SQUAREM(G, F0, Q0,  K=5, e=10**-4, maxiters=None):
    I, J = G.shape

    notconverged = True
    iteration = 0

    L_0 = L_1 = 0.0

    # Computing likelihood for the first time
    L_0 = liklihood(G, Q0, F0)
    print("Initial Likelihood is :{}".format(L_0))

    itertime = time.time()

    while notconverged and iteration <= maxiters:
        # update F, given Q
        F = F_SQUARED_Update(G, F0, Q0)
        F0 = Fupdate(G, F, Q0)

        # update Q, given F
        Q = Q_SQUARED_Update(G, F0, Q0)
        Q0 = Qupdate(G, F0, Q)

        # Compute likelihood once every 10 iterations
        if (iteration+1) % 10 == 0:
            L_0 = liklihood(G, Q, F)
            L_1 = liklihood(G, Q0, F0)

            itertime, lasttime = time.time(), itertime

            notconverged = (abs(L_1 - L_0) >= e)
            print ("{}, Likelihood:{}, delta:{}, time:{}".
                   format(iteration, L_1, abs(L_1-L_0), itertime-lasttime))

        if maxiters and iteration >= maxiters:
            break
        iteration += 1

    return Q


def parse_args():
    parser = argparse.ArgumentParser(description='An SQUAREM implementation to infer population structure and assign individuals to populations using genotype data')
    parser.add_argument('--outputfile', help='a file where admixture co-efficients to be written', default="Q.log.txt")
    parser.add_argument('--K', help='The number of populations(clusters)', type=int, default=5)
    parser.add_argument('--I', help='The number of variants', type=int, default=5)
    parser.add_argument('--J', help='The number of individuals', type=int, default=5)
    parser.add_argument('--vcf', help='VCF file input', default="filteredchr20.recode.vcf.gz")
    parser.add_argument('--ancestries', help='ancestries text file', default='samples-pops-subpops.csv')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    np.set_printoptions(threshold='nan')
    args = parse_args()
    outputfilename = args.outputfile
    I = args.I
    J = args.J
    K = args.K
    testdata = args.test
    eps = 1.e-4

    if not testdata:
        G, samps = Get_Genotypes_from_file(filename=args.vcf,
                                           mystart=0, myend=63025520,
                                           chrom=str(20), I=2504)
        pops, subpops = population_dictionaries(args.ancestries)
    else:
        G, samps, pops, subpops = Get_Synthetic_Genotypes(I, J, K)

    G = np.transpose(G)

    F = initialize_F(G, K, pops, samps)
    Q = initialize_Q(G, F, rounds=10)

    Q = SQUAREM(G, F, Q, K, e=eps, maxiters=100)

    np.savetxt(outputfilename, Q, fmt='%.6f')
    print("Output written to the file:{}".format(outputfilename))
