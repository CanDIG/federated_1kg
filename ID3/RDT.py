import pprint
import random
import numpy as np

def RandomDecisionTree(D, X):

        R = BuildTreeStructure(X)
        UpdateStatistics(R, D)
        # No pruning for private version
        return R

# Only 2 children when splitting PCA results into two categories
# 1. > mean
# 2. <= mean
ValToChild = {0.0:0, 1.0:1}
m = len(ValToChild)

def BuildTreeStructure(X, max_h, curr_h):
        if X == [] or curr_h == max_h:
                return {'attribute':None, 'children':[], 'alpha':{}}
        else:
                F = random.choice(X)
                r = {'attribute':F, 'children':[]}
                c = []
                
                # Deep copy
                X = X[:]
                
                X.remove(F)
                for i in range(m):
                        c.append(BuildTreeStructure(X, max_h, curr_h+1))
                        r['children'].append(c[i])
        return r

def UpdateStatistics(r, D):
        for x in D:
                AddInstance(r, x)

def AddInstance(r, x):
        if not r['children'] == []:
               F = r['attribute']
               child = ValToChild[x[F]]
               c = r['children'][child]
               AddInstance(c, x)
        else:
                # r is a leaf node

                t = x['pop']
                
                if not 'alpha' in r:
                        r['alpha'] = {}
                if not t in r['alpha']:
                        r['alpha'][t] = 0
                r['alpha'][t] += 1

def AddNoise(r, pop_set, eps):
        if eps == -1:
            return
            
        if not r['children'] == []:
                for child in r['children']:
                        if AddNoise(child, pop_set, eps):
                                r['children'].remove(child)
        else:
                # r is a leaf node
                if 'alpha' in r:
                    for key in pop_set:
                        if not key in r['alpha']:
                            r['alpha'][key] = 0
                            
                        # add abs so that counts do not become negative
                        r['alpha'][key] += abs(np.random.laplace(scale=float(1)/eps))

                        
def rdtClassifyEnsemble(R, x):
        
        ensembleClassifier = {}
        
        for r in R:
                classifier = rdtClassify(r, x)
                
                for label, numer_denom in classifier.iteritems():
                        numer = numer_denom['numer']
                        denom = numer_denom['denom']
                        
                        if not label in ensembleClassifier:
                                ensembleClassifier[label] = {'numer':numer, 'denom':denom}
                        else:
                                ensembleClassifier[label]['numer'] += numer
                                ensembleClassifier[label]['denom'] += denom
                                
                        ensembleClassifier[label]['prob'] = ensembleClassifier[label]['numer']/float(ensembleClassifier[label]['denom'])

        highestProb = 0
        prediction = None

        for label, numer_denom_prob in ensembleClassifier.iteritems():
                prob = numer_denom_prob['prob']
                if prob > highestProb:
                        highestProb = prob
                        prediction = label

        return prediction

def rdtClassify(r, x):

        if 'alpha' in r:
                ret = {}
                denom = float(sum(r['alpha'].values()))
                for pop, numer in r['alpha'].iteritems():
                        ret[pop] = {'numer':numer, 'denom':denom}
                return ret
        else:
                F = r['attribute']
                child = ValToChild[x[F]]

                c = r['children'][child]
                return rdtClassify(c, x)
