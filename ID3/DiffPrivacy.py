import random 
import numpy as np
from collections import OrderedDict 

class Diff_Privacy(object):

	epsilon = 0.0050
        
        @staticmethod
	def noise(sensitivity, num=1):
                #compute the sensitivity of a query using its output.
		scale = sensitivity/Diff_Privacy.epsilon
        	return abs(np.random.laplace(loc=0.0, scale=scale, size=num))



        #given two outputs of a Mechanism, compute the sensitivity 
        @staticmethod 
        def sensitivity(d1, d2):
                #print("d1:{}, d2:{}".format(d1,d2))
		total_sens = 0.
                if type(d1) is dict:
			for k,v in d1.items():
                                if k in d2:   
					total_sens += abs(d2[k]-v)
				else: total_sens += v 

                else: 
			total_sens = abs(d2-d1) 

                return total_sens
   

        @staticmethod
	def compute_sensitivity(func, d1, **kwargs):

		max_sens = 0.
                #print ("d1:{}".format(d1))
                inputs = kwargs['input']
                
                #remove one item at a time from input to see its effect on the output 
        	for i in range(len(inputs)):
                        #remove an item from input attr: values 
			inputs_cpy = inputs[:]
			inputs_cpy.pop(i)
                        #now make a copy of all input  
                        kwargs_cpy = kwargs.copy()
			kwargs_cpy['input'] = inputs_cpy

                        #compute the function now on new input   
			d2 = func(**kwargs_cpy)
                        
                        curr_sens = Diff_Privacy.sensitivity(d1,d2)
                        
			if curr_sens > max_sens: max_sens = curr_sens
		return max_sens	
		

def privatize(func):
	
	def func_wrapper(**kwargs):
      		res = func(**kwargs)
                sens = Diff_Privacy.compute_sensitivity(func, res, **kwargs)

                
                if type(res) is dict:
                        res = OrderedDict(sorted(res.items(), key=lambda t: t[1]))
                        noisy_res = res.copy()
                        noise_vec = Diff_Privacy.noise(sens, len(res.items()))
			noise_vec.sort()
			#noise_vec = sorted(noise_vec, key=int, reverse=True)		
                         #print ("noise vector:{}".format(noise_vec)) 
                        
                       	for ind, k in enumerate(noisy_res.keys()):
				 noisy_res[k] = noisy_res[k]+ noise_vec[ind]
                                
      		
                	#print("Original Result:{}, Noisy Results:{}".format(res, noisy_res))
        		return noisy_res
		else: 	
			noisy_res = res + Diff_Privacy.noise(sens)[0]
			#print("Original Result:{}, Noisy Results:{}".format(res, noisy_res))				 
                        return noisy_res
        return func_wrapper




@privatize
def average(lst):
	return sum(lst)/float(len(lst))



if __name__ == "__main__":
     #privacy settings. 
     Diff_Privacy.epsilon = 0.55

     #input attribute values that need to be protected
     lst = [1,2,3,4,5]    

     #find the average in diff: private way  		
     myres = average(lst)
     print("Diff: Private Result:{}".format(myres))
     


